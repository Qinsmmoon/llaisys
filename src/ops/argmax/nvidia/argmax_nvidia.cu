#include "argmax_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>

#include "../../../utils.hpp" // 视工程结构，确保包含路径正确（提供了 types.hpp / check.hpp）

namespace llaisys::ops::nvidia {

static inline int next_power_of_two(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

// Device helpers for conversion between 16-bit formats and float
__device__ inline float bf16_bits_to_float(uint16_t bits) {
    // bf16: store in high 16 bits of float32
    uint32_t bits32 = static_cast<uint32_t>(bits) << 16;
    return __int_as_float(static_cast<int>(bits32));
}

__device__ inline uint16_t float_to_bf16_bits(float v) {
    uint32_t bits32 = static_cast<uint32_t>(__float_as_int(v));
    const uint32_t rounding_bias = 0x00007FFFu + ((bits32 >> 16) & 1u);
    uint16_t bf = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return bf;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
// we can use __half intrinsics
__device__ inline float fp16_bits_to_float(uint16_t bits) {
    // reinterpret bits as __half
    __half h = *reinterpret_cast<const __half *>(&bits);
    return __half2float(h);
}

__device__ inline uint16_t float_to_fp16_bits(float v) {
    __half h = __float2half_rn(v);
    uint16_t out = *reinterpret_cast<uint16_t *>(&h);
    return out;
}
#else
// Fallback software conversion (rarely needed on modern nvcc, but keep for completeness)
__device__ inline float fp16_bits_to_float(uint16_t h) {
    // software conversion: expand fp16 to float32
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) f32 = sign | 0x7F800000u | (mantissa << 13);
        else f32 = sign | 0x7F800000u;
    } else if (exponent == 0) {
        if (mantissa == 0) f32 = sign;
        else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    return __int_as_float(static_cast<int>(f32));
}

__device__ inline uint16_t float_to_fp16_bits(float val) {
    uint32_t f32 = static_cast<uint32_t>(__float_as_int(val));
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFFu;
    uint16_t sign = static_cast<uint16_t>((f32 >> 16) & 0x8000u);
    if (exponent >= 16) {
        // NaN or Inf
        if (exponent == 128 && mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7E00u);
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    } else if (exponent >= -14) {
        return static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13));
    } else if (exponent >= -24) {
        mantissa |= 0x800000u;
        mantissa >>= (-14 - exponent);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else {
        return sign;
    }
}
#endif

// Kernel: each block reduces a chunk and writes one block result
template <int BLOCK_SIZE>
__global__ void argmax_block_reduce_kernel(const unsigned char *vals_bytes, size_t numel,
                                           llaisysDataType_t val_type,
                                           float *out_block_vals, unsigned long long *out_block_idxs) {
    const size_t tid = threadIdx.x;
    const size_t bid = blockIdx.x;
    const size_t bstride = blockDim.x * gridDim.x;
    const size_t start = bid * blockDim.x + tid;

    float local_max = -INFINITY;
    unsigned long long local_idx = 0;
    bool has_value = false;

    // compute step to traverse elements: stride = blockDim.x * gridDim.x
    for (size_t i = start; i < numel; i += bstride) {
        float v = 0.0f;
        // load value according to val_type
        if (val_type == LLAISYS_DTYPE_F32) {
            const float *vals = reinterpret_cast<const float *>(vals_bytes);
            v = vals[i];
        } else if (val_type == LLAISYS_DTYPE_F16) {
            const uint16_t *vals = reinterpret_cast<const uint16_t *>(vals_bytes);
            v = fp16_bits_to_float(vals[i]);
        } else if (val_type == LLAISYS_DTYPE_BF16) {
            const uint16_t *vals = reinterpret_cast<const uint16_t *>(vals_bytes);
            v = bf16_bits_to_float(vals[i]);
        } else {
            // Unsupported types handled on host; kernel should only be launched for supported types
            v = -INFINITY;
        }

        if (!has_value || v > local_max) {
            local_max = v;
            local_idx = i;
            has_value = true;
        }
    }

    // Shared mem reduction
    extern __shared__ unsigned char shared_mem[]; // dynamic
    float *svals = reinterpret_cast<float *>(shared_mem); // BLOCK_SIZE floats
    unsigned long long *sidx = reinterpret_cast<unsigned long long *>(shared_mem + BLOCK_SIZE * sizeof(float)); // BLOCK_SIZE ull

    svals[tid] = has_value ? local_max : -INFINITY;
    sidx[tid] = local_idx;

    __syncthreads();

    // reduce in-place (max with index)
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float a = svals[tid];
            float b = svals[tid + offset];
            unsigned long long bi = sidx[tid + offset];
            if (b > a) {
                svals[tid] = b;
                sidx[tid] = bi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_block_vals[bid] = svals[0];
        out_block_idxs[bid] = sidx[0];
    }
}

// Final reduction kernel over block results (assume num_blocks <= 65536)
template <int BLOCK_SIZE>
__global__ void final_reduce_kernel(const float *block_vals, const unsigned long long *block_idxs, int num_blocks,
                                    float *out_val, unsigned long long *out_idx) {
    const int tid = threadIdx.x;
    float local_max = -INFINITY;
    unsigned long long local_idx = 0;
    bool has_value = false;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        if (!has_value || v > local_max) {
            local_max = v;
            local_idx = block_idxs[i];
            has_value = true;
        }
    }

    extern __shared__ char smem[];
    float *svals = reinterpret_cast<float *>(smem);
    unsigned long long *sidx = reinterpret_cast<unsigned long long *>(smem + BLOCK_SIZE * sizeof(float));

    svals[tid] = has_value ? local_max : -INFINITY;
    sidx[tid] = local_idx;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float a = svals[tid];
            float b = svals[tid + offset];
            unsigned long long bi = sidx[tid + offset];
            if (b > a) {
                svals[tid] = b;
                sidx[tid] = bi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_val = svals[0];
        *out_idx = sidx[0];
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t val_type,
            llaisysDataType_t idx_type, size_t numel) {
    ASSERT(numel > 0, "Argmax (CUDA): vals must contain at least one element.");

    // Only support F32, F16, BF16 here
    if (!(val_type == LLAISYS_DTYPE_F32 || val_type == LLAISYS_DTYPE_F16 || val_type == LLAISYS_DTYPE_BF16)) {
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }

    // Determine launch config
    const int MAX_BLOCKS = 1024;
    const int BLOCK_SIZE = 256;
    int num_blocks = std::min(MAX_BLOCKS, (int)((numel + BLOCK_SIZE - 1) / BLOCK_SIZE));
    if (num_blocks <= 0) num_blocks = 1;

    // Allocate device scratch buffers for per-block results
    float *d_block_vals = nullptr;
    unsigned long long *d_block_idxs = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_block_vals, sizeof(float) * num_blocks);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed for block_vals: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed");
    }
    err = cudaMalloc(&d_block_idxs, sizeof(unsigned long long) * num_blocks);
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        std::cerr << "[ERROR] cudaMalloc failed for block_idxs: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed");
    }

    // Launch first kernel
    size_t shared_bytes = BLOCK_SIZE * sizeof(float) + BLOCK_SIZE * sizeof(unsigned long long);
    const unsigned char *vals_bytes = reinterpret_cast<const unsigned char *>(vals);

    // instantiate kernel
    argmax_block_reduce_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, shared_bytes>>>(vals_bytes, numel, val_type, d_block_vals, d_block_idxs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cuda kernel launch failed");
    }

    // Final reduction kernel: we reduce num_blocks elements to 1
    float *d_out_val = nullptr;
    unsigned long long *d_out_idx = nullptr;
    err = cudaMalloc(&d_out_val, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
        std::cerr << "[ERROR] cudaMalloc failed for out_val: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed");
    }
    err = cudaMalloc(&d_out_idx, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
        cudaFree(d_out_val);
        std::cerr << "[ERROR] cudaMalloc failed for out_idx: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed");
    }

    // final kernel uses one block (or maybe small number) with BLOCK_SIZE threads
    size_t final_shared_bytes = BLOCK_SIZE * sizeof(float) + BLOCK_SIZE * sizeof(unsigned long long);
    final_reduce_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, final_shared_bytes>>>(d_block_vals, d_block_idxs, num_blocks, d_out_val, d_out_idx);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
        cudaFree(d_out_val);
        cudaFree(d_out_idx);
        std::cerr << "[ERROR] CUDA final kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cuda kernel launch failed");
    }

    // Now d_out_val (float) and d_out_idx (unsigned long long) hold the final results.
    // We need to write them into max_val and max_idx with appropriate types.
    // We'll copy them to host temporaries then write into device destination appropriately.
    float h_max_val_f = 0.0f;
    unsigned long long h_max_idx = 0;

    err = cudaMemcpy(&h_max_val_f, d_out_val, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
        cudaFree(d_out_val);
        cudaFree(d_out_idx);
        std::cerr << "[ERROR] cudaMemcpy D->H failed for out_val: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed");
    }
    err = cudaMemcpy(&h_max_idx, d_out_idx, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
        cudaFree(d_out_val);
        cudaFree(d_out_idx);
        std::cerr << "[ERROR] cudaMemcpy D->H failed for out_idx: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed");
    }

    // free scratch
    cudaFree(d_block_vals);
    cudaFree(d_block_idxs);
    cudaFree(d_out_val);
    cudaFree(d_out_idx);

    // Write max_val back to device memory max_val according to val_type
    if (val_type == LLAISYS_DTYPE_F32) {
        // max_val points to device float memory
        cudaMemcpy(max_val, &h_max_val_f, sizeof(float), cudaMemcpyHostToDevice);
    } else if (val_type == LLAISYS_DTYPE_F16) {
        uint16_t bits;
        // Use host conversion from llaisys::utils::cast
        llaisys::fp16_t hf = llaisys::utils::cast<llaisys::fp16_t>(h_max_val_f); // should resolve via utils.hpp
        bits = hf._v;
        cudaMemcpy(max_val, &bits, sizeof(uint16_t), cudaMemcpyHostToDevice);
    } else if (val_type == LLAISYS_DTYPE_BF16) {
        uint16_t bits = llaisys::utils::cast<llaisys::bf16_t>(h_max_val_f)._v;
        cudaMemcpy(max_val, &bits, sizeof(uint16_t), cudaMemcpyHostToDevice);
    } else {
        // shouldn't happen due to check above
        throw std::runtime_error("Unsupported dtype in CUDA argmax");
    }

    // write index based on idx_type
    if (idx_type == LLAISYS_DTYPE_I64) {
        int64_t idx64 = static_cast<int64_t>(h_max_idx);
        cudaMemcpy(max_idx, &idx64, sizeof(int64_t), cudaMemcpyHostToDevice);
    } else if (idx_type == LLAISYS_DTYPE_I32) {
        int32_t idx32 = static_cast<int32_t>(h_max_idx);
        cudaMemcpy(max_idx, &idx32, sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
    }

    // done
}

} // namespace llaisys::ops::nvidia