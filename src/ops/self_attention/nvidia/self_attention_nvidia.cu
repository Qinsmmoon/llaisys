#include "self_attention_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::nvidia {

// Helper: reduction in shared memory to compute max / sum for block
__inline__ __device__ double block_reduce_max(double val) {
    extern __shared__ double sdata[]; // size: blockDim.x
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            double other = sdata[tid + offset];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    return sdata[0];
}

__inline__ __device__ double block_reduce_sum(double val) {
    extern __shared__ double sdata_sum[]; // reused same shared mem region
    int tid = threadIdx.x;
    sdata_sum[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata_sum[tid] += sdata_sum[tid + offset];
        }
        __syncthreads();
    }
    return sdata_sum[0];
}

// F32 kernel
__global__ void self_attn_kernel_f32(float *out, const float *q, const float *k, const float *v,
                                    float scale,
                                    size_t seqlen, size_t nhead, size_t d,
                                    size_t total_len, size_t nkvhead, size_t dv,
                                    ptrdiff_t offset) {
    // Each block handles one (t, h)
    size_t bid = static_cast<size_t>(blockIdx.x);
    size_t t = bid / nhead;
    size_t h = bid % nhead;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // pointers
    const float* q_ptr = q + (t * nhead + h) * d; // q_row_stride = nhead * d
    // k_row_stride = nkvhead * d
    // v_row_stride = nkvhead * dv

    // 1) compute max score across allowed keys
    double local_max = -INFINITY;
    for (size_t ki = tid; ki < total_len; ki += nthreads) {
        if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
        size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
        const float* k_ptr = k + (ki * nkvhead + k_head) * d;

        double acc = 0.0;
        for (size_t c = 0; c < d; ++c) {
            acc += static_cast<double>(q_ptr[c]) * static_cast<double>(k_ptr[c]);
        }
        acc *= static_cast<double>(scale);
        if (acc > local_max) local_max = acc;
    }

    // reduce to block max
    double block_max = block_reduce_max(local_max);
    __syncthreads();

    // 2) compute sum_exp
    double local_sum = 0.0;
    for (size_t ki = tid; ki < total_len; ki += nthreads) {
        if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
        size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
        const float* k_ptr = k + (ki * nkvhead + k_head) * d;

        double acc = 0.0;
        for (size_t c = 0; c < d; ++c) {
            acc += static_cast<double>(q_ptr[c]) * static_cast<double>(k_ptr[c]);
        }
        acc *= static_cast<double>(scale);
        double e = std::exp(acc - block_max);
        local_sum += e;
    }

    double sum_exp = block_reduce_sum(local_sum);
    __syncthreads();

    // 3) compute weighted sum over v for each output dimension
    // Each thread handles multiple m in [0, dv)
    for (size_t m = tid; m < dv; m += nthreads) {
        double accv = 0.0;
        if (sum_exp != 0.0) {
            for (size_t ki = 0; ki < total_len; ++ki) {
                if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
                size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
                const float* k_ptr = k + (ki * nkvhead + k_head) * d;
                const float* v_ptr = v + (ki * nkvhead + k_head) * dv;

                double acc = 0.0;
                for (size_t c = 0; c < d; ++c) {
                    acc += static_cast<double>(q_ptr[c]) * static_cast<double>(k_ptr[c]);
                }
                acc *= static_cast<double>(scale);
                double w = std::exp(acc - block_max) / sum_exp;
                accv += w * static_cast<double>(v_ptr[m]);
            }
        }
        out[(t * nhead + h) * dv + m] = static_cast<float>(accv);
    }
}

// F16 kernel
__global__ void self_attn_kernel_f16(__half *out, const __half *q, const __half *k, const __half *v,
                                    float scale,
                                    size_t seqlen, size_t nhead, size_t d,
                                    size_t total_len, size_t nkvhead, size_t dv,
                                    ptrdiff_t offset) {
    size_t bid = static_cast<size_t>(blockIdx.x);
    size_t t = bid / nhead;
    size_t h = bid % nhead;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const __half* q_ptr = q + (t * nhead + h) * d;

    double local_max = -INFINITY;
    for (size_t ki = tid; ki < total_len; ki += nthreads) {
        if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
        size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
        const __half* k_ptr = k + (ki * nkvhead + k_head) * d;

        double acc = 0.0;
        for (size_t c = 0; c < d; ++c) {
            float qv = __half2float(q_ptr[c]);
            float kv = __half2float(k_ptr[c]);
            acc += static_cast<double>(qv) * static_cast<double>(kv);
        }
        acc *= static_cast<double>(scale);
        if (acc > local_max) local_max = acc;
    }

    double block_max = block_reduce_max(local_max);
    __syncthreads();

    double local_sum = 0.0;
    for (size_t ki = tid; ki < total_len; ki += nthreads) {
        if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
        size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
        const __half* k_ptr = k + (ki * nkvhead + k_head) * d;

        double acc = 0.0;
        for (size_t c = 0; c < d; ++c) {
            float qv = __half2float(q_ptr[c]);
            float kv = __half2float(k_ptr[c]);
            acc += static_cast<double>(qv) * static_cast<double>(kv);
        }
        acc *= static_cast<double>(scale);
        double e = std::exp(acc - block_max);
        local_sum += e;
    }

    double sum_exp = block_reduce_sum(local_sum);
    __syncthreads();

    for (size_t m = tid; m < dv; m += nthreads) {
        double accv = 0.0;
        if (sum_exp != 0.0) {
            for (size_t ki = 0; ki < total_len; ++ki) {
                if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
                size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
                const __half* k_ptr = k + (ki * nkvhead + k_head) * d;
                const __half* v_ptr = v + (ki * nkvhead + k_head) * dv;

                double acc = 0.0;
                for (size_t c = 0; c < d; ++c) {
                    float qv = __half2float(q_ptr[c]);
                    float kv = __half2float(k_ptr[c]);
                    acc += static_cast<double>(qv) * static_cast<double>(kv);
                }
                acc *= static_cast<double>(scale);
                double w = std::exp(acc - block_max) / sum_exp;
                accv += w * static_cast<double>(__half2float(v_ptr[m]));
            }
        }
        out[(t * nhead + h) * dv + m] = __float2half(static_cast<float>(accv));
    }
}

// BF16 kernel
__global__ void self_attn_kernel_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                                     float scale,
                                     size_t seqlen, size_t nhead, size_t d,
                                     size_t total_len, size_t nkvhead, size_t dv,
                                     ptrdiff_t offset) {
    size_t bid = static_cast<size_t>(blockIdx.x);
    size_t t = bid / nhead;
    size_t h = bid % nhead;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const __nv_bfloat16* q_ptr = q + (t * nhead + h) * d;

    double local_max = -INFINITY;
    for (size_t ki = tid; ki < total_len; ki += nthreads) {
        if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
        size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
        const __nv_bfloat16* k_ptr = k + (ki * nkvhead + k_head) * d;

        double acc = 0.0;
        for (size_t c = 0; c < d; ++c) {
            float qv = __bfloat162float(q_ptr[c]);
            float kv = __bfloat162float(k_ptr[c]);
            acc += static_cast<double>(qv) * static_cast<double>(kv);
        }
        acc *= static_cast<double>(scale);
        if (acc > local_max) local_max = acc;
    }

    double block_max = block_reduce_max(local_max);
    __syncthreads();

    double local_sum = 0.0;
    for (size_t ki = tid; ki < total_len; ki += nthreads) {
        if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
        size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
        const __nv_bfloat16* k_ptr = k + (ki * nkvhead + k_head) * d;

        double acc = 0.0;
        for (size_t c = 0; c < d; ++c) {
            float qv = __bfloat162float(q_ptr[c]);
            float kv = __bfloat162float(k_ptr[c]);
            acc += static_cast<double>(qv) * static_cast<double>(kv);
        }
        acc *= static_cast<double>(scale);
        double e = std::exp(acc - block_max);
        local_sum += e;
    }

    double sum_exp = block_reduce_sum(local_sum);
    __syncthreads();

    for (size_t m = tid; m < dv; m += nthreads) {
        double accv = 0.0;
        if (sum_exp != 0.0) {
            for (size_t ki = 0; ki < total_len; ++ki) {
                if (static_cast<ptrdiff_t>(ki) > offset + static_cast<ptrdiff_t>(t)) continue;
                size_t k_head = (nkvhead == nhead) ? h : (h / (nhead / nkvhead));
                const __nv_bfloat16* k_ptr = k + (ki * nkvhead + k_head) * d;
                const __nv_bfloat16* v_ptr = v + (ki * nkvhead + k_head) * dv;

                double acc = 0.0;
                for (size_t c = 0; c < d; ++c) {
                    float qv = __bfloat162float(q_ptr[c]);
                    float kv = __bfloat162float(k_ptr[c]);
                    acc += static_cast<double>(qv) * static_cast<double>(kv);
                }
                acc *= static_cast<double>(scale);
                double w = std::exp(acc - block_max) / sum_exp;
                accv += w * static_cast<double>(__bfloat162float(v_ptr[m]));
            }
        }
        out[(t * nhead + h) * dv + m] = __float2bfloat16(static_cast<float>(accv));
    }
}

void self_attention(std::byte *attn_val_b, const std::byte *q_b, const std::byte *k_b, const std::byte *v_b,
                    llaisysDataType_t dtype, float scale,
                    size_t seqlen, size_t nhead, size_t d,
                    size_t total_len, size_t nkvhead, size_t dv) {
    if (nkvhead != nhead) {
        ASSERT((nhead % nkvhead) == 0, "self_attention (nvidia): nhead must be a multiple of nkvhead when repeating heads.");
    }

    // one block per (t, h)
    size_t blocks = seqlen * nhead;
    const int threads = 256;
    // shared memory used for reduction (block_reduce_x) requires blockDim.x * sizeof(double)
    size_t shared_bytes = threads * sizeof(double);

    ptrdiff_t offset = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen);

    cudaError_t err = cudaSuccess;

    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        float* out_f = reinterpret_cast<float*>(attn_val_b);
        const float* q_f = reinterpret_cast<const float*>(q_b);
        const float* k_f = reinterpret_cast<const float*>(k_b);
        const float* v_f = reinterpret_cast<const float*>(v_b);
        self_attn_kernel_f32<<<static_cast<unsigned int>(blocks), threads, shared_bytes>>>(out_f, q_f, k_f, v_f,
                                                                                          scale, seqlen, nhead, d,
                                                                                          total_len, nkvhead, dv, offset);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        __half* out_h = reinterpret_cast<__half*>(attn_val_b);
        const __half* q_h = reinterpret_cast<const __half*>(q_b);
        const __half* k_h = reinterpret_cast<const __half*>(k_b);
        const __half* v_h = reinterpret_cast<const __half*>(v_b);
        self_attn_kernel_f16<<<static_cast<unsigned int>(blocks), threads, shared_bytes>>>(out_h, q_h, k_h, v_h,
                                                                                          scale, seqlen, nhead, d,
                                                                                          total_len, nkvhead, dv, offset);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        __nv_bfloat16* out_bf = reinterpret_cast<__nv_bfloat16*>(attn_val_b);
        const __nv_bfloat16* q_bf = reinterpret_cast<const __nv_bfloat16*>(q_b);
        const __nv_bfloat16* k_bf = reinterpret_cast<const __nv_bfloat16*>(k_b);
        const __nv_bfloat16* v_bf = reinterpret_cast<const __nv_bfloat16*>(v_b);
        self_attn_kernel_bf16<<<static_cast<unsigned int>(blocks), threads, shared_bytes>>>(out_bf, q_bf, k_bf, v_bf,
                                                                                             scale, seqlen, nhead, d,
                                                                                             total_len, nkvhead, dv, offset);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia