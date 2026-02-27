#include "rope_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../utils.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>

namespace llaisys::ops::nvidia {

// Kernel for float32
__global__ void rope_kernel_f32(float *out, const float *in, const int64_t *pos_ids,
                                const double *denom, size_t seqlen, size_t nhead,
                                size_t d, size_t half) {
    size_t bid = static_cast<size_t>(blockIdx.x);
    size_t i = bid / nhead;
    size_t h = bid % nhead;

    size_t tid = static_cast<size_t>(threadIdx.x);
    size_t base = (i * nhead + h) * d;

    int64_t p = pos_ids[i];
    for (size_t j = tid; j < half; j += blockDim.x) {
        float a = in[base + j];
        float b = in[base + half + j];
        // use double precision for phi and trig to match CPU implementation
        double phi = static_cast<double>(p) / denom[j];
        double c = cos(phi);
        double s = sin(phi);
        float oa = static_cast<float>(static_cast<double>(a) * c - static_cast<double>(b) * s);
        float ob = static_cast<float>(static_cast<double>(b) * c + static_cast<double>(a) * s);
        out[base + j] = oa;
        out[base + half + j] = ob;
    }
}

// Kernel for fp16 (__half)
__global__ void rope_kernel_f16(__half *out, const __half *in, const int64_t *pos_ids,
                                const double *denom, size_t seqlen, size_t nhead,
                                size_t d, size_t half) {
    size_t bid = static_cast<size_t>(blockIdx.x);
    size_t i = bid / nhead;
    size_t h = bid % nhead;

    size_t tid = static_cast<size_t>(threadIdx.x);
    size_t base = (i * nhead + h) * d;

    int64_t p = pos_ids[i];
    for (size_t j = tid; j < half; j += blockDim.x) {
        float a = __half2float(in[base + j]);
        float b = __half2float(in[base + half + j]);
        double phi = static_cast<double>(p) / denom[j];
        double c = cos(phi);
        double s = sin(phi);
        float oa = static_cast<float>(static_cast<double>(a) * c - static_cast<double>(b) * s);
        float ob = static_cast<float>(static_cast<double>(b) * c + static_cast<double>(a) * s);
        out[base + j] = __float2half(oa);
        out[base + half + j] = __float2half(ob);
    }
}

// Kernel for bf16 (__nv_bfloat16)
__global__ void rope_kernel_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, const int64_t *pos_ids,
                                 const double *denom, size_t seqlen, size_t nhead,
                                 size_t d, size_t half) {
    size_t bid = static_cast<size_t>(blockIdx.x);
    size_t i = bid / nhead;
    size_t h = bid % nhead;

    size_t tid = static_cast<size_t>(threadIdx.x);
    size_t base = (i * nhead + h) * d;

    int64_t p = pos_ids[i];
    for (size_t j = tid; j < half; j += blockDim.x) {
        float a = __bfloat162float(in[base + j]);
        float b = __bfloat162float(in[base + half + j]);
        double phi = static_cast<double>(p) / denom[j];
        double c = cos(phi);
        double s = sin(phi);
        float oa = static_cast<float>(static_cast<double>(a) * c - static_cast<double>(b) * s);
        float ob = static_cast<float>(static_cast<double>(b) * c + static_cast<double>(a) * s);
        out[base + j] = __float2bfloat16(oa);
        out[base + half + j] = __float2bfloat16(ob);
    }
}

void rope(std::byte *out_b, const std::byte *in_b, const std::byte *pos_b, llaisysDataType_t dtype,
          size_t seqlen, size_t nhead, size_t d, float theta) {
    ASSERT(seqlen > 0 && nhead > 0 && d > 0 && (d % 2 == 0), "Rope (GPU): invalid dimensions (d must be even)");

    const int threads = 256;
    const size_t half = d / 2;
    const size_t blocks = seqlen * nhead; // one block per (pos, head)

    // Precompute denom on host using double and copy to device as double
    std::vector<double> h_denom(half);
    for (size_t j = 0; j < half; ++j) {
        double exponent = 2.0 * static_cast<double>(j) / static_cast<double>(d);
        double val = std::pow(static_cast<double>(theta), exponent);
        h_denom[j] = val;
    }

    double *d_denom = nullptr;
    size_t denom_bytes = half * sizeof(double);
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_denom), denom_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc denom failed: ") + cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_denom, h_denom.data(), denom_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_denom);
        throw std::runtime_error(std::string("cudaMemcpy denom failed: ") + cudaGetErrorString(err));
    }

    // pos_b is device pointer to int64_t array
    const int64_t *d_pos = reinterpret_cast<const int64_t *>(pos_b);

    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float *out_f = reinterpret_cast<float *>(out_b);
            const float *in_f = reinterpret_cast<const float *>(in_b);
            rope_kernel_f32<<<static_cast<unsigned int>(blocks), threads>>>(out_f, in_f, d_pos, d_denom, seqlen, nhead, d, half);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            __half *out_h = reinterpret_cast<__half *>(out_b);
            const __half *in_h = reinterpret_cast<const __half *>(in_b);
            rope_kernel_f16<<<static_cast<unsigned int>(blocks), threads>>>(out_h, in_h, d_pos, d_denom, seqlen, nhead, d, half);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            __nv_bfloat16 *out_bf = reinterpret_cast<__nv_bfloat16 *>(out_b);
            const __nv_bfloat16 *in_bf = reinterpret_cast<const __nv_bfloat16 *>(in_b);
            rope_kernel_bf16<<<static_cast<unsigned int>(blocks), threads>>>(out_bf, in_bf, d_pos, d_denom, seqlen, nhead, d, half);
            break;
        }
        default:
            cudaFree(d_denom);
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    // check launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_denom);
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    // synchronize and check
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_denom);
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }

    cudaFree(d_denom);
}

} // namespace llaisys::ops::nvidia