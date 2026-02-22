#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cblas.h>      // OpenBLAS
#include <cstring>      // for memcpy
#include <vector>       // 用于临时缓冲区
#include <limits>
#include <stdexcept>
#include <thread> 
namespace llaisys::ops::cpu {

static void init_openblas() {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        
        // 获取 CPU 核心数
        int num_cores = std::thread::hardware_concurrency();
        
        // 设置 OpenBLAS 线程数
        openblas_set_num_threads(num_cores);
        
        // 打印诊断信息
        printf("OpenBLAS initialized with %d threads on %d cores\n", 
               openblas_get_num_threads(), openblas_get_num_procs());
        printf("OpenBLAS config: %s\n", openblas_get_config());
        printf("OpenBLAS core: %s\n", openblas_get_corename());
    }
}

// 直接使用 OpenBLAS 的 sgemm 处理 float 类型
void linear_f32(float *out, const float *in, const float *weight, const float *bias,
                size_t N, size_t D_in, size_t D_out) {
    // 防止 blasint 溢出（blasint 在某些实现中是 int32）
    if (N > static_cast<size_t>(std::numeric_limits<blasint>::max()) ||
        D_in > static_cast<size_t>(std::numeric_limits<blasint>::max()) ||
        D_out > static_cast<size_t>(std::numeric_limits<blasint>::max())) {
        throw std::runtime_error("matrix dimension too large for blasint");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 注意：对于 CblasRowMajor 且 TransA = NoTrans, TransB = Trans
    // A: M x K (N x D_in) -> lda = K = D_in
    // B: D_out x D_in 存储为 row-major，TransB = Trans -> ldb = K = D_in
    // C: M x N (N x D_out) -> ldc = Ncols = D_out
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<blasint>(N), static_cast<blasint>(D_out), static_cast<blasint>(D_in),
                alpha,
                in, static_cast<blasint>(D_in),        // lda = D_in
                weight, static_cast<blasint>(D_in),    // ldb = D_in  <-- 保持为 D_in
                beta,
                out, static_cast<blasint>(D_out));     // ldc = D_out

    // 添加偏置（如果有）
    if (bias) {
        for (size_t i = 0; i < N; ++i) {
            float *out_row = out + i * D_out;
            for (size_t j = 0; j < D_out; ++j) {
                out_row[j] += bias[j];
            }
        }
    }
}

// f16 / bf16 版保持不变（转换到 f32 -> sgemm -> 转回）
void linear_f16(llaisys::fp16_t *out, const llaisys::fp16_t *in,
                const llaisys::fp16_t *weight, const llaisys::fp16_t *bias,
                size_t N, size_t D_in, size_t D_out) {
    const size_t in_size = N * D_in;
    const size_t weight_size = D_out * D_in;
    const size_t out_size = N * D_out;

    std::vector<float> in_f32(in_size);
    std::vector<float> weight_f32(weight_size);
    std::vector<float> out_f32(out_size);
    std::vector<float> bias_f32;

    for (size_t i = 0; i < in_size; ++i) in_f32[i] = llaisys::utils::cast<float>(in[i]);
    for (size_t i = 0; i < weight_size; ++i) weight_f32[i] = llaisys::utils::cast<float>(weight[i]);

    float *bias_ptr = nullptr;
    if (bias) {
        bias_f32.resize(D_out);
        for (size_t j = 0; j < D_out; ++j) bias_f32[j] = llaisys::utils::cast<float>(bias[j]);
        bias_ptr = bias_f32.data();
    }

    linear_f32(out_f32.data(), in_f32.data(), weight_f32.data(), bias_ptr, N, D_in, D_out);

    for (size_t i = 0; i < out_size; ++i) out[i] = llaisys::utils::cast<llaisys::fp16_t>(out_f32[i]);
}

void linear_bf16(llaisys::bf16_t *out, const llaisys::bf16_t *in,
                 const llaisys::bf16_t *weight, const llaisys::bf16_t *bias,
                 size_t N, size_t D_in, size_t D_out) {
    const size_t in_size = N * D_in;
    const size_t weight_size = D_out * D_in;
    const size_t out_size = N * D_out;

    std::vector<float> in_f32(in_size);
    std::vector<float> weight_f32(weight_size);
    std::vector<float> out_f32(out_size);
    std::vector<float> bias_f32;

    for (size_t i = 0; i < in_size; ++i) in_f32[i] = llaisys::utils::cast<float>(in[i]);
    for (size_t i = 0; i < weight_size; ++i) weight_f32[i] = llaisys::utils::cast<float>(weight[i]);

    float *bias_ptr = nullptr;
    if (bias) {
        bias_f32.resize(D_out);
        for (size_t j = 0; j < D_out; ++j) bias_f32[j] = llaisys::utils::cast<float>(bias[j]);
        bias_ptr = bias_f32.data();
    }

    linear_f32(out_f32.data(), in_f32.data(), weight_f32.data(), bias_ptr, N, D_in, D_out);

    for (size_t i = 0; i < out_size; ++i) out[i] = llaisys::utils::cast<llaisys::bf16_t>(out_f32[i]);
}


void linear(std::byte *out_b, const std::byte *in_b, const std::byte *weight_b, const std::byte *bias_b,
            llaisysDataType_t dtype, size_t N, size_t D_in, size_t D_out) {   
                

    static bool first_call = true;
    if (first_call) {
        first_call = false;
        init_openblas();
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_f32(reinterpret_cast<float *>(out_b),
                   reinterpret_cast<const float *>(in_b),
                   reinterpret_cast<const float *>(weight_b),
                   bias_b ? reinterpret_cast<const float *>(bias_b) : nullptr,
                   N, D_in, D_out);
        break;

    case LLAISYS_DTYPE_F16:
        linear_f16(reinterpret_cast<llaisys::fp16_t *>(out_b),
                   reinterpret_cast<const llaisys::fp16_t *>(in_b),
                   reinterpret_cast<const llaisys::fp16_t *>(weight_b),
                   bias_b ? reinterpret_cast<const llaisys::fp16_t *>(bias_b) : nullptr,
                   N, D_in, D_out);
        break;

    case LLAISYS_DTYPE_BF16:
        linear_bf16(reinterpret_cast<llaisys::bf16_t *>(out_b),
                    reinterpret_cast<const llaisys::bf16_t *>(in_b),
                    reinterpret_cast<const llaisys::bf16_t *>(weight_b),
                    bias_b ? reinterpret_cast<const llaisys::bf16_t *>(bias_b) : nullptr,
                    N, D_in, D_out);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu