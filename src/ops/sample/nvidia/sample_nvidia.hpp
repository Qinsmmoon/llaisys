#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

/**
 * @brief GPU采样实现（使用CUDA + Thrust）
 * @param token_ptr   输出token的host指针（int64_t类型）
 * @param logits_ptr  输入logits的设备指针
 * @param logits_dtype logits数据类型
 * @param vocab_size  词表大小
 * @param temperature 温度参数（<=0时退化为argmax）
 * @param top_k       top-k参数（<=0或>=vocab_size时不限制）
 * @param top_p       top-p参数（0~1，>=1时不限制）
 * @param stream      CUDA流（默认0）
 */
void sample_nvidia(
    std::byte* token_ptr,
    const std::byte* logits_ptr,
    llaisysDataType_t logits_dtype,
    size_t vocab_size,
    float temperature,
    int top_k,
    float top_p,
    cudaStream_t stream = 0
);

} // namespace llaisys::ops::nvidia