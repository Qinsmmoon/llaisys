#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
    /**
     * CPU 采样实现。
     * @param token_ptr   输出 buffer（int64_t 类型，长度 1）
     * @param logits_ptr  输入 logits buffer
     * @param logits_dtype logits 的数据类型（支持 F32, BF16, F16）
     * @param vocab_size  词表大小
     * @param temperature 温度
     * @param top_k       top-k
     * @param top_p       top-p
     */
    void sample_cpu(std::byte* token_ptr,
                    const std::byte* logits_ptr,
                    llaisysDataType_t logits_dtype,
                    size_t vocab_size,
                    float temperature,
                    int top_k,
                    float top_p);
}