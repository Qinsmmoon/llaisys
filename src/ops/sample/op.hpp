#pragma once
#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    /**
     * 从 logits 中采样一个 token，支持 temperature, top-k, top-p。
     * @param token   输出张量，1 个元素，类型 I64，必须位于 CPU。
     * @param logits  输入 logits，1D 张量，形状 [vocab_size]。
     * @param temperature  温度参数，<=0 时退化为 argmax。
     * @param top_k        top-k 采样参数（整数），<=0 或 >= vocab_size 表示不限制。
     * @param top_p        top-p 采样参数（0~1），>=1 表示不限制。
     */
    void sample(tensor_t token, tensor_t logits, float temperature, int top_k, float top_p);
}