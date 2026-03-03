#include "sample_cpu.hpp"
#include "../../../utils.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <cstring>

namespace llaisys::ops::cpu {

// 辅助：将不同数据类型的 logits 转换为 float 向量
static std::vector<float> logits_to_float(const std::byte* data, llaisysDataType_t dtype, size_t n) {
    std::vector<float> result(n);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float* src = reinterpret_cast<const float*>(data);
        std::copy(src, src + n, result.begin());
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t* src = reinterpret_cast<const bf16_t*>(data);
        for (size_t i = 0; i < n; ++i)
            result[i] = llaisys::utils::cast<float>(src[i]);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t* src = reinterpret_cast<const fp16_t*>(data);
        for (size_t i = 0; i < n; ++i)
            result[i] = llaisys::utils::cast<float>(src[i]);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    return result;
}

void sample_cpu(std::byte* token_ptr,
                const std::byte* logits_ptr,
                llaisysDataType_t logits_dtype,
                size_t vocab_size,
                float temperature,
                int top_k,
                float top_p) {
    // 1. 将 logits 转为 float
    std::vector<float> logits = logits_to_float(logits_ptr, logits_dtype, vocab_size);

    // 2. 处理温度（如果 temperature <= 0，退化为 argmax）
    if (temperature <= 0.0f) {
        size_t max_idx = 0;
        float max_val = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        int64_t* token_out = reinterpret_cast<int64_t*>(token_ptr);
        *token_out = static_cast<int64_t>(max_idx);
        return;
    }

    // 温度缩放
    float inv_temp = 1.0f / temperature;
    for (float& v : logits) {
        v *= inv_temp;
    }

    // 3. top-k 过滤
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
        // 使用 nth_element 找到第 k 大的值
        std::vector<float> copy = logits;  // 需要保留原始顺序，所以拷贝一份进行部分排序
        std::nth_element(copy.begin(), copy.begin() + top_k - 1, copy.end(), std::greater<float>());
        float threshold = copy[top_k - 1];
        for (float& v : logits) {
            if (v < threshold) {
                v = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // 4. softmax 得到概率
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (float& p : probs) {
        p *= inv_sum;
    }

    // 5. top-p (nucleus) 过滤
    if (top_p > 0.0f && top_p < 1.0f - 1e-6f) {
        // 创建索引并按概率降序排序
        std::vector<size_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });

        float cum = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            cum += probs[indices[i]];
            if (cum > top_p) {
                cutoff = i + 1;  // 保留前 cutoff 个 token
                break;
            }
        }
        if (cutoff == 0) cutoff = vocab_size;  // 未达到阈值则全部保留

        // 将 cutoff 之后的概率置零
        std::vector<bool> keep(vocab_size, false);
        for (size_t i = 0; i < cutoff; ++i) {
            keep[indices[i]] = true;
        }
        float sum_keep = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!keep[i]) probs[i] = 0.0f;
            else sum_keep += probs[i];
        }
        // 重新归一化
        if (sum_keep > 0.0f) {
            float inv_sum_keep = 1.0f / sum_keep;
            for (float& p : probs) p *= inv_sum_keep;
        } else {
            // 极罕见情况：所有概率都被滤除，退化为均匀分布（可替换为 argmax）
            std::fill(probs.begin(), probs.end(), 1.0f / vocab_size);
        }
    }

    // 6. 从概率分布中采样
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    size_t sampled_idx = dist(gen);

    // 7. 写入输出
    int64_t* token_out = reinterpret_cast<int64_t*>(token_ptr);
    *token_out = static_cast<int64_t>(sampled_idx);
}

} // namespace llaisys::ops::cpu