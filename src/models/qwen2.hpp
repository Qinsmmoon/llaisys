#pragma once
#ifndef LLAISYS_MODELS_QWEN2_HPP
#define LLAISYS_MODELS_QWEN2_HPP

#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace llaisys {
namespace models {

class Qwen2Model {
public:
    /**
     * @brief 构造函数：初始化模型配置与计算设备
     * @param meta 包含层数、头数等超参数的结构体
     * @param device 设备类型（CPU/GPU）
     * @param device_ids 物理设备 ID 列表
     */
    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    ~Qwen2Model();

    /**
     * @brief 推理接口：输入 Token 序列并生成下一个 Token
     * @param token_ids 输入的 Token ID 数组
     * @param ntoken 输入序列的长度
     * @param out_ids 输出缓冲区（存放生成的 Token）
     */
    int64_t infer(const int64_t *token_ids, size_t ntoken, int64_t *out_ids, size_t out_capacity,
              float temperature, int top_k, float top_p);

    /**
     * @brief 加载权重：将外部数据加载到指定的模型张量中
     */
    int load_tensor(const char *name, const void *data, size_t ndim, const int64_t *shape);

    // 内部辅助函数：在指定设备上创建张量
    tensor_t _create_tensor(const std::vector<size_t>& shape, llaisysDataType_t dtype);

private:
    LlaisysQwen2Meta _meta;           // 模型元数据（层数、维度等）
    llaisysDeviceType_t _device_type; // 计算设备类型
    int _device_id;                   // 设备索引

    // 权重结构体
    LlaisysQwen2Weights _cweights;

    // KV Cache 容器：每个元素对应一层的 K 或 V 张量
    // 布局通常为: [seq_len, nkv_heads, head_dim]
    std::vector<tensor_t> _k_cache, _v_cache;

    size_t _cache_seq_len; // 当前已缓存的序列长度，用于增量推理
    
    // 存储所有 Tensor 的智能指针，确保在推理生命周期内引用计数不为 0
    std::vector<tensor_t> _tensors;
};

} // namespace models
} // namespace llaisys

__C {
    struct LlaisysQwen2Model; // C 接口前向声明
}

#endif