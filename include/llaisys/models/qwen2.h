#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    /**
     * @brief Qwen2 模型超参数元数据
     * 存储模型结构相关的常量，用于初始化张量形状和分配内存。
     */
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype; // 数据类型 (如 FP16, BF16, INT8)
        size_t nlayer;           // Transformer 层数 (num_hidden_layers)
        size_t hs;               // 隐藏层维度 (hidden_size)
        size_t nh;               // 注意力头数 (num_attention_heads)
        size_t nkvh;             // KV 注意力头数 (num_key_value_heads, GQA 关键参数)
        size_t dh;               // 每个头的维度 (head_dim, 通常为 hs / nh)
        size_t di;               // MLP 中间层维度 (intermediate_size)
        size_t maxseq;           // 支持的最大序列长度 (max_position_embeddings)
        size_t voc;              // 词表大小 (vocab_size)
        float epsilon;           // RMSNorm 的微小偏置项 (layer_norm_epsilon)
        float theta;             // RoPE (旋转位置编码) 的底数 base
        int64_t end_token;       // 终止符 Token ID (EOS)
    };

    /**
     * @brief Qwen2 模型权重集合
     * 包含所有层所需的张量指针。
     */
    struct LlaisysQwen2Weights {
        /* --- 全局权重 --- */
        // 输入嵌入层 [vocab_size, hidden_size]
        llaisysTensor_t in_embed;
        
        // 输出投影层 (LM Head) [vocab_size, hidden_size]
        // Qwen2 默认不使用 Weight Tying (权重共享)，故此处独立定义
        llaisysTensor_t out_embed;
        
        // Transformer 最后一层后的归一化权重 [hidden_size]
        llaisysTensor_t out_norm_w;

        /* --- 逐层权重 (指针数组，长度均为 nlayer) --- */
        
        // Attention 前置 RMSNorm 权重
        llaisysTensor_t *attn_norm_w; 
        
        // Self-Attention 投影权重与偏置
        // Qwen2 包含 Q/K/V 的偏置项，这与 Llama 标准实现有所不同
        llaisysTensor_t *attn_q_w; // [hidden_size, hidden_size]
        llaisysTensor_t *attn_q_b; // [hidden_size]
        
        // K/V 投影 (注意：在 GQA 下维度为 [nkvh * dh, hidden_size])
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        
        // Attention 输出投影 (O) [hidden_size, hidden_size]
        llaisysTensor_t *attn_o_w;
        
        // MLP 前置 RMSNorm 权重 (通常位于 Attention 残差连接之后)
        llaisysTensor_t *mlp_norm_w;
        
        // SwiGLU MLP 层权重
        // 运算过程：down(silu(gate(x)) * up(x))
        llaisysTensor_t *mlp_gate_w; // [intermediate_size, hidden_size]
        llaisysTensor_t *mlp_up_w;   // [intermediate_size, hidden_size]
        llaisysTensor_t *mlp_down_w; // [hidden_size, intermediate_size]
    };

    /**
     * @brief Qwen2 模型实例句柄
     */
    struct LlaisysQwen2Model {
        void *model; // 内部实现结构的封装指针
    };

    /**
     * @brief 创建模型实例
     * @param meta 模型结构配置
     * @param device 运行设备类型 (CPU/GPU)
     * @param device_ids 具体的设备 ID 列表 (用于多卡并行)
     * @param ndevice 设备数量
     * @return 成功返回模型指针，失败返回 NULL
     */
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, 
        llaisysDeviceType_t device, 
        int *device_ids, 
        int ndevice
    );

    /**
     * @brief 销毁模型实例并释放内存
     */
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    /**
     * @brief 获取模型的权重结构
     */
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    /**
     * @brief 执行模型推理 (生成)
     * @param model 模型句柄
     * @param token_ids 输入 Token 序列
     * @param ntoken 输入序列长度
     * @param out_ids 输出 Token 缓冲区
     * @param out_capacity 缓冲区最大容量
     */
    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        const int64_t *token_ids,
        size_t ntoken,
        int64_t *out_ids,
        size_t out_capacity,
        float temperature,
        int top_k,
        float top_p
    );
    
    /**
     * @brief 向模型加载特定名称的张量权重
     * @param name 张量在模型中的全路径名 (例如 "model.layers.0.self_attn.q_proj.weight")
     * @param data 原始权重数据指针
     * @param ndim 维度数量
     * @param shape 维度数组
     */
    __export int llaisysQwen2ModelLoadTensor(
        struct LlaisysQwen2Model *model,
        const char *name,
        const void *data,
        int ndim,
        const int64_t *shape
    );
}
#endif // LLAISYS_MODELS_QWEN2_H
