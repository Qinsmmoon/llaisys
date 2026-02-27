#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        // 嵌入层权重 [vocab_size, hidden_size] = [151936, 1536]
        llaisysTensor_t in_embed;
        
        // 输出投影/LM头权重 [vocab_size, hidden_size] = [151936, 1536]
        // 注意：Qwen2中lm_head权重与in_embed不共享
        llaisysTensor_t out_embed;
        
        // 最终层归一化权重 [hidden_size] = [1536]
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        
        // 注意力前归一化权重 [hidden_size] = [1536] (每层)
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        
        // Q投影权重 [hidden_size, hidden_size] = [1536, 1536]
        // Q偏置 [hidden_size] = [1536] (可选，Qwen2有)
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        
        // K投影权重 [kv_dim, hidden_size] = [256, 1536]
        // 注意：由于GQA，K维度缩小为 hidden_size * (num_key_value_heads/num_attention_heads) = 1536 * (2/12) = 256
        // K偏置 [kv_dim] = [256]
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        
        // V投影权重 [kv_dim, hidden_size] = [256, 1536]
        // V偏置 [kv_dim] = [256]
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        
        // 输出投影权重 [hidden_size, hidden_size] = [1536, 1536]
        llaisysTensor_t *attn_o_w;
        
        // MLP前归一化权重 [hidden_size] = [1536] (每层)
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        
        // 门控投影权重 [intermediate_size, hidden_size] = [8960, 1536]
        // 注意：SwiGLU激活函数需要gate和up两个投影
        llaisysTensor_t *mlp_gate_w;
        
        // 上投影权重 [intermediate_size, hidden_size] = [8960, 1536]
        llaisysTensor_t *mlp_up_w;
        
        // 下投影权重 [hidden_size, intermediate_size] = [1536, 8960]
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model {
        void *model; 
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, 
        llaisysDeviceType_t device, 
        int *device_ids, 
        int ndevice
    );

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        const int64_t *token_ids,
        size_t ntoken,
        int64_t *out_ids,
        size_t out_capacity
    );
    
    __export int llaisysQwen2ModelLoadTensor(
        struct LlaisysQwen2Model *model,
        const char *name,
        const void *data,
        int ndim,
        const int64_t *shape
    );
}
#endif // LLAISYS_MODELS_QWEN2_H
