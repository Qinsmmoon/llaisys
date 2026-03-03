// qwen2.cpp
#include "qwen2.hpp"
#include "../core/context/context.hpp"
#include "../tensor/tensor.hpp"
#include <vector>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"

using namespace llaisys;
using namespace llaisys::models;
using namespace llaisys::ops;
using namespace llaisys::utils;
using namespace std;

// 辅助函数：将int64_t形状转换为size_t
std::vector<size_t> int64_to_size_t(const int64_t* shape, size_t ndim) {
    std::vector<size_t> result(ndim);
    for (size_t i = 0; i < ndim; i++) {
        result[i] = static_cast<size_t>(shape[i]);
    }
    return result;
}

// 创建包装tensor的LlaisysTensor结构体
llaisysTensor_t create_llaisys_tensor(const tensor_t& tensor) {
    llaisysTensor_t c_tensor = new LlaisysTensor();
    c_tensor->tensor = tensor;
    return c_tensor;
}

// 获取tensor的原始指针
tensor_t get_tensor_from_ptr(llaisysTensor_t c_tensor) {
    if (!c_tensor) return nullptr;
    return c_tensor->tensor;
}

// 辅助函数：创建用于推理的临时tensor
tensor_t Qwen2Model::_create_tensor(const std::vector<size_t>& shape, llaisysDataType_t dtype) {
    return Tensor::create(shape, dtype, _device_type, _device_id);
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (meta == nullptr) {
        throw std::invalid_argument("meta cannot be null");
    }
    
    // 复制元数据
    _meta = *meta;
    _device_type = device;
    _device_id = (ndevice > 0) ? device_ids[0] : 0;
    _cache_seq_len = 0;
    
    // 初始化 C 权重结构
    memset(&_cweights, 0, sizeof(LlaisysQwen2Weights));
    
    // 为每一层分配指针数组
    size_t nlayer = meta->nlayer;
    
    _cweights.attn_norm_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.mlp_norm_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_q_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_q_b = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_k_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_k_b = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_v_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_v_b = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.attn_o_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.mlp_gate_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.mlp_up_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    _cweights.mlp_down_w = (llaisysTensor_t*)calloc(nlayer, sizeof(llaisysTensor_t));
    
    // 初始化 KV 缓存 (每层的 entry 默认为 nullptr)
    _k_cache.resize(nlayer);
    _v_cache.resize(nlayer);
    for (size_t i = 0; i < nlayer; ++i) {
        _k_cache[i] = nullptr;
        _v_cache[i] = nullptr;
    }
    
    // 初始化内部的tensor存储
    _tensors.clear();
    
    std::cout << "Qwen2Model created with " << nlayer << " layers" << std::endl;
}

Qwen2Model::~Qwen2Model() {
    std::cout << "Qwen2Model destructor called" << std::endl;
    
    // 释放所有LlaisysTensor对象
    auto free_tensor_array = [](llaisysTensor_t* arr, size_t n) {
        if (arr) {
            for (size_t i = 0; i < n; i++) {
                if (arr[i]) {
                    delete arr[i];
                }
            }
            free(arr);
        }
    };
    
    // 释放单个tensor
    auto free_single_tensor = [](llaisysTensor_t tensor) {
        if (tensor) {
            delete tensor;
        }
    };
    
    free_single_tensor(_cweights.in_embed);
    free_single_tensor(_cweights.out_embed);
    free_single_tensor(_cweights.out_norm_w);
    
    size_t nlayer = _meta.nlayer;
    free_tensor_array(_cweights.attn_norm_w, nlayer);
    free_tensor_array(_cweights.mlp_norm_w, nlayer);
    free_tensor_array(_cweights.attn_q_w, nlayer);
    free_tensor_array(_cweights.attn_q_b, nlayer);
    free_tensor_array(_cweights.attn_k_w, nlayer);
    free_tensor_array(_cweights.attn_k_b, nlayer);
    free_tensor_array(_cweights.attn_v_w, nlayer);
    free_tensor_array(_cweights.attn_v_b, nlayer);
    free_tensor_array(_cweights.attn_o_w, nlayer);
    free_tensor_array(_cweights.mlp_gate_w, nlayer);
    free_tensor_array(_cweights.mlp_up_w, nlayer);
    free_tensor_array(_cweights.mlp_down_w, nlayer);
    
    // 清理缓存和内部存储
    _k_cache.clear();
    _v_cache.clear();
    _tensors.clear();
}

int Qwen2Model::load_tensor(const char *name, const void *data, size_t ndim, const int64_t *shape) {
    if (!name || !data || ndim == 0 || !shape) {
        std::cerr << "Invalid parameters for load_tensor" << std::endl;
        return -1;
    }
    
    std::string tensor_name(name);
    
    try {
        // 解析层索引
        int layer_idx = -1;
        if (tensor_name.find("model.layers.") != std::string::npos) {
            size_t start = tensor_name.find("model.layers.") + 13;
            size_t end = tensor_name.find(".", start);
            if (end != std::string::npos) {
                layer_idx = std::stoi(tensor_name.substr(start, end - start));
            }
        }
        
        // 转换形状为 size_t 向量
        std::vector<size_t> shape_vec = int64_to_size_t(shape, ndim);
        
        // 创建tensor对象
        tensor_t tensor = Tensor::create(shape_vec, _meta.dtype, _device_type, _device_id);
        
        // 加载数据到tensor
        tensor->load(data);
        
        // 创建C接口的tensor包装器
        llaisysTensor_t c_tensor = create_llaisys_tensor(tensor);
        
        // 将tensor存储在内部列表中以防止过早释放
        _tensors.push_back(tensor);
        
        // 处理不同类型的张量
        if (tensor_name == "model.embed_tokens.weight") {
            // 输入嵌入层
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.voc || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for embed_tokens.weight: expected [" 
                          << _meta.voc << ", " << _meta.hs << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            _cweights.in_embed = c_tensor;
            std::cout << "  Loaded embed_tokens.weight" << std::endl;
            
        } else if (tensor_name == "lm_head.weight") {
            // 输出层权重
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.voc || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for lm_head.weight: expected [" 
                          << _meta.voc << ", " << _meta.hs << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            _cweights.out_embed = c_tensor;
            std::cout << "  Loaded lm_head.weight" << std::endl;
            
        } else if (tensor_name == "model.norm.weight") {
            // 输出层归一化
            if (ndim != 1 || static_cast<size_t>(shape[0]) != _meta.hs) {
                std::cerr << "Invalid shape for model.norm.weight: expected [" 
                          << _meta.hs << "], got [" << shape[0] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            _cweights.out_norm_w = c_tensor;
            std::cout << "  Loaded model.norm.weight" << std::endl;
            
        } else if (tensor_name.find("input_layernorm.weight") != std::string::npos && layer_idx >= 0) {
            // 注意力层输入归一化
            if (ndim != 1 || static_cast<size_t>(shape[0]) != _meta.hs) {
                std::cerr << "Invalid shape for input_layernorm.weight: expected [" 
                          << _meta.hs << "], got [" << shape[0] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_norm_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " input_layernorm.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("post_attention_layernorm.weight") != std::string::npos && layer_idx >= 0) {
            // MLP层输入归一化
            if (ndim != 1 || static_cast<size_t>(shape[0]) != _meta.hs) {
                std::cerr << "Invalid shape for post_attention_layernorm.weight: expected [" 
                          << _meta.hs << "], got [" << shape[0] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.mlp_norm_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " post_attention_layernorm.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.q_proj.weight") != std::string::npos && layer_idx >= 0) {
            // Q投影权重
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.hs || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for q_proj.weight: expected [" 
                          << _meta.hs << ", " << _meta.hs << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_q_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " q_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.q_proj.bias") != std::string::npos && layer_idx >= 0) {
            // Q投影偏置
            if (ndim != 1 || static_cast<size_t>(shape[0]) != _meta.hs) {
                std::cerr << "Invalid shape for q_proj.bias: expected [" 
                          << _meta.hs << "], got [" << shape[0] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_q_b[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " q_proj.bias" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.k_proj.weight") != std::string::npos && layer_idx >= 0) {
            // K投影权重 (GQA: 实际维度是 nkvh * dh)
            size_t expected_dim = static_cast<size_t>(_meta.nkvh * _meta.dh);
            if (ndim != 2 || static_cast<size_t>(shape[0]) != expected_dim || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for k_proj.weight: expected [" << expected_dim 
                          << ", " << _meta.hs << "], got [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_k_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " k_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.k_proj.bias") != std::string::npos && layer_idx >= 0) {
            // K投影偏置 (GQA)
            size_t expected_dim = static_cast<size_t>(_meta.nkvh * _meta.dh);
            if (ndim != 1 || static_cast<size_t>(shape[0]) != expected_dim) {
                std::cerr << "Invalid shape for k_proj.bias: expected [" << expected_dim 
                          << "], got [" << shape[0] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_k_b[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " k_proj.bias" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.v_proj.weight") != std::string::npos && layer_idx >= 0) {
            // V投影权重 (GQA)
            size_t expected_dim = static_cast<size_t>(_meta.nkvh * _meta.dh);
            if (ndim != 2 || static_cast<size_t>(shape[0]) != expected_dim || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for v_proj.weight: expected [" << expected_dim 
                          << ", " << _meta.hs << "], got [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_v_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " v_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.v_proj.bias") != std::string::npos && layer_idx >= 0) {
            // V投影偏置 (GQA)
            size_t expected_dim = static_cast<size_t>(_meta.nkvh * _meta.dh);
            if (ndim != 1 || static_cast<size_t>(shape[0]) != expected_dim) {
                std::cerr << "Invalid shape for v_proj.bias: expected [" << expected_dim 
                          << "], got [" << shape[0] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_v_b[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " v_proj.bias" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("self_attn.o_proj.weight") != std::string::npos && layer_idx >= 0) {
            // 输出投影权重
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.hs || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for o_proj.weight: expected [" 
                          << _meta.hs << ", " << _meta.hs << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.attn_o_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " o_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("mlp.gate_proj.weight") != std::string::npos && layer_idx >= 0) {
            // MLP门投影权重
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.di || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for gate_proj.weight: expected [" 
                          << _meta.di << ", " << _meta.hs << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.mlp_gate_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " gate_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("mlp.up_proj.weight") != std::string::npos && layer_idx >= 0) {
            // MLP上投影权重
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.di || 
                static_cast<size_t>(shape[1]) != _meta.hs) {
                std::cerr << "Invalid shape for up_proj.weight: expected [" 
                          << _meta.di << ", " << _meta.hs << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.mlp_up_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " up_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else if (tensor_name.find("mlp.down_proj.weight") != std::string::npos && layer_idx >= 0) {
            // MLP下投影权重
            if (ndim != 2 || static_cast<size_t>(shape[0]) != _meta.hs || 
                static_cast<size_t>(shape[1]) != _meta.di) {
                std::cerr << "Invalid shape for down_proj.weight: expected [" 
                          << _meta.hs << ", " << _meta.di << "], got [" 
                          << shape[0] << ", " << shape[1] << "]" << std::endl;
                delete c_tensor;
                return -1;
            }
            if (layer_idx >= 0 && layer_idx < static_cast<int>(_meta.nlayer)) {
                _cweights.mlp_down_w[layer_idx] = c_tensor;
                std::cout << "  Loaded layer " << layer_idx << " down_proj.weight" << std::endl;
            } else {
                delete c_tensor;
                return -1;
            }
            
        } else {
            std::cerr << "Unknown tensor: " << tensor_name << std::endl;
            delete c_tensor;
            return -1;
        }
        
        return 0;
        
    } catch (const std::exception &e) {
        std::cerr << "Error loading tensor " << tensor_name << ": " << e.what() << std::endl;
        return -1;
    }
}

// Helper: compute number of elements from shape
static size_t num_elements_from_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

// Helper: concat two tensors along dim 0 (sequence dimension). Returns a new tensor.
// Uses _meta.dtype as dtype for the new tensor.
static tensor_t concat_along_seq(const tensor_t& a, const tensor_t& b, const LlaisysQwen2Meta &meta) {
    if (!a) return b;
    if (!b) return a;
    auto ash = a->shape();
    auto bsh = b->shape();
    if (ash.size() != bsh.size()) {
        throw std::runtime_error("concat_along_seq: rank mismatch");
    }
    for (size_t i = 1; i < ash.size(); ++i) {
        if (ash[i] != bsh[i]) {
            throw std::runtime_error("concat_along_seq: non-seq dims mismatch");
        }
    }
    std::vector<size_t> out_shape = ash;
    out_shape[0] = ash[0] + bsh[0];
    tensor_t out = Tensor::create(out_shape, meta.dtype, LLAISYS_DEVICE_CPU, 0);
    
    // copy bytes
    size_t a_elems = num_elements_from_shape(ash);
    size_t b_elems = num_elements_from_shape(bsh);
    size_t elem_size = a->elementSize();
    size_t a_bytes = a_elems * elem_size;
    size_t b_bytes = b_elems * elem_size;
    
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out->data());
    uint8_t* a_ptr = reinterpret_cast<uint8_t*>(a->data());
    uint8_t* b_ptr = reinterpret_cast<uint8_t*>(b->data());
    
    if (!out_ptr || !a_ptr || !b_ptr) {
        throw std::runtime_error("concat_along_seq: null data pointer");
    }
    memcpy(out_ptr, a_ptr, a_bytes);
    memcpy(out_ptr + a_bytes, b_ptr, b_bytes);
    return out;
}

// 单步推理：处理一个token序列，生成下一个token
int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken, int64_t *out_ids, size_t out_capacity,
              float temperature, int top_k, float top_p) {
    if (!token_ids || ntoken == 0 || !out_ids || out_capacity == 0) {
        std::cerr << "[ERROR] Invalid parameters for infer" << std::endl;
        std::cerr << "[ERROR] token_ids=" << token_ids << ", ntoken=" << ntoken 
                  << ", out_ids=" << out_ids << ", out_capacity=" << out_capacity << std::endl;
        return -1;
    }
    
    try {
        // If incoming sequence length is <= cached length, we assume generation restarted or user truncated:
        // clear caches to avoid mismatch. This is a conservative but safe behavior.
        if (ntoken <= _cache_seq_len) {
            if (_cache_seq_len > 0) {
                std::cout << "[KV-CACHE] Resetting caches because ntoken (" << ntoken 
                          << ") <= cached_seq_len (" << _cache_seq_len << ")" << std::endl;
            }
            for (size_t li = 0; li < _k_cache.size(); ++li) {
                _k_cache[li] = nullptr;
                _v_cache[li] = nullptr;
            }
            _cache_seq_len = 0;
        }
        
        std::vector<int64_t> input_tokens(token_ids, token_ids + ntoken);
        auto index_tensor = _create_tensor({ntoken}, LLAISYS_DTYPE_I64);
        index_tensor->load(input_tokens.data());
        auto embedding_weight = get_tensor_from_ptr(_cweights.in_embed);

        if (!embedding_weight) {
            std::cerr << "Embedding weight not loaded" << std::endl;
            return -1;
        }

        // hidden_states contains full sequence hidden states; we'll update slices for new tokens
        auto hidden_states = _create_tensor({ntoken, _meta.hs}, _meta.dtype);
        embedding(hidden_states, index_tensor, embedding_weight);
        
        auto pos_ids = _create_tensor({ntoken}, LLAISYS_DTYPE_I64);
        std::vector<int64_t> positions(ntoken);
        for (size_t i = 0; i < ntoken; i++) {
            positions[i] = static_cast<int64_t>(i);
        }
        pos_ids->load(positions.data());
        
        const size_t hidden_size = _meta.hs;  // 1536
        const size_t num_heads = _meta.nh; // 12
        const size_t num_kv_heads = _meta.nkvh; // 2
        const size_t head_dim = hidden_size / num_heads; // 128
        // const size_t num_queries_per_kv = num_heads / num_kv_heads; // 6

        // previous cached length
        size_t prev_cache = _cache_seq_len;
        size_t new_tokens = (ntoken > prev_cache) ? (ntoken - prev_cache) : 0;

        for (size_t layer_idx = 0; layer_idx < _meta.nlayer; layer_idx++) {
            // 保存残差连接（对于增量部分，我们将只对 slice 进行更新）
            auto residual = hidden_states;
            
            // Attention前的RMSNorm (对整个序列做norm，便于保持接口简单)
            auto attn_norm_weight = get_tensor_from_ptr(_cweights.attn_norm_w[layer_idx]);
            if (!attn_norm_weight) {
                std::cerr << "Attention norm weight for layer " << layer_idx << " not loaded" << std::endl;
                return -1;
            }
            
            auto normed_hidden = _create_tensor(hidden_states->shape(), _meta.dtype);
            rms_norm(normed_hidden, hidden_states, attn_norm_weight, _meta.epsilon);
            
            // Q投影: 对新增部分（若 prev_cache == 0 则是全部）
            auto q_proj_weight = get_tensor_from_ptr(_cweights.attn_q_w[layer_idx]);
            auto q_proj_bias = get_tensor_from_ptr(_cweights.attn_q_b[layer_idx]);
            auto k_proj_weight = get_tensor_from_ptr(_cweights.attn_k_w[layer_idx]);
            auto k_proj_bias = get_tensor_from_ptr(_cweights.attn_k_b[layer_idx]);
            auto v_proj_weight = get_tensor_from_ptr(_cweights.attn_v_w[layer_idx]);
            auto v_proj_bias = get_tensor_from_ptr(_cweights.attn_v_b[layer_idx]);
            
            if (!q_proj_weight || !k_proj_weight || !v_proj_weight) {
                std::cerr << "Projection weights for layer " << layer_idx << " not loaded" << std::endl;
                return -1;
            }
            
            // For Q we only need to compute for new tokens (or all if no cache)
            tensor_t normed_new_slice = nullptr;
            if (new_tokens > 0) {
                normed_new_slice = normed_hidden->slice(0, prev_cache, ntoken); // shape [new_tokens, hs]
            } else {
                // No new tokens to process this layer; skip to next
                continue;
            }
            
            // Q_proj for new tokens: shape [new_tokens, hidden_size]
            auto q_proj_new = _create_tensor({new_tokens, hidden_size}, _meta.dtype);
            linear(q_proj_new, normed_new_slice, q_proj_weight, q_proj_bias);
            
            // K_proj for new tokens: shape [new_tokens, num_kv_heads * head_dim]
            auto k_proj_new = _create_tensor({new_tokens, num_kv_heads * head_dim}, _meta.dtype);
            linear(k_proj_new, normed_new_slice, k_proj_weight, k_proj_bias);
            
            // V_proj for new tokens: shape [new_tokens, num_kv_heads * head_dim]
            auto v_proj_new = _create_tensor({new_tokens, num_kv_heads * head_dim}, _meta.dtype);
            linear(v_proj_new, normed_new_slice, v_proj_weight, v_proj_bias);
            
            // ================== 重塑/分割并缓存K/V ==================
            auto q_reshaped_new = q_proj_new->view({new_tokens, num_heads, head_dim});
            auto k_reshaped_new = k_proj_new->view({new_tokens, num_kv_heads, head_dim});
            auto v_reshaped_new = v_proj_new->view({new_tokens, num_kv_heads, head_dim});
            
            // 对新增 K 应用 RoPE（注意我们使用 meta.theta）
            auto k_rope_new = _create_tensor({new_tokens, num_kv_heads, head_dim}, _meta.dtype);
            auto pos_slice = pos_ids->slice(0, prev_cache, ntoken); // positions for new tokens
            rope(k_rope_new, k_reshaped_new, pos_slice, _meta.theta);
            
            // Q 的 RoPE 也只在新 Q 上应用
            auto q_rope_new = _create_tensor({new_tokens, num_heads, head_dim}, _meta.dtype);
            rope(q_rope_new, q_reshaped_new, pos_slice, _meta.theta);
            
            // Append new k/v to layer cache
            try {
                _k_cache[layer_idx] = concat_along_seq(_k_cache[layer_idx], k_rope_new, _meta);
                _v_cache[layer_idx] = concat_along_seq(_v_cache[layer_idx], v_reshaped_new, _meta);
            } catch (const std::exception &e) {
                std::cerr << "[KV-CACHE] Error concatenating caches at layer " << layer_idx << ": " << e.what() << std::endl;
                return -1;
            }
            
            // 访问缓存后的完整 K/V
            tensor_t k_full = _k_cache[layer_idx]; // shape [cached_seq_len + new_tokens, num_kv_heads, head_dim]
            tensor_t v_full = _v_cache[layer_idx];

            // ================== 自注意力（仅对新增 tokens 的 Q 进行计算，K/V 使用完整缓存） ==================
            float scale = 1.0f / sqrtf(static_cast<float>(_meta.dh));
            auto attn_output_new = _create_tensor({new_tokens, _meta.nh, _meta.dh}, _meta.dtype);
            self_attention(attn_output_new, q_rope_new, k_full, v_full, scale);
            
            // 投影输出并写回 hidden_states 的对应 slice
            auto attn_reshaped_new = attn_output_new->view({new_tokens, _meta.hs});
            auto proj_output_new = _create_tensor({new_tokens, _meta.hs}, _meta.dtype);
            auto o_proj_weight = get_tensor_from_ptr(_cweights.attn_o_w[layer_idx]);
            if (!o_proj_weight) {
                std::cerr << "Output projection weight for layer " << layer_idx << " not loaded" << std::endl;
                return -1;
            }
            linear(proj_output_new, attn_reshaped_new, o_proj_weight, nullptr); // 通常没有bias
            
            // 进行残差连接： hidden_states[new_slice] = hidden_states[new_slice] + proj_output_new
            auto hidden_new_slice = hidden_states->slice(0, prev_cache, ntoken);
            auto residual_new_slice = hidden_new_slice; // copy-reference to previous values
            add(hidden_new_slice, residual_new_slice, proj_output_new);

            // =============== MLP 块（仅在新增 tokens 上执行） ===============
            auto mlp_norm_weight = get_tensor_from_ptr(_cweights.mlp_norm_w[layer_idx]);
            if (!mlp_norm_weight) {
                std::cerr << "MLP norm weight for layer " << layer_idx << " not loaded" << std::endl;
                return -1;
            }
            
            auto mlp_normed_new = _create_tensor(hidden_new_slice->shape(), _meta.dtype);
            rms_norm(mlp_normed_new, hidden_new_slice, mlp_norm_weight, _meta.epsilon);
            
            auto gate_weight = get_tensor_from_ptr(_cweights.mlp_gate_w[layer_idx]);
            auto gate = _create_tensor({new_tokens, _meta.di}, _meta.dtype);
            linear(gate, mlp_normed_new, gate_weight, nullptr);
            
            auto up_weight = get_tensor_from_ptr(_cweights.mlp_up_w[layer_idx]);
            auto up = _create_tensor({new_tokens, _meta.di}, _meta.dtype);
            linear(up, mlp_normed_new, up_weight, nullptr);
            
            auto swiglu_out = _create_tensor({new_tokens, _meta.di}, _meta.dtype);
            swiglu(swiglu_out, gate, up);
            
            auto down_weight = get_tensor_from_ptr(_cweights.mlp_down_w[layer_idx]);
            auto mlp_output_new = _create_tensor({new_tokens, _meta.hs}, _meta.dtype);
            linear(mlp_output_new, swiglu_out, down_weight, nullptr);
            
            // 残差连接写回 hidden_states[new_slice] = hidden_states[new_slice] + mlp_output_new
            add(hidden_new_slice, hidden_new_slice, mlp_output_new);
            
            // 到下一层：此处 hidden_states 的对应 slice已经被更新
        }
        
        // 更新缓存长度
        _cache_seq_len = ntoken;
        
        // 最终RMSNorm（对整个 hidden_states 做norm）
        auto final_norm_weight = get_tensor_from_ptr(_cweights.out_norm_w);
        if (!final_norm_weight) {
            std::cerr << "Final norm weight not loaded" << std::endl;
            return -1;
        }

        auto final_normed = _create_tensor(hidden_states->shape(), _meta.dtype);
        rms_norm(final_normed, hidden_states, final_norm_weight, _meta.epsilon);
        
        // 输出投影（LM Head）
        auto lm_head_weight = get_tensor_from_ptr(_cweights.out_embed);
        if (!lm_head_weight) {
            std::cerr << "LM head weight not loaded" << std::endl;
            return -1;
        }
        
        // 只使用最后一个token的隐藏状态
        auto last_hidden = final_normed->slice(0, ntoken-1, ntoken);
        auto logits = _create_tensor({1, _meta.voc}, _meta.dtype);
        linear(logits, last_hidden, lm_head_weight, nullptr);
        
        // 直接使用logits（已经是最后一个token的logits）
        auto last_logits_flat = logits->view({_meta.voc});    // [vocab_size]
        
        // 使用sample获取下一个token
        auto token_tensor = _create_tensor({1}, LLAISYS_DTYPE_I64);
   
        // 调用采样算子
        ops::sample(token_tensor, last_logits_flat, temperature, top_k, top_p);

        // 读取结果
        int64_t next_token;

        // 拷贝数据
        memcpy(&next_token, token_tensor->data(), sizeof(int64_t));

        // 返回结果
        out_ids[0] = next_token;
        
        return 1;
        
    } catch (const std::exception &e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }
}