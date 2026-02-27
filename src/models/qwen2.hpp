#pragma once
#ifndef LLAISYS_MODELS_QWEN2_HPP
#define LLAISYS_MODELS_QWEN2_HPP

#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace llaisys {
namespace models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    ~Qwen2Model();

    int64_t infer(const int64_t *token_ids, size_t ntoken, int64_t *out_ids, size_t out_capacity);

    int load_tensor(const char *name, const void *data, size_t ndim, const int64_t *shape);

private:
    // Stored copy of meta
    LlaisysQwen2Meta _meta;

    // Device info for where model should run
    llaisysDeviceType_t _device_type;

    int _device_id;

    // C-view of weights to return through the C API.
    // Fields are initialized to nullptr; future loading code may populate them.
    LlaisysQwen2Weights _cweights;

    // per-layer kv cache, layout: (seq_len, nkv_heads, head_dim)
    std::vector<tensor_t> _k_cache, _v_cache;

    size_t _cache_seq_len;

    int64_t _last_token;
    
    // 存储所有的tensor智能指针，防止它们被释放
    std::vector<tensor_t> _tensors;

};

} // namespace models
} // namespace llaisys

// IMPORTANT:
// The C struct `LlaisysQwen2Model` is defined in the C header "llaisys/models/qwen2.h".
// Do NOT redefine it here. If you need a forward-declaration for C++ code, use the
// following simple forward declaration (it does not repeat/define the struct).
__C {
    struct LlaisysQwen2Model;
}

#endif // LLAISYS_MODELS_QWEN2_HPP