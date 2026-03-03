#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// attn_val: [seqlen, nhead, dv] (序列长度 × 注意力头数 × 值维度)
// q: [seqlen, nhead, d] (序列长度 × 注意力头数 × 头维度)
// k: [total_len, nkvhead, d] (总长度 × Key-Value头数 × 头维度)
// v: [total_len, nkvhead, dv] (总长度 × Key-Value头数 × 值维度)
// dtype: datatype (F32/F16/BF16 supported)
// scale: scaling factor applied to QK^T
// seqlen, nhead, d: query shape
// total_len, nkvhead: key/value leading length and head count
// dv: value dimension
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, float scale,
                    size_t seqlen, size_t nhead, size_t d,
                    size_t total_len, size_t nkvhead, size_t dv);
}