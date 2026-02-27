#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// attn_val: output buffer (seqlen x nhead x dv)
// q: (seqlen x nhead x d)
// k: (total_len x nkvhead x d)
// v: (total_len x nkvhead x dv)
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