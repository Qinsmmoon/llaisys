#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// out: output buffer (seqlen * nhead * d)
// in: input buffer (seqlen * nhead * d)
// pos_ids: int64 array of length seqlen
// dtype: datatype of in/out (F32/F16/BF16 supported)
// seqlen: sequence length
// nhead: number of heads (or nkvhead)
// d: hidden dimension (must be even)
// theta: base frequency parameter (e.g., 10000)
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t dtype,
          size_t seqlen, size_t nhead, size_t d, float theta);
}