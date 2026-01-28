#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// out: output buffer (N x D)
// gate: gate buffer (N x D)
// up:   up buffer (N x D)
// dtype: datatype of all tensors (F32/F16/BF16 supported)
// N: rows
// D: cols
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t N, size_t D);
}