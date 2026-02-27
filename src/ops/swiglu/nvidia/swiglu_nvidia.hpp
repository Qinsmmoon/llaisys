#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// out: device output buffer (N x D)
// gate: device gate buffer (N x D)
// up:   device up buffer (N x D)
// dtype: datatype of all tensors (F32/F16/BF16 supported)
// N: rows
// D: cols
// Note: pointers are expected to be device pointers.
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t N, size_t D);

} // namespace llaisys::ops::nvidia