#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// out: output buffer (N x d)
// in: input buffer (N x d)
// weight: scale buffer (d)
// dtype: datatype of out/in/weight
// N: number of rows
// d: length of each row (last dim)
// eps: small epsilon to avoid divide by zero
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype, size_t N, size_t d, float eps);
}