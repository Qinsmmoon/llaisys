#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
// out: output buffer (N x D_out)
// in: input buffer (N x D_in)
// weight: weight buffer (D_out x D_in)  -- note: weight is stored as rows = output features
// bias: optional bias buffer (D_out) or nullptr if absent
// dtype: datatype of out/in/weight/bias
// N: batch size (rows of in)
// D_in: input feature dim (cols of in, cols of weight)
// D_out: output feature dim (rows of weight)
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype,
            size_t N, size_t D_in, size_t D_out);
}