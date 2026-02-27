#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype,
               size_t N, size_t weight_rows, size_t emb_dim);
}