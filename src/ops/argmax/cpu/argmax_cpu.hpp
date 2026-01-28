#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// max_idx: pointer to index output buffer
// max_val: pointer to value output buffer
// vals: input values buffer
// val_type: datatype of vals and max_val
// idx_type: datatype of max_idx (e.g., I64 or I32)
// numel: number of elements in vals
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t val_type,
            llaisysDataType_t idx_type, size_t numel);
}