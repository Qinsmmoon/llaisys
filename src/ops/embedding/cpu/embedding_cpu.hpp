#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// out: output buffer (N x emb_dim)
// index: input indices (N) of type I64
// weight: weight matrix (weight_rows x emb_dim)
// dtype: datatype of out and weight (F32/F16/BF16 supported)
// N: number of indices / output rows
// weight_rows: number of rows in weight (num_embeddings)
// emb_dim: embedding dimension (columns)
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype,
               size_t N, size_t weight_rows, size_t emb_dim);
}