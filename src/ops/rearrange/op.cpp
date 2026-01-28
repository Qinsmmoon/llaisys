#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <vector>
#include <cstring>

namespace llaisys::ops {

static size_t dtype_size(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sizeof(float);
    case LLAISYS_DTYPE_F16:
        return sizeof(llaisys::fp16_t);
    case LLAISYS_DTYPE_BF16:
        return sizeof(llaisys::bf16_t);
    case LLAISYS_DTYPE_I64:
        return sizeof(int64_t);
    case LLAISYS_DTYPE_I32:
        return sizeof(int32_t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    return 0;
}

void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);

    // Shapes must be identical
    const auto in_shape = in->shape();
    const auto out_shape = out->shape();
    ASSERT(in_shape == out_shape, "rearrange: input and output must have the same shape.");

    const size_t ndim = in_shape.size();
    ASSERT(ndim >= 1, "rearrange: tensors must have at least 1 dimension.");

    // dtype must match
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    const size_t elem_size = dtype_size(in->dtype());
    const size_t total_elems = static_cast<size_t>(in->numel());

    // Fast path: if both contiguous, single memcpy
    if (in->isContiguous() && out->isContiguous()) {
        std::memcpy(out->data(), in->data(), total_elems * elem_size);
        return;
    }

    // Need strides (in elements) to compute offsets. Expect tensor API to provide strides().
    const auto in_strides = in->strides();
    const auto out_strides = out->strides();
    ASSERT(in_strides.size() == ndim && out_strides.size() == ndim,
           "rearrange: tensor strides size mismatch.");

    // We'll iterate linear index -> multi-index, compute source/dst offsets (in elements), then copy element bytes.
    std::vector<size_t> idx_coords(ndim);
    std::byte *out_ptr = reinterpret_cast<std::byte *>(out->data());
    const std::byte *in_ptr = reinterpret_cast<const std::byte *>(in->data());

    for (size_t linear = 0; linear < total_elems; ++linear) {
        // compute multi-index (mixed-radix) from linear
        size_t rem = linear;
        for (size_t d = ndim; d-- > 0;) {
            const size_t dim = in_shape[d];
            idx_coords[d] = rem % dim;
            rem /= dim;
        }

        // compute element offsets (in elements)
        size_t in_off = 0;
        size_t out_off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            in_off += idx_coords[d] * static_cast<size_t>(in_strides[d]);
            out_off += idx_coords[d] * static_cast<size_t>(out_strides[d]);
        }

        const std::byte *src = in_ptr + in_off * elem_size;
        std::byte *dst = out_ptr + out_off * elem_size;
        std::memcpy(dst, src, elem_size);
    }
}

} // namespace llaisys::ops