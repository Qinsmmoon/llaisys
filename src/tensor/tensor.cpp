#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);

    // 计算步长strides 从最后一个维度向前计算
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        // 特殊情况：请求CPU内存但当前在GPU环境中
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        // 正常情况：GPU请求或纯CPU环境
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const auto &sh = this->shape();
    const auto &st = this->strides();
    size_t n = sh.size();

    if (n == 0) {
        return true;
    }

    size_t expected = 1;
    for (ptrdiff_t i = static_cast<ptrdiff_t>(n) - 1; i >= 0; --i) {
        if (static_cast<size_t>(st[i]) != expected) {
            return false;
        }
        expected *= sh[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != this->ndim()) {
        throw std::invalid_argument("permute: order size must equal number of dimensions");
    }
    
    size_t n = this->ndim();
    std::vector<char> seen(n, 0);
    for (size_t v : order) {
        if (v >= n) {
            throw std::out_of_range("permute: index out of range");
        }
        if (seen[v]) {
            throw std::invalid_argument("permute: duplicate dimension in order");
        }
        seen[v] = 1;
    }

    TensorMeta meta;
    meta.dtype = this->dtype();
    meta.shape.resize(n);
    meta.strides.resize(n);
    for (size_t i = 0; i < n; ++i) {
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 确保与原张量的元素数量相同
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::invalid_argument("view: total elements mismatch");
    }

    // view操作要求张量在内存中是连续存储的
    if (!this->isContiguous()) {
        throw std::runtime_error("view: tensor must be contiguous to view");
    }

    // 计算新步幅
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; ++i) {
        new_strides[ndim_ - i] = static_cast<ptrdiff_t>(stride);
        stride *= shape[ndim_ - i];
    }

    // 零拷贝操作：没有数据复制，只是改变了对数据的"视图"
    TensorMeta meta{this->dtype(), shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= this->ndim()) {
        throw std::out_of_range("slice: dim out of range");
    }
    if (start > end || end > _meta.shape[dim]) {
        throw std::out_of_range("slice: invalid start/end");
    }

    TensorMeta meta = _meta;  // 拷贝元数据
    meta.shape[dim] = end - start;  // 更新切片维度的形状

    // 计算字节偏移量
    size_t element_offset = start * _meta.strides[dim];
    size_t byte_offset = element_offset * this->elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset + byte_offset));
}

void Tensor::load(const void *src_) {
    if (src_ == nullptr) {
        return;
    }

    if (this->numel() == 0) {
        return;  
    }

    // 获取张量自己的运行时
    auto& runtime = _storage->runtime(); 
    
    size_t bytes = this->numel() * this->elementSize();
    std::byte* dst = this->data();

    if (_storage->isHost()) {
        // 主机内存：直接memcpy
        std::memcpy(dst, src_, bytes);
    } else {
        // 设备内存：通过runtime的API复制
        runtime.api()->memcpy_sync(
            dst,                    // 目标：设备内存
            src_,                   // 源：主机内存
            bytes,                  // 字节数
            LLAISYS_MEMCPY_H2D      // 方向：主机→设备
        );
    }
}

tensor_t Tensor::contiguous() const {
    // todo
    return nullptr;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // todo
    return nullptr;
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
   // todo
    return nullptr;
}   

} // namespace llaisys
