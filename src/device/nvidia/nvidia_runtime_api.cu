#include "../runtime_api.hpp"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

namespace llaisys::device::nvidia {

namespace runtime_api {

// 检查 CUDA 错误
inline void checkCudaError(cudaError_t err, const char* func) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in " << func << ": " 
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA operation failed");
    }
}

// 获取设备数量
int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " 
                  << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return count;
}

// 设置当前设备
void setDevice(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    checkCudaError(err, "setDevice");
}

// 设备同步
void deviceSynchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    checkCudaError(err, "deviceSynchronize");
}

// 创建流
llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    checkCudaError(err, "createStream");
    return reinterpret_cast<llaisysStream_t>(stream);
}

// 销毁流
void destroyStream(llaisysStream_t stream) {
    if (stream) {
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        cudaError_t err = cudaStreamDestroy(cuda_stream);
        checkCudaError(err, "destroyStream");
    }
}

// 流同步
void streamSynchronize(llaisysStream_t stream) {
    if (stream) {
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        cudaError_t err = cudaStreamSynchronize(cuda_stream);
        checkCudaError(err, "streamSynchronize");
    }
}

// 设备内存分配
void *mallocDevice(size_t size) {
    if (size == 0) return nullptr;
    
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    checkCudaError(err, "mallocDevice");
    return ptr;
}

// 释放设备内存
void freeDevice(void *ptr) {
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        checkCudaError(err, "freeDevice");
    }
}

// 主机内存分配（页锁定内存，提高传输速度）
void *mallocHost(size_t size) {
    if (size == 0) return nullptr;
    
    void *ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    checkCudaError(err, "mallocHost");
    return ptr;
}

// 释放主机内存
void freeHost(void *ptr) {
    if (ptr) {
        cudaError_t err = cudaFreeHost(ptr);
        checkCudaError(err, "freeHost");
    }
}

// 同步内存拷贝 - 使用 llaisysMemcpyKind_t 枚举
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) return;
    
    // 转换内存拷贝类型 - 匹配 llaisys.h 中的定义
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:  // 主机到主机
            cuda_kind = cudaMemcpyHostToHost;
            break;
        case LLAISYS_MEMCPY_H2D:  // 主机到设备
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case LLAISYS_MEMCPY_D2H:  // 设备到主机
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case LLAISYS_MEMCPY_D2D:  // 设备到设备
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            std::cerr << "Invalid memcpy kind: " << kind << std::endl;
            return;
    }
    
    cudaError_t err = cudaMemcpy(dst, src, size, cuda_kind);
    checkCudaError(err, "memcpySync");
}

// 异步内存拷贝（无流版本 - 使用默认流）- 注意：这是一个独立实现，不调用5参数版本
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) return;
    
    // 转换内存拷贝类型
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        case LLAISYS_MEMCPY_H2D:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case LLAISYS_MEMCPY_D2H:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case LLAISYS_MEMCPY_D2D:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            std::cerr << "Invalid memcpy kind: " << kind << std::endl;
            return;
    }
    
    // 使用默认流 (nullptr)
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cuda_kind, nullptr);
    checkCudaError(err, "memcpyAsync");
}

// 异步内存拷贝（带流版本）
void memcpyAsync(void *dst, const void *src, size_t size, 
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {  // 注意参数类型改为 llaisysStream_t
    if (size == 0) return;
    
    // 转换内存拷贝类型
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        case LLAISYS_MEMCPY_H2D:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case LLAISYS_MEMCPY_D2H:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case LLAISYS_MEMCPY_D2D:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            std::cerr << "Invalid memcpy kind: " << kind << std::endl;
            return;
    }
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cuda_kind, cuda_stream);
    checkCudaError(err, "memcpyAsync");
}

// 静态运行时 API 结构体实例
static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync
};

} // namespace runtime_api

// 对外接口：获取运行时 API 结构体指针
const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia