#include "sample_nvidia.hpp"
#include "../../../utils.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <limits>
#include <random>

namespace llaisys::ops::nvidia {

// 辅助：获取数据类型元素大小（字节）
static inline size_t elementSize(llaisysDataType_t dtype) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32: return 4;
        case LLAISYS_DTYPE_F16: return 2;
        case LLAISYS_DTYPE_BF16: return 2;
        case LLAISYS_DTYPE_I64: return 8;
        case LLAISYS_DTYPE_I32: return 4;
        default: return 0;
    }
}

// 内核：将不同数据类型的logits转换为float
__global__ void convert_to_float_kernel(
    const std::byte* logits_ptr,
    llaisysDataType_t dtype,
    float* logits_float,
    size_t vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    
    const std::byte* src = logits_ptr + idx * elementSize(dtype);
    float val = 0.0f;
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            val = *reinterpret_cast<const float*>(src);
            break;
        case LLAISYS_DTYPE_BF16: {
            bf16_t b = *reinterpret_cast<const bf16_t*>(src);
            val = llaisys::utils::cast<float>(b);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            fp16_t f = *reinterpret_cast<const fp16_t*>(src);
            val = llaisys::utils::cast<float>(f);
            break;
        }
        default:
            val = -std::numeric_limits<float>::infinity();
    }
    logits_float[idx] = val;
}

void sample_nvidia(
    std::byte* token_ptr,
    const std::byte* logits_ptr,
    llaisysDataType_t logits_dtype,
    size_t vocab_size,
    float temperature,
    int top_k,
    float top_p,
    cudaStream_t stream
) {
    // 分配设备内存存放float logits
    float* d_logits = nullptr;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    
    // 转换数据类型到float
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    convert_to_float_kernel<<<blocks, threads, 0, stream>>>(
        logits_ptr, logits_dtype, d_logits, vocab_size
    );
    cudaStreamSynchronize(stream);
    
    thrust::device_ptr<float> dev_ptr(d_logits);
    
    // 处理温度退化情况（argmax）
    if (temperature <= 0.0f) {
        thrust::device_vector<float>::iterator iter = thrust::max_element(dev_ptr, dev_ptr + vocab_size);
        size_t max_idx = iter - dev_ptr;
        int64_t token = static_cast<int64_t>(max_idx);
        cudaMemcpy(token_ptr, &token, sizeof(int64_t), cudaMemcpyHostToHost);
        cudaFree(d_logits);
        return;
    }
    
    // 温度缩放
    if (temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        thrust::transform(dev_ptr, dev_ptr + vocab_size, dev_ptr,
                          [inv_temp] __device__ (float x) { return x * inv_temp; });
    }
    
    // top-k 过滤
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
        // 创建索引数组并排序（按logits降序）
        thrust::device_vector<int> indices(vocab_size);
        thrust::sequence(indices.begin(), indices.end());
        
        // 复制logits用于排序（避免破坏原始顺序）
        thrust::device_vector<float> sorted_logits(d_logits, d_logits + vocab_size);
        thrust::sort_by_key(sorted_logits.begin(), sorted_logits.end(), indices.begin(), thrust::greater<float>());
        
        // 获取第k大的值
        float kth_val = sorted_logits[top_k - 1];
        
        // 将小于阈值的logits置为 -inf
        thrust::transform(dev_ptr, dev_ptr + vocab_size, dev_ptr,
                          [kth_val] __device__ (float x) { return x < kth_val ? -std::numeric_limits<float>::infinity() : x; });
    }
    
    // softmax: 减去最大值，exp，归一化
    float max_val = *thrust::max_element(dev_ptr, dev_ptr + vocab_size);
    thrust::transform(dev_ptr, dev_ptr + vocab_size, dev_ptr,
                      [max_val] __device__ (float x) { return expf(x - max_val); });
    float sum = thrust::reduce(dev_ptr, dev_ptr + vocab_size, 0.0f, thrust::plus<float>());
    thrust::transform(dev_ptr, dev_ptr + vocab_size, dev_ptr,
                      [sum] __device__ (float x) { return x / sum; });
    
    // top-p 过滤 (nucleus)
    if (top_p > 0.0f && top_p < 1.0f - 1e-6f) {
        // 创建索引数组
        thrust::device_vector<int> idx(vocab_size);
        thrust::sequence(idx.begin(), idx.end());
        
        // 复制概率用于排序
        thrust::device_vector<float> probs_sorted(d_logits, d_logits + vocab_size);
        thrust::sort_by_key(probs_sorted.begin(), probs_sorted.end(), idx.begin(), thrust::greater<float>());
        
        // 计算前缀和
        thrust::device_vector<float> cumsum(vocab_size);
        thrust::inclusive_scan(probs_sorted.begin(), probs_sorted.end(), cumsum.begin());
        
        // 找到第一个累积概率 > top_p 的位置
        thrust::device_vector<float>::iterator cutoff_iter = thrust::find_if(cumsum.begin(), cumsum.end(),
            [top_p] __device__ (float c) { return c > top_p; });
        size_t cutoff = cutoff_iter - cumsum.begin();
        if (cutoff == 0) cutoff = vocab_size;
        
        // 标记保留的token
        thrust::device_vector<bool> keep(vocab_size, false);
        for (size_t i = 0; i < cutoff; ++i) {
            int orig_idx = idx[i];
            keep[orig_idx] = true;
        }
        
        // 将未保留的概率置零
        thrust::transform(dev_ptr, dev_ptr + vocab_size, keep.begin(), dev_ptr,
                          [] __device__ (float p, bool k) { return k ? p : 0.0f; });
        
        // 重新归一化
        float new_sum = thrust::reduce(dev_ptr, dev_ptr + vocab_size, 0.0f, thrust::plus<float>());
        if (new_sum > 0.0f) {
            thrust::transform(dev_ptr, dev_ptr + vocab_size, dev_ptr,
                              [new_sum] __device__ (float x) { return x / new_sum; });
        } else {
            thrust::fill(dev_ptr, dev_ptr + vocab_size, 1.0f / vocab_size);
        }
    }
    
    // 计算累积概率（用于采样）
    thrust::device_vector<float> cumsum(vocab_size);
    thrust::inclusive_scan(dev_ptr, dev_ptr + vocab_size, cumsum.begin());
    
    // 生成随机数（host端，简单处理）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen);
    
    // 二分查找累积概率 >= r 的位置
    thrust::device_vector<float>::iterator idx_iter = thrust::lower_bound(cumsum.begin(), cumsum.end(), r);
    size_t sampled_idx = idx_iter - cumsum.begin();
    if (sampled_idx == vocab_size) sampled_idx = vocab_size - 1;
    
    int64_t token = static_cast<int64_t>(sampled_idx);
    cudaMemcpy(token_ptr, &token, sizeof(int64_t), cudaMemcpyHostToHost);
    
    cudaFree(d_logits);
}

} // namespace llaisys::ops::nvidia