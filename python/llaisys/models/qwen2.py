from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from ..libllaisys.llaisys_types import llaisysDataType_t
from pathlib import Path
import safetensors
from safetensors import safe_open
import numpy as np
import json
import gc
import time
import ctypes
from ctypes import (
    c_int64,
    c_int,
    c_size_t,
    POINTER,
    byref,
    c_void_p,
    c_char_p,
)

import tempfile
import os

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, max_seq_len: int = 4096):
        """
        初始化 Qwen2 模型
        
        Args:
            model_path: 模型目录路径，包含 safetensors 文件和 config.json
            device: 运行设备类型 (CPU/GPU)
            max_seq_len: 最大序列长度
        """
        model_path = Path(model_path)
        
        # 1. 读取配置文件
        config = self._load_config(model_path / "config.json")
        
        # 2. 创建模型元数据
        meta = self._create_meta_from_config(config, max_seq_len)
        
        # 3. 创建 C 模型对象
        self._model_ptr = self._create_model(meta, device)
        if not self._model_ptr:
            raise RuntimeError("Failed to create Qwen2 model")
            
        # 4. 加载权重
        self._load_weights(model_path)
        
        # 5. 保存重要参数
        self.config = config
        self.max_seq_len = max_seq_len
        self._device = device
        
        print(f"Qwen2 model initialized successfully with {config.get('num_hidden_layers', 28)} layers")
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _create_meta_from_config(self, config: Dict[str, Any], max_seq_len: int) -> ctypes.Structure:
        """从配置文件创建元数据结构"""
        
        meta = LlaisysQwen2Meta()
        
        meta.dtype = 19  # LLAISYS_DTYPE_BF16
        
        # 从配置获取模型参数
        meta.nlayer = config.get("num_hidden_layers", 28)
        meta.hs = config.get("hidden_size", 1536)  # 隐藏层大小
        meta.nh = config.get("num_attention_heads", 12)  # 注意力头数
        meta.nkvh = config.get("num_key_value_heads", 2)  # KV头数（GQA）
        meta.dh = meta.hs // meta.nh  # 每个头的维度
        meta.di = config.get("intermediate_size", 8960)  # MLP中间层大小
        meta.maxseq = min(max_seq_len, config.get("max_position_embeddings", 131072))
        meta.voc = config.get("vocab_size", 151936)
        
        # RMSNorm 参数
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        
        # RoPE 参数
        meta.theta = config.get("rope_theta", 10000.0)
        
        # 结束标记（注意：Qwen2 的 bos_token_id 和 eos_token_id 相同）
        meta.end_token = config.get("eos_token_id", 151643)
        
        print(f"Model config: layers={meta.nlayer}, hidden_size={meta.hs}, "
              f"heads={meta.nh}, kv_heads={meta.nkvh}, vocab={meta.voc}, "
              f"max_seq={meta.maxseq}, dtype=bfloat16")
        return meta
    
    def _create_model(self, meta, device: DeviceType) -> ctypes.c_void_p:
        """创建 C 模型对象"""
        # 创建元数据指针
        meta_ptr = ctypes.pointer(meta)
        device_ids = (c_int * 1)(0)
        # 调用 C 函数创建模型
        model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta_ptr,
            device,
            device_ids,  # device_ids
            1            # ndevice
        )
        
        if not model_ptr:
            raise RuntimeError("Failed to create model in C++ backend")
            
        return model_ptr
    
    def _float32_to_bfloat16_vectorized(self, f32_array):
        """使用向量化操作转换float32到BF16（更高效）"""
        # 将float32数组重新解释为uint32
        f32_as_uint32 = f32_array.view(np.uint32)
        
        # 应用舍入：添加0x7FFF，然后加上最低位的进位
        rounding_bias = 0x7FFF + ((f32_as_uint32 >> 16) & 1)
        bf16_as_uint32 = (f32_as_uint32 + rounding_bias) >> 16
        
        # 取低16位作为BF16
        bf16 = bf16_as_uint32.astype(np.uint16)
        
        return bf16
    
    def _float32_to_bfloat16_chunked(self, f32_array, chunk_size_mb=100):
        """
        分块转换float32到BF16，避免内存不足
        
        Args:
            f32_array: numpy float32数组
            chunk_size_mb: 每块大小（MB）
        """
        # 计算每个元素的大小（字节）
        element_size_bytes = f32_array.itemsize
        
        # 计算每块应该有多少个元素
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        elements_per_chunk = chunk_size_bytes // element_size_bytes
        
        # 如果是1D或2D数组，按行分块
        if f32_array.ndim <= 2:
            total_elements = f32_array.size
            if total_elements <= elements_per_chunk:
                # 如果整个数组可以一次性处理
                return self._float32_to_bfloat16_vectorized(f32_array)
            else:
                # 分块处理
                bf16_parts = []
                rows = f32_array.shape[0]
                elements_per_row = f32_array.size // rows
                rows_per_chunk = max(1, elements_per_chunk // elements_per_row)
                
                for i in range(0, rows, rows_per_chunk):
                    end = min(i + rows_per_chunk, rows)
                    chunk = f32_array[i:end]
                    bf16_chunk = self._float32_to_bfloat16_vectorized(chunk)
                    bf16_parts.append(bf16_chunk)
                    # 释放chunk内存
                    del chunk
                    gc.collect()
                
                # 合并结果
                return np.concatenate(bf16_parts, axis=0)
        else:
            # 对于更高维度的数组，使用原始方法
            return self._float32_to_bfloat16_vectorized(f32_array)
    
    def _load_weights(self, model_path: Path):
        """加载所有权重文件（强制转换为BF16）- 优化版本"""
        import gc
        
        # 查找所有 safetensors 文件
        tensor_files = sorted(model_path.glob("*.safetensors"))
        if not tensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        total_tensors = 0
        total_original_size_mb = 0
        total_bf16_size_mb = 0
        
        for file in tensor_files:
            
            # 使用 safetensors 加载
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data:
                for name in data.keys():
                
                    # 获取张量
                    tensor = data.get_tensor(name)
                    
                    # 确保是float32
                    if tensor.dtype != np.float32:
                        tensor = tensor.astype(np.float32)
                    
                    original_size_mb = tensor.nbytes / (1024 * 1024)
                    total_original_size_mb += original_size_mb
                    
                    # 根据大小选择转换方法
                    if original_size_mb > 200:  # 对于大于200MB的张量使用分块转换
                        print(f"    Size: {original_size_mb:.2f} MB - using chunked conversion")
                        bf16_tensor = self._float32_to_bfloat16_chunked(tensor, chunk_size_mb=200)
                    else:
                        bf16_tensor = self._float32_to_bfloat16_vectorized(tensor)
                    
                    bf16_size_mb = bf16_tensor.nbytes / (1024 * 1024)
                    total_bf16_size_mb += bf16_size_mb
                    
                    # 加载到C++后端
                    data_ptr = bf16_tensor.ctypes.data_as(ctypes.c_void_p)
                    shape = tensor.shape
                    ndim = len(shape)
                    shape_arr = (ctypes.c_int64 * ndim)(*shape)
                    
                    result = LIB_LLAISYS.llaisysQwen2ModelLoadTensor(
                        self._model_ptr,
                        name.encode('utf-8'),
                        data_ptr,
                        ndim,
                        shape_arr
                    )
                    
                    if result != 0:
                        print(f"    Warning: Failed to load tensor {name}, error code: {result}")
                
                    total_tensors += 1
                    
                    # 强制释放内存
                    del tensor
                    del bf16_tensor
                    gc.collect()
        
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY:")
        print(f"  Total tensors loaded: {total_tensors}")
        print(f"  Original float32 size: {total_original_size_mb:.2f} MB")
        print(f"  BF16 size: {total_bf16_size_mb:.2f} MB")
        print(f"  Memory saved: {total_original_size_mb - total_bf16_size_mb:.2f} MB")
        print(f"  Reduction: {(total_original_size_mb - total_bf16_size_mb) / total_original_size_mb * 100:.1f}%")
        print(f"{'='*60}")
        
    def __del__(self):
        """清理模型"""
        if hasattr(self, '_model_ptr') and self._model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model_ptr)
            self._model_ptr = None
    
    def generate(self, inputs: Sequence[int], max_new_tokens: int = None, 
             top_k: int = 1, top_p: float = 0.8, temperature: float = 0.8) -> List[int]:
        """
        生成文本
        
        Args:
            inputs: 输入token序列
            max_new_tokens: 最大生成token数
            top_k: top-k采样参数（当前仅支持argmax，top_k=1）
            top_p: top-p采样参数（当前未使用）
            temperature: 温度参数（当前未使用）
            
        Returns:
            生成的token序列（包含输入）
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_seq_len - len(inputs)
        
        if max_new_tokens <= 0:
            return list(inputs)
        
        # 当前仅支持argmax采样（top_k=1）
        if top_k != 1:
            print(f"Warning: Only argmax sampling (top_k=1) is currently supported. Using argmax instead of top_k={top_k}")
        
        if temperature != 1.0:
            print(f"Warning: Temperature scaling is not implemented. Using temperature=1.0 instead of {temperature}")
        
        if top_p != 1.0:
            print(f"Warning: Top-p sampling is not implemented. Using top_p=1.0 instead of {top_p}")
        
        # 转换输入为列表
        input_tokens = list(inputs)
        generated_tokens = []
        
        print(f"Starting generation with {len(input_tokens)} input tokens, max_new_tokens={max_new_tokens}")
        print(f"Input tokens: {input_tokens}")
        
        for i in range(max_new_tokens):
            # 准备输入数组
            input_array = (c_int64 * len(input_tokens))(*input_tokens)
            
            # 准备输出数组
            output_array = (c_int64 * 1)(0)
            
            print(f"\n[Step {i+1}] Calling inference with {len(input_tokens)} tokens")
            
            try:
                result = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model_ptr,        # c_void_p
                    input_array,            # POINTER(c_int64)
                    len(input_tokens),      # c_size_t
                    output_array,           # POINTER(c_int64)
                    1                       # c_size_t
                )
                print(f"[Step {i+1}] Inference result: {result}")
                
                if result <= 0:
                    print(f"Warning: Inference failed at step {i}, result={result}")
                    break
                
                # 获取生成的token
                next_token = output_array[0]
                print(f"[Step {i+1}] Generated token: {next_token}")
                generated_tokens.append(next_token)
                
                # 更新输入序列
                input_tokens.append(next_token)
                
                # 如果序列太长，截断到最大长度
                if len(input_tokens) > self.max_seq_len:
                    input_tokens = input_tokens[-self.max_seq_len:]
                
                # 检查是否达到结束标记
                eos_token = self.config.get("eos_token_id", 151643)
                if next_token == eos_token:
                    print(f"Generated EOS token at step {i+1}")
                    break
                
            except Exception as e:
                print(f"[Step {i+1}] Exception during inference: {e}")
                break

            
            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{max_new_tokens} tokens")
        
        print(f"Generation completed. Generated {len(generated_tokens)} tokens")
        print(f"All generated tokens: {generated_tokens}")
        
        # 返回完整序列（输入+生成）
        return list(inputs) + generated_tokens