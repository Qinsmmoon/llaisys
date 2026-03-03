#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2 对话脚本 - 只使用 llaisys 模型
支持多轮对话和流式输出
"""

import argparse
import sys
import io
import time
from typing import List, Dict, Optional
from transformers import AutoTokenizer
import llaisys
from llaisys import DeviceType

# 设置标准输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

class Qwen2Chat:
    """Qwen2 对话类"""
    
    def __init__(self, model_path: str, device: str = "cpu", max_seq_len: int = 4096):
        """
        初始化对话模型
        
        Args:
            model_path: 模型路径
            device: 运行设备 ("cpu" 或 "nvidia")
            max_seq_len: 最大序列长度
        """
        print(f"正在加载模型从: {model_path}")
        print(f"设备: {device}")
        
        # 转换设备名称
        device_type = DeviceType.CPU if device == "cpu" else DeviceType.NVIDIA
        
        # 加载 tokenizer
        print("加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        print("加载 llaisys 模型...")
        self.model = llaisys.models.Qwen2(model_path, device_type, max_seq_len)
        
        # 对话历史
        self.messages: List[Dict[str, str]] = []
        
        # 模型配置
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        
        print("模型加载完成！")
        print(f"词汇表大小: {len(self.tokenizer)}")
        print(f"EOS token ID: {self.eos_token_id}")
        print()
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        应用对话模板
        
        Args:
            messages: 对话历史
            
        Returns:
            格式化后的对话文本
        """
        # 使用 transformers 的 chat_template
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def _format_history(self) -> str:
        """格式化对话历史"""
        return self._apply_chat_template(self.messages)
    
    def chat(
        self,
        user_input: str,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        stream: bool = False
    ) -> str:
        """
        单轮对话
        
        Args:
            user_input: 用户输入
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k采样参数
            top_p: top-p采样参数
            do_sample: 是否采样
            stream: 是否流式输出
            
        Returns:
            模型回复
        """
        # 添加用户消息到历史
        self.messages.append({"role": "user", "content": user_input})
        
        # 格式化对话历史
        prompt = self._format_history()
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt)
        
        print(f"\n[用户] {user_input}")
        print(f"[助手] ", end="", flush=True)
        
        if stream:
            # 流式生成
            full_response = ""
            for token_id in self.model.stream_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            ):
                # 解码当前token
                token_text = self.tokenizer.decode(token_id, skip_special_tokens=True)
                print(token_text, end="", flush=True)
                full_response += token_text
                
                # 检查是否结束
                if token_id == self.eos_token_id:
                    break
            print()  # 换行
        else:
            # 一次性生成
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
            
            # 只提取新生成的token
            new_ids = output_ids[len(input_ids):]
            full_response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            print(full_response)
        
        # 添加助手回复到历史
        self.messages.append({"role": "assistant", "content": full_response})
        
        return full_response
    
    def chat_loop(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        stream: bool = True,
        system_prompt: Optional[str] = None
    ):
        """
        交互式对话循环
        
        Args:
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k采样参数
            top_p: top-p采样参数
            do_sample: 是否采样
            stream: 是否流式输出
            system_prompt: 系统提示词
        """
        print("=" * 60)
        print("Qwen2 对话系统启动！")
        print("=" * 60)
        print("命令:")
        print("  /clear   - 清除对话历史")
        print("  /reset   - 重置模型状态")
        print("  /exit    - 退出对话")
        print("  /help    - 显示帮助")
        print("=" * 60)
        print()
        
        # 添加系统提示词
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
            print(f"[系统] {system_prompt}\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n[用户] ").strip()
                
                # 处理命令
                if user_input == "/exit":
                    print("[系统] 再见！")
                    break
                elif user_input == "/clear":
                    self.messages = []
                    if system_prompt:
                        self.messages.append({"role": "system", "content": system_prompt})
                    print("[系统] 对话历史已清除")
                    continue
                elif user_input == "/reset":
                    self.model.reset()  # 注意：需要在C++端实现reset接口
                    print("[系统] 模型状态已重置")
                    continue
                elif user_input == "/help":
                    print("命令:")
                    print("  /clear   - 清除对话历史")
                    print("  /reset   - 重置模型状态")
                    print("  /exit    - 退出对话")
                    print("  /help    - 显示帮助")
                    continue
                elif not user_input:
                    continue
                
                # 进行对话
                start_time = time.time()
                response = self.chat(
                    user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    stream=stream
                )
                end_time = time.time()
                
                # 显示统计信息
                elapsed = end_time - start_time
                print(f"\n[耗时: {elapsed:.2f}秒]")
                
            except KeyboardInterrupt:
                print("\n[系统] 对话中断")
                break
            except Exception as e:
                print(f"\n[错误] {e}")
                import traceback
                traceback.print_exc()
    
    def reset(self):
        """重置对话状态"""
        self.messages = []
        # 注意：需要在C++端实现reset_cache接口
        # self.model.reset()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen2 对话脚本")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "nvidia"],
        help="运行设备"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="温度参数"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="top-k采样参数"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="top-p采样参数"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="不使用采样（使用贪心解码）"
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="不使用流式输出"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="系统提示词"
    )
    parser.add_argument(
        "--one_shot",
        type=str,
        default=None,
        help="单次对话模式，输入问题后退出"
    )
    
    args = parser.parse_args()
    
    # 创建对话实例
    chat = Qwen2Chat(args.model, args.device)
    
    if args.one_shot:
        # 单次对话模式
        print(f"问题: {args.one_shot}")
        response = chat.chat(
            args.one_shot,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.no_sample,
            stream=not args.no_stream
        )
        print(f"\n回答: {response}")
    else:
        # 交互式对话模式
        chat.chat_loop(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.no_sample,
            stream=not args.no_stream,
            system_prompt=args.system
        )


if __name__ == "__main__":
    main()