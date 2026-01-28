from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 下载到指定目录
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="E:\\AI\\model"  # 自定义保存目录
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="E:\\AI\\model"
)

# 保存到本地
model.save_pretrained("E:\\AI\\model\\DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer.save_pretrained("E:\\AI\\model\\DeepSeek-R1-Distill-Qwen-1.5B")