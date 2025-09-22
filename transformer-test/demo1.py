from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google-bert/bert-base-chinese"
cache_dir = "model/google-bert/bert-base-chinese"

AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型分词器已下载到：{cache_dir}")