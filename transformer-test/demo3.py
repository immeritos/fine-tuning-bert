from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model_name = "bert-base-chinese"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

result = classifier("你好，我是一款语言模型")
print(result)