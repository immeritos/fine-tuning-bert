from transformers import BertTokenizer

token = BertTokenizer.from_pretrained("bert-base-chinese")

sents = ["酒店太旧了，大堂感觉像三星级的，房间也就是好点的三星级的条件，早餐走了两圈也没有找到可以吃的，太差了",
         "已经贴完了，又给小区妈妈买了一套，值得推荐",
         "屏幕大，本本薄，声音也还过得去，性价比高！",
         "酒店环境很好，就是有一点点偏，交通不是很便利，去哪儿都要打车，关键是不好打"]

# 批量编码
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,
    truncation=True,
    padding="max_length",
    return_tensors=None,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True    
)

print(out)
for k, v in out.items():
    print(k,":",v)
    
vocab = token.get_vocab()
print("阳" in vocab)
print("光" in vocab)
print("阳光" in vocab)

# 添加新词
token.add_tokens(new_tokens=["阳光","大地"])
vocab = token.get_vocab()
print("阳光" in vocab)

# 添加特殊符号
token.add_special_tokens({"eos_token":"[EOS]"})
vocab = token.get_vocab()
print(vocab)

# 编码新句子
out = token.encode(
    text="阳光照在大地上[EOS]",
    text_pair=None,
    truncation=True,
    padding="max_length",
    max_length=10,
    add_special_tokens=True,
    return_tensors=None
)
print(out)
# 解码为源字符串
print(token.decode(out))