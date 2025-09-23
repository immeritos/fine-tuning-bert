from datasets import load_from_disk

dataset = load_from_disk(r"/mnt/g/projects/hugging-face-learning/data/COVID-19_weibo_emotion")

for i in dataset:
    print(i)