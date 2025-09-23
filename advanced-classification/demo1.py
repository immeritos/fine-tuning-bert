from datasets import load_dataset

ds = load_dataset("souljoy/COVID-19_weibo_emotion")

ds.save_to_disk("./data/COVID-19_weibo_emotion")

print(ds)
