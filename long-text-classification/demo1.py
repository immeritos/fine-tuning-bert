from datasets import load_dataset

ds = load_dataset("spiritx2023/ThuCnews")

ds.save_to_disk("./data/ThuCnews")

print(ds)