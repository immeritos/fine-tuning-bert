from datasets import load_dataset

ds = load_dataset("lansinuote/ChnSentiCorp")

ds.save_to_disk("./data/ChnSentiCorp_arrow")

print(ds)
