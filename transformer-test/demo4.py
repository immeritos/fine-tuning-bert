from datasets import load_dataset

ds = load_dataset("ndiy/ChnSentiCorp")

ds.save_to_disk("./data/ChnSentiCorp_arrow")

import os
os.makedirs("./data/ChnSentiCorp_csv", exist_ok=True)
for split in ds.keys():  # 例如: 'train', 'validation', 'test'
    ds[split].to_csv(f"./data/ChnSentiCorp_csv/{split}.csv", index=False)
print(ds)