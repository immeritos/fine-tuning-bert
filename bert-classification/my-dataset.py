from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataSet(Dataset):
    # 初始化数据
    def __init__(self, split):
        # 从磁盘加载数据
        self.dataset = load_from_disk(r"/mnt/g/projects/hugging-face-learning/data/ChnSentiCorp_arrow")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            print("数据集名称错误！")
    
    # 获取数据集的长度
    def __len__(self):
        return len(self.dataset)
    
    # 对数据作定制化处理
    def __getitem__(self, item):
        pass