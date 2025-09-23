from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    # 初始化数据
    def __init__(self, split):
        # 从磁盘加载数据
        self.dataset = load_from_disk(r"/mnt/g/projects/hugging-face-learning/data/ThuCnews")
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
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label
    
if __name__ == "__main__":
    
    dataset = MyDataset("test")
    
    labels = [label for _, label in dataset]
    
    unique_labels = set(labels)
    
    print("测试集样本总数：", len(dataset))
    print("测试机标签总数：", len(unique_labels))
    print("标签种类：", unique_labels)