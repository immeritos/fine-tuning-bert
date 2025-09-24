from __future__ import annotations
from typing import Callable, Optional, Tuple, Any
from torch.utils.data import Dataset
from datasets import load_from_disk, DatasetDict, Dataset as HFDataset 

DATA_DIR = "/mnt/g/projects/hugging-face-learning/data/COVID-19_weibo_emotion"

class MyDataset(Dataset):
    """
    return (text, label), to be used with customized collate_fn
    optimal transform parameter, to be used to preprocess the text (e.g. cleaning or normalization)
    """
    def __init__(
        self, 
        split: str,
        data_dir: str = DATA_DIR,
        transform: Optional[Callable[[str], str]] = None,
    ) -> None:
        
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be 'train'|'validation'|'test', receive: {split}")
        
        # load data from disk
        ds: DatasetDict = load_from_disk(data_dir) # type: ignore # A dictionary that stores multiple data splits
        if split not in ds:
            raise ValueError(f"Dataset doesn't have split: {split}; {list(ds.keys())} is available")
        
        self.dataset: HFDataset = ds[split] # To avoid confusion with cumstomized Dataset class.
        self.transform = transform # a preprocessing function to clean/normalize the text
        
        for key in ("text", "label"):
            if key not in self.dataset.column_names:
                raise KeyError(f"The data is missing '{key}', in fact: {self.dataset.column_names}")
    
    # get the length of the dataset
    def __len__(self) ->int:
        return len(self.dataset)
    
    # cumtomize the dataset
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        item: Any = self.dataset[idx]
        text : str = item["text"]
        label : int = int(item["label"])
        if self.transform is not None:
            text = self.transform(text)
        return text, label
    
if __name__ == "__main__":
    ds = MyDataset("validation")
    print(len(ds), ds[0])
        