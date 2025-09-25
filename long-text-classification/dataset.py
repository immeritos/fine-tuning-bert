from __future__ import annotations
from typing import Callable, Optional, Tuple, Any, Dict, List
from torch.utils.data import Dataset
from datasets import load_from_disk, DatasetDict, Dataset as HFDataset 

DATA_DIR = "/mnt/g/projects/hugging-face-learning/data/ThuCnews"

class MyDataset(Dataset):
    """
    Sliding window segmentation for long text dataset
    return chuck-level samples, including information such as doc_id for document-level aggregation
    """
    def __init__(
        self, 
        split: str,
        data_dir: str = DATA_DIR,
        transform: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        stride: int = 160,
        add_special_tokens: bool = True,
    ) -> None:
        
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be 'train'|'validation'|'test', receive: {split}")
        
        # load data from disk
        ds: DatasetDict = load_from_disk(data_dir) # type: ignore # A dictionary that stores multiple data splits
        if split not in ds:
            raise ValueError(f"Dataset doesn't have split: {split}; {list(ds.keys())} is available")
        
        self.dataset: HFDataset = ds[split] # To avoid confusion with cumstomized Dataset class.
        self.transform = transform # a preprocessing function to clean/normalize the text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.add_special_tokens = add_special_tokens
        
        # basic field validation
        for key in ("text", "label"):
            if key not in self.dataset.column_names:
                raise KeyError(f"The data is missing '{key}', in fact: {self.dataset.column_names}")
            
        if self.tokenizer is None:
            raise ValueError(
                "MyDataset for long-text chunking requires a HuggingFace tokenizer. "
                "Please pass 'tokenizer=AutoTokenizer.from_pretraind(...)`."
            )
        
        if self.max_length < 8:
            raise ValueError("`max_length` too small; should be >= 16, typically 384/512 etc. ")
        
        # Pre-build "chunk index" to transform variable-length documents into fixed-length sample views
        self._build_chunk_index()
    
    def _build_chunk_index(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        content_len = self.max_length - (2 if self.add_special_tokens else 0)
        
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        if self.add_special_tokens and (cls_id is None or sep_id is None):
            raise ValueError("Tokenizer must have CLS/SEP token ids when add_special_tokens=True. ")
        
        for doc_idx in range(len(self.dataset)):
            item: Any = self.dataset[doc_idx]
            text: str = item["text"]
            label: int = int(item["label"])
            
            if self.transform is not None:
                text = self.transform(text)
                
            enc = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_id=False,
            )
            ids: List[int] = enc["input_ids"]
            
            # empty document handling: At least one empty segment is generated
            if len(ids) == 0:
                if self.add_special_tokens:
                    chunk_ids = [cls_id, sep_id]
                else:
                    chunk_ids = []
                self.chunks.append({
                    "doc_id": doc_idx,
                    "chunk_idx": 0,
                    "num_chunks": 1,
                    "input_ids": chunk_ids,
                    "label": label,
                })
                continue
            
            # Generate sliding window slices
            starts: List[int] = []
            step = max(1, content_len - self.stride)
            for start in range(0, len(ids), step):
                starts.append(start)
                if start + content_len >= len(ids):
                    break
                
            num_chunks = len(starts)
            for k, start in enumerate(starts):
                piece = ids[start:start + content_len]
                if self.add_special_tokens:
                    piece = [cls_id] + piece + [sep_id]
                    
                self.chunks.append({
                    "doc_id": doc_idx,
                    "chunk_idx": k,
                    "num_chunks": num_chunks,
                    "input_ids": piece,
                    "label": label,
                })
            
        
    # get the length of the dataset, i.e. total number of chunks
    def __len__(self) ->int:
        return len(self.chunks)
    
    # get the i-th chunk
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.chunks[idx]
        ids: List[int] = meta["input_ids"]
        
        sample = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "token_type_ids": [0] * len(ids),
            "label": int(meta["label"]),
            "doc_id": int(meta["doc_id"]),
            "chunk_idx": int(meta["chunk_idx"]),
            "num_chunks": int(meta["num_chunks"]),
        }

        return sample
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")
    ds = MyDataset("validation", tokenizer=tok, max_length=512, stride=160)