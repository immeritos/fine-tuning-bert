# PyTorch 知识框架（面向 BERT 微调与大模型场景）

> 本文汇总了我们对话中的要点：从 **PyTorch 基础**、**BERT 微调常用实践**，到 **分布式计算图 / 动静态图**、**大模型训练与部署** 的全景框架。配合示例代码，便于查阅与实战。

---

## 目录
1. [总览：PyTorch 在大模型中的角色](#总览pytorch-在大模型中的角色)
2. [张量与数据类型（`torch.tensor` vs `torch.long`）](#张量与数据类型torchtensor-vs-torchlong)
3. [Dataset / DataLoader / `collate_fn` 与 Tokenizer](#dataset--dataloader--collate_fn-与-tokenizer)
4. [模型输入与输出：logits 形状与概率](#模型输入与输出logits-形状与概率)
5. [Transformers 默认 `loss` 机制与自定义损失](#transformers-默认-loss-机制与自定义损失)
6. [优化器、参数分组与学习率调度](#优化器参数分组与学习率调度)
7. [训练循环：AMP、梯度裁剪、梯度累积](#训练循环amp梯度裁剪梯度累积)
8. [类不平衡对策与评估指标](#类不平衡对策与评估指标)
9. [模型保存/加载：`state_dict`、checkpoint 与 `.pt` 文件](#模型保存加载state_dictcheckpoint-与-pt-文件)
10. [推理与长文本滑动窗口（对比 RAG chunking）](#推理与长文本滑动窗口对比-rag-chunking)
11. [计算图与执行模式：动态图、静态图与分布式计算图](#计算图与执行模式动态图静态图与分布式计算图)
12. [大规模训练：DDP / FSDP / ZeRO / 激活检查点](#大规模训练ddp--fsdp--zero--激活检查点)
13. [推理加速与部署：`torch.compile`、ONNX、量化](#推理加速与部署torchcompileonnx量化)
14. [调试、可视化与性能分析](#调试可视化与性能分析)
15. [常见坑位与排查清单](#常见坑位与排查清单)
16. [附：常用代码片段 Cheat Sheet](#附常用代码片段-cheat-sheet)

---

## 总览：PyTorch 在大模型中的角色
- **基础计算**：张量运算、自动求导（Autograd）。
- **模型构建/训练**：`nn.Module`、优化器、调度器、AMP、梯度累积。
- **规模化训练**：DDP、FSDP、ZeRO、模型并行、激活检查点。
- **推理与部署**：`torch.compile`、TorchScript/ONNX、TensorRT/TVM、量化/蒸馏/裁剪、服务化。
- **生态扩展**：Transformers、Diffusers、TorchVision/Audio、RL 等。

---

## 张量与数据类型（`torch.tensor` vs `torch.long`）
- **`torch.tensor(...)`**：创建张量的**函数**；可指定 `dtype`、`device`。
- **`torch.long`**：`int64` 的 **dtype 标识**，常用于 `input_ids`/`labels`。
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.tensor([1,2,3])                 # 默认 float32
b = torch.tensor([1,2,3], dtype=torch.long, device=device)
a_long = a.to(torch.long)
```
- NLP 微调：`input_ids`/`attention_mask`/`token_type_ids`/`labels` 大多为 **long**；概率/损失为 **float**。

---

## Dataset / DataLoader / `collate_fn` 与 Tokenizer
- `collate_fn` 负责 **batch 内对齐与打包**；输出 `dict[str, torch.Tensor]`。
- Tokenizer 来自 **Hugging Face Transformers**（非 PyTorch 原生）。
```python
def collate_fn(batch, tokenizer, max_length=256):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts),
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")  # 返回 BatchEncoding，行为同 dict
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

# DataLoader 性能：
loader = DataLoader(ds, batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True,
                    collate_fn=lambda b: collate_fn(b, tokenizer))
```
- `pin_memory=True` + `.to(device, non_blocking=True)` 提升主机→GPU 传输效率。

---

## 模型输入与输出：logits 形状与概率
- **logits** 是未归一化分数，不是概率。
- 维度习惯：**[B, C]**（批大小 × 类别数）。
```python
outputs = model(**batch)        # 包含 logits
logits = outputs.logits         # [B, C]
probs  = torch.softmax(logits, dim=-1)  # 多分类
preds  = probs.argmax(dim=-1)           # [B]
# 二分类单节点：logits [B, 1] → sigmoid → 概率
```
- 为什么不是一维？因为 **batch** 与 **多类** 都需要对应维度。

---

## Transformers 默认 `loss` 机制与自定义损失
- `BertForSequenceClassification`：有 `labels` → **CrossEntropyLoss**。
- 回归（`num_labels=1` + float labels）→ **MSELoss**。
- Token 分类（NER）→ **CrossEntropyLoss**（忽略 `-100`）。
- QA 任务：`start_logits`/`end_logits` 各自交叉熵再平均。
- 使用纯 backbone（`BertModel`）时 **不会**自动算 loss，需要自己加 head+loss。
```python
# 自定义加权交叉熵（类不平衡）
weights = torch.tensor([w0, w1, ...], dtype=torch.float, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
loss = criterion(logits, labels)
```

---

## 优化器、参数分组与学习率调度
```python
no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
param_groups = [
    {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(param_groups, lr=2e-5)

from transformers import get_linear_schedule_with_warmup
num_training_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)
```

---

## 训练循环：AMP、梯度裁剪、梯度累积
```python
scaler = torch.cuda.amp.GradScaler()
model.train()
for step, batch in enumerate(train_loader):
    batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
    with torch.cuda.amp.autocast():
        out = model(**batch)
        loss = out.loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```
- **梯度累积**：显存不够时用更大“有效 batch”。

---

## 类不平衡对策与评估指标
- **数据**：`WeightedRandomSampler`、过/下采样、数据增强。
- **损失**：加权交叉熵、Focal Loss、标签重映射/合并。
- **阈值**：按类设定不同阈值或成本矩阵。
- **指标**：
  - `accuracy`（总体），`macro F1`（各类平均，抗不平衡）
  - `ROC-AUC` vs `PR-AUC`：极不平衡时 **PR-AUC 更可靠**
- **混淆矩阵**：定位“易混类”（如“悲伤/愤怒”），指导采样/增强/后处理。

---

## 模型保存/加载：`state_dict`、checkpoint 与 `.pt` 文件
- `state_dict()` 含 **参数/缓冲区**（模型）或 **动量/超参**（优化器）等状态。
- **推理部署**：仅需 `model.state_dict()`；
- **断点续训**：还需 `optimizer`、`scheduler` 的 state。
- `.pt/.pth`：pickle 序列化的二进制文件；推荐用 `torch.load`。
```python
# 保存
torch.save({"model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sch": scheduler.state_dict()}, "checkpoint_best.pt")

# 加载
ckpt = torch.load("checkpoint_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["opt"])
scheduler.load_state_dict(ckpt["sch"])
```

---

## 推理与长文本滑动窗口（对比 RAG chunking）
- **滑动窗口**：为绕过 `max_length` 限制切片 → 片段预测 → 规则聚合（投票/概率平均/位置加权）。
- **与 RAG 的区别**：两者都“切块”，但滑窗是 **长度适配的分类策略**；RAG 是 **检索增强**（召回外部知识再生成/理解）。

---

## 计算图与执行模式：动态图、静态图与分布式计算图
- **计算图**：记录前向运算关系，支持自动求导。
- **静态图**：先定义后执行，图优化充分、部署高效，调试不便（典：TF 1.x）。
- **动态图**：边执行边构图，灵活易调试（典：PyTorch, TF2 eager）。
- **分布式计算图**：将图切分到多 GPU/多节点执行，自动插入通信（如 AllReduce）以同步梯度/参数。

---

## 大规模训练：DDP / FSDP / ZeRO / 激活检查点
- **DDP**（数据并行）：每 GPU 一份模型，反向 **AllReduce** 梯度。
- **FSDP / ZeRO**：分片参数/优化器状态/梯度，显存占用显著下降。
- **模型并行**：管道并行 / 张量并行，适配超大模型。
- **激活检查点**：丢弃中间激活，反向时重算，**省显存换算力**。

---

## 推理加速与部署：`torch.compile`、ONNX、量化
- **`torch.compile`（PyTorch 2.x）**：图捕获与算子融合，训练/推理加速（需验证收益）。
- **ONNX 导出**：连通 TensorRT/TVM/OpenVINO 等推理引擎。
- **量化/蒸馏/裁剪**：在延迟/吞吐/能耗之间权衡。
- **服务化**：TorchServe、Triton Inference Server、HF Text Generation Inference。

---

## 调试、可视化与性能分析
- **可视化**：TensorBoard、Weights & Biases。
- **Profiler**：时间/内存热点、算子分布（`torch.profiler`）。
- **Hook**：前/后向钩子，观察中间激活与梯度。
- **注意力/Top-k 可视化**：诊断为主，避免过度解读。

---

## 常见坑位与排查清单
- dtype 不匹配：`labels`/`input_ids` 应为 `long`，概率/损失为 `float`。
- 形状不匹配：`logits [B,C]` vs `labels [B]`。
- 忘记 `model.train()` / `model.eval()`。
- DataLoader 卡住：Windows 试 `num_workers=0`；Linux 合理设置 workers。
- 显存：AMP、梯度累积、合理 `max_length`、激活检查点。
- 断点恢复：别忘了加载优化器与调度器 state。

---

## 附：常用代码片段 Cheat Sheet
```python
# 设随机种子（可复现）
def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 评估循环（多分类）
model.eval(); all_p, all_y = [], []
with torch.no_grad():
    for batch in dev_loader:
        y = batch["labels"]
        x = {k:v.to(device) for k,v in batch.items() if k != "labels"}
        logits = model(**x).logits
        pred = logits.argmax(-1).cpu()
        all_p.append(pred); all_y.append(y)
all_p = torch.cat(all_p); all_y = torch.cat(all_y)

# 二分类（单节点）概率与阈值
logits = model(**x).logits            # [B, 1]
probs  = torch.sigmoid(logits).squeeze(-1)  # [B]
preds  = (probs > 0.5).long()
```
---

> 如果你希望，我可以在此框架基础上，补充与你仓库脚本（`trainer.py / inference.py / metrics.py`）一一对应的段落与调用示例，形成“项目定制版手册”。
