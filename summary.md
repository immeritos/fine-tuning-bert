# 微调 BERT 做评论情感二分类：知识总结框架

## 1. 任务与数据

- **任务定义**
  - 输入：中文评论文本（短文本为主）
  - 输出：二分类标签（正向/负向）
  - 评价目标：总体准确率与宏平均 F1

- **数据来源与划分**
  - 数据集：ChnSentiCorp（或你的实际数据）
  - 划分方式：train/validation/test 比例与样本量
  - 清洗与规范化：去重、空文本、超长样本处理（保留/截断）
  - 类别分布：是否存在**类不平衡**（占比、对训练的影响）

- **标注规范**
  - label 映射：{0: negative, 1: positive}
  - 存储格式：列名（`text`, `label`）与文件/arrow 存储


## 2. 模型与分词器

- **预训练模型**
  - `bert-base-chinese`（vocab、字符级/子词级特点）
  - 最大长度：512 token（含 [CLS]/[SEP]）
  - `from_pretrained` 会：
    - 下载/读取对应的模型配置（config.json）
    - 下载/读取模型权重（pytorch_model.bin）
    - 构建一个 PyTorch BertModel 实例，并把权重加载进去
  - `config`：是模型的配置对象（BertConfig），存放了模型的超参数，比如：
    - `hidden_size`（每层隐向量维度，BERT-base = 768）
    - `num_attention_heads`（注意力头数）
    - `num_hidden_layers`（Transformer 层数）
    - `vocab_size`（词表大小）等。
  - `hidden_size`：表示 BERT 每个 token 表示向量的维度:
    - 在 BERT-base 里是 768
    - 在 BERT-large 里是 1024

- **分词器（Tokenizer）**
  - `AutoTokenizer/BertTokenizerFast`
  - 关键配置：`max_length`、`truncation`、`padding`、`return_tensors`

- **下游头（Classification Head）**
  - 结构：`[CLS]` → Dropout → Linear(num_labels=2)
  - Dropout作用:
    - 在训练时随机“丢掉”一部分神经元的输出（例如 10%–30%），强迫模型不要过分依赖某些特征，提升泛化能力。
    - 在小数据集里，BERT 的 1 亿多参数很容易过拟合 → 加 Dropout 可以缓解。
  - 池化对比（可选）：`[CLS]` vs mean pooling（经验与结论）



## 3. 数据加载与批处理（Python/PyTorch）

- **Dataset 设计（`dataset.py`）**
  - 返回 `(text, label)` 的最小接口
  - 可选 `transform`（清洗、正则化）
  - 错误与异常处理（缺列、空文本）

- **Collate 函数**
  - 动态 padding：`DataCollatorWithPadding(tokenizer=...)`
  - 旧管线：collate 内完成**批量分词**
  - 张量化与 dtypes（`input_ids`, `attention_mask`, `labels`）
  - 流程： 
    1. 拿到样本（"我爱北京天安门"）
    2. tokenizer 编码
    3. collator 打包 batch

- **DataLoader**
  - `batch_size`、`shuffle`、`num_workers`
  - 长度截断策略与显存权衡



## 4. 训练配置（Trainer / 自写训练循环）

- **损失函数**
  - 二分类：`nn.CrossEntropyLoss`: 衡量预测概率和真实分布的差异
  - 参数logits：模型的原始输出（未 softmax），PyTorch 会自动处理

- **优化器与调度器**
  - `AdamW`/`AdamWeightDecay`（LR、`weight_decay`）
  - `gradient_accumulation_steps`（小显存利器）

- **正则化与稳定性**
  - `max_grad_norm`（梯度裁剪）
  - 冻结骨干（可选）：冻结 BERT encoder、只训分类头的收益与风险

- **混合精度与设备**
  - AMP：`fp16=True`（节省显存、加速）
  - 单卡/多卡（DDP）注意事项：随机性控制、`seed`、`torch.backends.cudnn.deterministic`
  - 使用`GradScaler`，让 FP16 训练更稳定（避免梯度下溢/溢出）：
    1. 正向传播：在 autocast() 里部分算子用 FP16 运行。
    2. 反向传播前：Scaler 会把 loss 乘以一个“缩放因子”（scale factor），让梯度变大，避免梯度在 FP16 下变成 0（下溢）。
    3. 反向传播后：Scaler 会再把梯度缩放回来。
    4. 动态调整：如果检测到溢出，Scaler 会自动减小 scale factor，下次迭代再试。

- **早停与检查点**
  - EarlyStopping（patience、监控指标）
  - 在每个 batch 开始前都要 `zero_grad()`, 把梯度清零
  - `load_best_model_at_end=True`；保存路径规范（目录 vs 文件）



## 5. 评测体系（metrics 与验证集）

- **指标定义**
  - Accuracy（整体）
  - Macro Precision/Recall/F1（更关注类别均衡性）
  - 可选：ROC-AUC、PR-AUC（阈值敏感型分析）

- **验证策略**
  - 验证频率：每 epoch / 固定步数
  - 最优模型选择标准：`metric_for_best_model="accuracy"`（或 F1）

- **错误分析**
  - 混淆矩阵：常错对、错误案例抽样
  - 误判归因：讽刺、否定词、领域外词、表情/emoji、拼写

- **阈值与校准（可选）**
  - 从 `softmax` 概率出发的阈值选择
  - 温度缩放/Platt scaling（部署前置信度更稳）


## 6. 推理与上线（`inference.py`）

- **单条/批量推理流程**
  - 文本 → 分词编码（与训练一致的 `max_length`/截断侧）
  - `model.eval()` + `torch.no_grad()`；`softmax` 概率与类别
  - Top-k 打印与可解释性（关键词、注意力可视化：可选）

- **性能与资源**
  - 批大小、延迟、吞吐估计
  - CPU vs GPU 推理差异（`torch.set_num_threads`）

- **健壮性**
  - OOV、emoji、英文串、超长文本处理策略
  - 非法输入（空串、全空白）兜底


## 7. 常见坑位与排查清单

- **分词与长度**
  - 以 token 为单位，而非字符；特殊符号数量（[CLS]/[SEP]）
  - 训练/验证/推理三处 `max_length` 和 `truncation_side` 一致

- **标签对齐**
  - `labels` dtype=int64（`long`）；与 `num_labels` 对上
  - `id2label/label2id` 映射一致、随模型一起保存

- **Trainer 配置**
  - `remove_unused_columns=False`（避免字段被自动丢弃）
  - `processing_class` 无效（如版本不支持）；统一用 `data_collator`

- **保存/加载**
  - `from_pretrained(path)` 需要**目录**；检查目录内容完整（`config.json`, `pytorch_model.bin`, `tokenizer.json` 等）

---

# 微博六情感分类：微调 BERT 知识总结框架


## 1. 任务与数据

- **任务定义**
  - 输入：微博短文本（含口语、表情、@用户、#话题标签#、链接等）
  - 输出：六分类情感标签（单标签多分类）
  - 目标：Macro-F1 / Weighted-F1 为主，兼顾整体 Accuracy

- **数据来源与划分**
  - 数据集与规模（样本数、每类占比）
  - 划分策略：**分层抽样**（stratified split）确保各类在 Train/Val/Test 中比例一致
  
- **领域偏移（domain shift）**:
  - 常见类型：
    1. 词汇、话题、emoji 用法变了
    2. 疫情期“恐惧/愤怒”比例特别高，上线后“乐/惊讶”增多
    3. 同样一句“离谱”，一年后语义色彩从愤怒变成戏谑或中性。
    4. 词义、梗、热词随时间演化
  - 如何定位：
    - 域分层评测：按话题/时间切分做 holdout（疫情相关 vs 非疫情），对比 Macro-F1
    - 置信度/校准检查：ECE、可靠性图，观察过/欠置信
    - 词分布漂移：对比 TF-IDF/子词频率前后 KL 距离、OOV 率变化
    - 错误聚类：聚类误判样本，看是否集中在某些新话题/新表情/新体裁
    - 学习曲线：少量目标域增量数据对性能的提升斜率，能反映偏移严重度
  - 缓解手段：
    1. 预处理对齐：确保清洗、分词、emoji/@/URL/Hashtag 规则与训练一致
    2. 阈值与校准：在目标域验证集上重选阈值；做温度缩放/Platt 校准，修正置信度
    3. 少量目标域微调（TAPT/DAPT）
    4. 增量标注 + 主动学习：挑选不确定/代表性样本少量标注，周期性小步微调

- **清洗与规范化**
  - 协议：统一处理 `@mention`、`#Hashtag#`、URL、表情/Emoji、拉丁字母/数字、繁简转换
  - 去重、空文本、极短文本（如长度<2）处理
  - 表情词典/emoji 映射策略（可选：保留为 token、转情感占位符）

- **标签体系**
  - 类别集合与 `id2label/label2id`
  - 是否存在**近义或易混类**（如“悲伤 vs 厌恶/愤怒”）：
    - 拉低两类的 F1
    - 在混淆矩阵里形成显著的 off-diagonal
    - 造成模型对负向细分类的边界不稳定和过度自信
  - 类别不平衡分析与对策（见 §4）



## 2. 模型与分词器

- **预训练模型**
  - `bert-base-chinese` 或社媒适配模型（若有）
  - 512 token 上限；微博通常较短，可设 `max_length=128~256`

- **分词器**
  - `BertTokenizerFast`；设置：`truncation=True`、`padding=longest/batch`、`return_tensors`
  - Truncation side 选择与理由（right 通常足够；如关注结尾情绪可考 left）

- **分类头**
  - `[CLS]` → Dropout → Linear(num_labels=6)
  - 池化对比（可选）：`[CLS]` vs mean/max pooling
  - 是否启用 LayerNorm：
    - 不更新 encoder 时，头部的 LN 能适配固定表征的动态范围，提升收敛与校准
    - mean pooling 会随有效长度/掩码变化；LN 可“统一量纲”，让线性分类器更好训



## 3. 数据管线与批处理（Python/PyTorch）

- **Dataset 设计**
  - 字段：`text`, `label`；可选 `meta`（时间、是否转发、设备来源等）
  - `transform`：清洗、表情/URL/mention 归一；是否做繁简体转换

- **Collate 函数**
  - `DataCollatorWithPadding(tokenizer=...)` 动态 padding
  - Dtype 校验：`labels` 为 `long`；`input_ids/attention_mask` 为 `long`

- **DataLoader**
  - `batch_size` 与显存平衡（常见 16–64）
  - `num_workers`、pin_memory；随机种子固定



## 4. 训练配置与技巧

- **损失函数**
  - `CrossEntropyLoss`（默认）
  - 类别不平衡策略：
    - `class_weight`（基于 1/freq 或有效样本数 reweighting）
    - Focal Loss（γ=1–2）对长尾、难例更稳（可对比实验）
    - 采样：`WeightedRandomSampler`（与 class_weight 二选一为佳）

- **优化与调度**
  - `AdamW / AdamWeightDecay`；典型 LR：1e-5 ~ 5e-5
  - Warmup（5%–10%）+ 线性降；或 Cosine 退火
  - `gradient_accumulation_steps` 与 `max_grad_norm=1.0`

- **泛化与稳定性（可选增强）**
  - Label Smoothing（ε=0.05~0.1）
  - R-Drop（KL 双向一致性，下降过拟合）
  - 对抗训练（FGM/PGD）在情感分类常有效
  - 优化器增强：SAM/ASAM；EMA（指数滑动平均）
  - 冻结骨干对小数据的收益与边界

- **正则与早停**
  - Dropout（0.1–0.3）
  - EarlyStopping（patience=2–3，以 Val Macro-F1 监控）
  - `load_best_model_at_end=True`


## 5. 评测与分析

- **指标**
  - Accuracy、Macro-F1（主）、Weighted-F1、每类 Precision/Recall/F1
  - 报告：`classification_report`、混淆矩阵（标出最易混对）

- **错误分析**
  - 讽刺/反语、双重否定、语气助词、表情/emoji 引导
  - 同一条微博含多种情绪但标注为单类的“噪声”
  - 主题词掩盖情感（如客观陈述带轻微情绪）

- **鲁棒性检查**
  - 去除 URL/@/话题后性能变化
  - 对 OOV、新词、口语缩写、拼写错误的敏感性
  - 领域外迁移（非疫情微博）性能对比



## 6. 推理与输出

- **单条/批量推理**
  - 与训练一致的分词/长度策略
  - 输出：预测类别 + 置信度；Top-k 排序（用于质检）

- **可解释性（可选）**
  - 关键词高亮（基于梯度/注意力/占位词规则）
  - 对 “@账号/#话题#” 的贡献分析

- **上线注意**
  - 文本清洗与训练一致；空串/超短文本兜底
  - 错误日志与反馈回流（主动发现新俚语/新表情）


## 7. 常见坑位与排查清单

- `num_labels=6`、`id2label/label2id` 与数据一致（加载/保存一致性）
- 类不平衡：**权重与采样不要同时强开**（先试 class_weight 或 Focal 二选一）
- 过拟合：Val Macro-F1 不升时优先尝试 **Label Smoothing / R-Drop / FGM**
- 文本归一与训练一致：emoji/URL/@/Hashtag 的处理
- 保存/加载路径：`from_pretrained(path)` 需要**目录**且包含 `config.json`、权重、tokenizer 文件
- `max_length` 太小会截断情绪证据；太大浪费算力（建议 128–256）

---

# 新闻主题分类（长文本）+ 滑动窗口方案：知识总结框架

## 1. 任务与数据

- **任务定义**
  - 输入：中文新闻报道（长文本，常 > 1000 字）
  - 输出：主题类别（示例：体育/娱乐/科技/教育/游戏/财经 等）
  - 目标指标：文档级 Accuracy、Macro-F1

- **数据来源与划分**
  - 数据集：如 THUCNews/自建新闻库（规模、类别分布）
  - 划分：train / validation / test（是否分层抽样）
  - 领域差异：不同频道写作风格差异、时间分布偏移

- **清洗与正则化**
  - 冗余去除：作者/来源/广告尾注、重复段落
  - 编码与符号：全半角统一、空白符规范、URL/数字表达
  - 超长文本处理：保留全文，交由“切块”机制处理



## 2. 长文本挑战与方案选择

- **BERT 输入上限**
  - `max_position_embeddings=512`（含 `[CLS]/[SEP]`），正文可用 ≈ 510 token
- **为何采用方案 A（滑动窗口切块 + 聚合）**
  - 简单稳定、易落地；兼容现有 `bert-base-chinese` 权重
  - 新闻主题判别具有**局部可见性**（关键词、实体、版块用语分布在段落中）
- **替代方案简述（对比）**
  - 方案 B：层级模型（块级表征 + 上层聚合器）
  - 方案 C：长序列 Transformer（Longformer/BigBird）
  - 方案 D：抽取式摘要后分类
  - 方案 E：扩位嵌入（不作常规解的原因与风险）



## 3. 分词与切块（Tokenizer + Sliding Window）

- **按 token 切块（非按字符）**
  - 需要 `BertTokenizerFast`：确保切块与模型 token 上限对齐
- **关键超参**
  - `max_length`（含 special tokens），推荐 512
  - `stride`（重叠区大小，建议 128–192）
  - `truncation_side` 与一致性（train/val/test 相同）
- **实现要点**
  - 不加 special tokens 先取 `input_ids` → 滑窗切片 → 每片再补 `[CLS]/[SEP]`
  - 空文本兜底（仅 `[CLS][SEP]` 片）
  - 每个 chunk 附带：`doc_id`、`chunk_idx`、`num_chunks`、`label`



## 4. 数据管线与批处理（Dataset / Collate）

- **Dataset（`dataset.py`）**
  - 输入：原始 `text`、`label`
  - 输出：**chunk 级样本**（`input_ids/attention_mask/token_type_ids`，以及 `doc_id/label` 等元信息）
- **Collate**
  - `DataCollatorWithPadding(tokenizer=...)` 动态 padding
  - 保留 `doc_id` 等元信息（转为 `LongTensor`），避免被丢弃
- **DataLoader**
  - `batch_size` 受“多 chunk”影响（酌情减小）
  - `num_workers`、pin_memory、seed 固定



## 5. 模型与训练（文档级聚合损失）

- **模型**
  - 仍用 `BertForSequenceClassification(num_labels=K)`
  - 片段级前向：每个 chunk → `logits ∈ ℝ^K`
- **文档级聚合（训练）**
  - 在一个 batch 内，按 `doc_id` 将同文档的 chunk logits 聚合为 `doc_logits`
  - 聚合函数选择：
    - **mean**（默认、稳健）
    - **max**（“任一强证据即判定”）
    - **logsumexp**（兼顾峰值与累积证据）
  - **一次性计算文档级损失**：`CE(doc_logits, doc_label)` 并反传
- **优化与正则**
  - `AdamW` / `AdamWeightDecay`、LR（1e-5 ~ 5e-5）
  - `max_grad_norm=1.0`、Dropout（head）
  - AMP（fp16）、可选冻结骨干（数据小时试验）



## 6. 验证与测试（**文档级评测**）

- **为什么是文档级**
  - 一篇新闻被切成多个 chunk，不能把 chunk 当独立样本计分
- **评测流程**
  1. 逐 chunk 推理得到 logits
  2. 按 `doc_id` 聚合为 `doc_logits`
  3. 取 `argmax(doc_logits)` 与**该文档标签**比较
- **指标**
  - 文档级 Accuracy、Macro-F1（主）、每类 P/R/F1、混淆矩阵
- **一致性**
  - 训练与评测使用同一种聚合（推荐先统一为 `mean`），也可在评测阶段额外对比 `max`/`logsumexp`



## 7. 推理与上线（Inference）

- **单条长文推理**
  - 文本 → 切块（同训练超参）→ 片段前向 → 聚合 logits → 概率与类别
  - 输出 Top-k 备查；可返回贡献度最高的若干 chunk（易做质检）
- **吞吐与延迟**
  - 块数 ≈ `ceil((n_tokens - 2) / (max_len - 2 - stride))`
  - 批大小、显存与时延的平衡
- **健壮性**
  - 冗余尾注、模板化开头对预测的影响
  - 标点、英文名词、数字、日期等的处理一致性



## 8. 常见坑位与排查清单

- **切块单位**：必须按 **token**，非字符；确保 special tokens 计入长度
- **聚合一致性**：训练与评测聚合策略不一致会导致离线/线上指标偏差
- **元信息丢失**：`doc_id` 在 collate 或 `Trainer` 里被自动丢弃（`remove_unused_columns=False`）
- **评测歧义**：误把 chunk 当样本计算指标 → 结果虚高
- **路径与保存**：`from_pretrained(path)` 需要目录；`id2label/label2id` 与训练一致
- **显存爆炸**：`batch_size × 平均chunks` 过大，需减小 batch 或 stride