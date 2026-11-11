---
id: sft-recipe
title: SFT 全流程配方：数据构造、loss 掩码、packing 与超参经验值
module: 08-sft-peft
tags:
- SFT
- 指令微调
- loss掩码
- packing
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: SFT 的 gain 主要来自数据质量而非数量（LIMA 用 1030 条打赢 52K Alpaca）；loss 必须对 prompt 做 -100
  掩码，packing 必须重置 position_ids 否则 cross-contamination。
source:
- https://arxiv.org/abs/2106.09685
questions:
- SFT 数据量到底要多少条？
- 为什么要做 epoch 2-3 而不是训 10 个 epoch？
- chat template 训练时和推理时不一致会怎样？
- padding 用哪个 token？为什么经常把 pad_token 设成 eos_token？
- 怎么区分过拟合和灾难性遗忘？
---

## 一句话结论

SFT（Supervised Fine-Tuning，有监督微调）的本质不是"教模型知识"，而是**把预训练模型已有的能力对齐到某种交互格式和输出风格上**。所以它的收益上限由基座模型决定，SFT 只负责"解锁"。

三条最硬的工程结论：

1. **数据质量 > 数据数量**。LIMA（Less Is More for Alignment）用 1030 条人工精挑的样本做 SFT，在人类偏好评测上优于用 52000 条 Alpaca 数据训练的模型。数据量从 1K 增到 52000，格式遵循能力提升，但事实正确性和有用性不涨，甚至被噪声拖累。
2. **loss 只能算在 assistant 回复上**。prompt 部分的 token label 必须设成 `-100`（`CrossEntropyLoss` 默认 `ignore_index`）。不掩码会让模型去学"怎么生成问题"，浪费容量并放大复读。
3. **packing 必须配 block-diagonal attention**。不重置 `position_ids`、不切分 attention 的朴素 packing，会让样本 A 的 token 关注到样本 B，即 cross-contamination，长数据上明显掉点。

## 原理推导

### 1. SFT 的目标函数

给定一条样本 $(x, y)$，$x$ 是指令（含 system prompt 和用户输入），$y$ 是目标回复。整个序列拼接后长度为 $T$：

$$\mathcal{L} = -\frac{1}{\sum_{t} m_t} \sum_{t=1}^{T} m_t \cdot \log p_\theta(y_t \mid y_{<t}, x)$$

其中 $m_t \in \{0,1\}$ 是掩码：只有 $y$ 部分的 token 取 1。

注意分母的两种取法，这是个常考点：

- **token-level mean**（主流）：分母是 batch 内所有非忽略 token 总数。长样本天然权重大，等价于"每个 token 等权"。
- **sample-level mean**：先对每条样本内部求平均，再对样本平均。等价于"每条样本等权"，短样本中单个 token 的梯度被放大。

两者在长度差异大的数据集上训练结果不同。多数框架（HF Trainer）默认走 token-level。

### 2. 为什么 prompt 要掩码

设 prompt 有 $P$ 个 token，回复有 $A$ 个 token，$T = P + A$。不掩码时，loss 里 prompt 项占比 $\frac{P}{P+A}$。指令模板是高度重复的（"You are a helpful assistant"、`<|im_start|>user`），这部分梯度方向几乎一致且方差极小，等同于用大量梯度步去拟合一个常量模式。后果：模型对模板过拟合、对真正的回复学得慢、eval loss 看起来很低但实际生成质量差。

一个细节：**是否掩码 loss 与是否在 instructions 上算 loss 的争议**。Shi 等人（Instruction Tuning With Loss Over Instructions, 2024）发现，在**短回复 + 小数据集**场景下，保留 instruction loss 反而有益，它相当于对输入做了正则；但数据量大（> 数十万条）或回复较长时，两者无显著差异。工程上默认掩码，数据少且回复短时可以打开做对比实验。

### 3. padding 与 packing 的取舍

设 batch 内最长序列 $L_{\max}$，第 $i$ 条长度 $L_i$，padding 浪费率：

$$\text{waste} = 1 - \frac{\sum_i L_i}{B \cdot L_{\max}}$$

长度分布长尾时（比如 95% 样本 < 512 token，但 5% 到 4096），不做长度分组的随机 batching 浪费率可达 50%+。

**Padding 方案**：同 batch 内补 pad token，用 `attention_mask` 屏蔽，label 对应位置设为 `-100`。改进手段是 **length-grouped batching**（按长度排序后组 batch），能把浪费率压到 10% 以内，代价是 batch 间不再独立同分布，需要配套 shuffle。

**Packing 方案**：把多条样本首尾相接塞进一条定长序列，浪费率降到接近 0。但必须处理两件事：

- **attention 隔离**：用 block-diagonal 的 attention mask，或调用 FlashAttention 的 varlen 接口传 `cu_seqlens`（每条子样本的累积长度），让不同子样本之间互不 attention。
- **位置编码复位**：每条子样本的 `position_ids` 从 0 重新计数，否则第二条样本看到的位置是 `len(sample1) + j`，RoPE 的相对位置虽然不变（RoPE 只依赖相对距离），但跨样本的相对距离会变得巨大，等于告诉模型"上一段话离你几万 token 远"，与推理时的真实分布不一致。

### 4. 显存与算力估算

全参数 SFT 一个 7B 模型，混合精度下每参数字节数：

| 组件 | dtype | 字节/参数 | 7B 总量 |
|---|---|---|---|
| 模型权重 | bf16 | 2 | 14 GB |
| 梯度 | bf16 | 2 | 14 GB |
| 优化器 m | fp32 | 4 | 28 GB |
| 优化器 v | fp32 | 4 | 28 GB |
| fp32 主权重 | fp32 | 4 | 28 GB |

合计 **16 字节/参数 = 112 GB**，单卡 80GB 装不下，必须上 ZeRO-3 或 8-bit 优化器（8-bit Adam 把 m/v 各压到 1 字节，总开销降到 10 字节/参数 = 70 GB）。

训练 FLOPs 估算（不含 recompute）：

$$\text{FLOPs} \approx 6 N D$$

$N$ 是参数量，$D$ 是训练总 token 数（含 epoch 重复计数）。

## 具体例子

**例 1：给一条样本手写 label 掩码**

tokenizer 后（为可读性用单词表示）：

```
ids:    [<|im_start|>, user, \n, 1, +, 1, 等于几, <|im_end|>, \n, <|im_start|>, assistant, \n, 2, <|im_end|>]
labels: [  -100,      -100, -100,-100,-100,-100, -100,   -100,    -100, -100,      -100,      -100, 2,   <|im_end|>]
```

序列长 15，参与 loss 的 token 只有 2 个（`2` 和 `<|im_end|>`）。注意 `<|im_end|>` **必须保留**进 label，否则模型学不会停止，生成时会一直输出直到撞上 max_new_tokens。

**例 2：10 万条 SFT 数据的训练开销**

设平均每条 512 token，共 $D_1 = 5.12 \times 10^7$ token，训 2 个 epoch，则 $D = 1.024 \times 10^8$。

7B 模型：

$$\text{FLOPs} = 6 \times 7\times10^9 \times 1.024\times10^8 = 4.30 \times 10^{18}$$

单张 A100（bf16 峰值 312 TFLOPS），MFU 取 40%，有效算力 $1.25\times10^{14}$ FLOP/s：

$$t = \frac{4.30\times10^{18}}{1.25\times10^{14}} = 3.44\times10^4 \text{ s} \approx 9.6 \text{ 小时}$$

加 gradient checkpointing（重算前向，约 +33% FLOPs）后约 12.8 小时。

**例 3：padding 浪费实测**

batch size 16，长度分布为：12 条 256 token、3 条 1024 token、1 条 4096 token。

$$\sum L_i = 12\times256 + 3\times1024 + 4096 = 3072 + 3072 + 4096 = 10240$$
$$\text{waste} = 1 - \frac{10240}{16 \times 4096} = 1 - \frac{10240}{65536} = 84.4\%$$

这个 batch 里 84% 的算力在做 padding。按长度分组后，短样本单独组 batch，浪费率可降到 5% 以内，实际吞吐差 3-5 倍。

## 代码速记

```python
import torch
import torch.nn.functional as F

IGNORE = -100


def build_example(prompt_ids, answer_ids, bos_id, eos_id):
    """拼接一条 SFT 样本：只有 answer 段 + eos 参与 loss。"""
    input_ids = [bos_id] + prompt_ids + answer_ids + [eos_id]
    labels = [IGNORE] * (1 + len(prompt_ids)) + answer_ids + [eos_id]
    return input_ids, labels


def collate(examples, pad_id, max_len=2048):
    """examples: List[(input_ids, labels)]，右侧 padding 并对齐 label 掩码。"""
    examples = [e[:max_len] for e in examples]
    L = max(len(ids) for ids, _ in examples)
    B = len(examples)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)      # (L,)
        labels[i, :n] = torch.tensor(lab)
        attn[i, :n] = 1                            # padding 位不参与 attention
    return input_ids, labels, attn


def sft_loss(model, input_ids, labels, attn):
    """token-level mean，等价于 HF 默认行为。"""
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # (B, L, V)
    # 第 t 个位置的 logits 预测第 t+1 个 token，故错位对齐
    shift_logits = logits[:, :-1, :]                                 # (B, L-1, V)
    shift_labels = labels[:, 1:]                                     # (B, L-1)
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),             # (B*(L-1), V)
        shift_labels.reshape(-1),                                    # (B*(L-1),)
        ignore_index=IGNORE,
    )


def pack(examples, max_len, pad_id):
    """packing：返回 position_ids 与 cu_seqlens，保证无 cross-contamination。
    cu_seqlens 传给 FlashAttention varlen 接口做 block-diagonal attention。"""
    packs, cur_ids, cur_lab, cur_pos, cur_cu = [], [], [], [], [0]
    for ids, lab in examples:
        if len(cur_ids) + len(ids) > max_len:                        # 放不下就收尾
            pad = max_len - len(cur_ids)
            cur_ids += [pad_id] * pad
            cur_lab += [IGNORE] * pad
            cur_pos += list(range(len(cur_pos), len(cur_pos) + pad))
            cur_cu.append(max_len)
            packs.append((cur_ids, cur_lab, cur_pos, cur_cu))
            cur_ids, cur_lab, cur_pos, cur_cu = [], [], [], [0]
        cur_ids += ids
        cur_lab += lab
        cur_pos += list(range(len(ids)))     # 关键：每段 position 从 0 重新计数
        cur_cu.append(len(cur_ids))
    if cur_ids:                                                      # 收尾补齐
        pad = max_len - len(cur_ids)
        cur_ids += [pad_id] * pad
        cur_lab += [IGNORE] * pad
        cur_pos += list(range(len(cur_pos), len(cur_pos) + pad))
        cur_cu.append(max_len)
        packs.append((cur_ids, cur_lab, cur_pos, cur_cu))
    to_t = lambda x: torch.tensor(x, dtype=torch.long)
    return [{"input_ids": to_t(a), "labels": to_t(b),
             "position_ids": to_t(c), "cu_seqlens": to_t(d)} for a, b, c, d in packs]
```

## 面试追问

**Q：SFT 数据量到底要多少条？**
A：没有固定阈值，取决于任务覆盖度和基座能力。经验区间：格式/风格对齐 1000-10000 条（LIMA 1030 条即可）；复杂能力（数学、代码、多轮工具调用）需要数万到数十万条，因为要覆盖解题路径的多样性。判断标准不是条数，而是 eval set 上的能力是否还在随数据增长——画一条数据量 vs 指标的曲线，拐点之后加数据就是浪费。

**Q：为什么要做 epoch 2-3 而不是训 10 个 epoch？**
A：SFT 数据的多样性远低于预训练语料，多 epoch 会迅速从"学习"变成"背诵"。判据：第 2-3 个 epoch 时 train loss 还在降但 eval loss 开始抬头，就是过拟合起点。指令微调任务的 loss 曲线往往在第 1 个 epoch 内就完成主要下降，后续 epoch 主要在做模板记忆。

**Q：chat template 训练时和推理时不一致会怎样？**
A：这是最高频的线上事故之一。训练用了模板 A（比如 ChatML `<|im_start|>`），推理时用了模板 B（比如 Llama-2 的 `[INST]`），模型会看到完全陌生的特殊 token 序列，表现为：不理会 system prompt、不会停（`<|im_end|>` 没被学到）、或者直接复读输入。根治办法是用 tokenizer 的 `apply_chat_template()` 统一，并把模板固化进模型的 `tokenizer_config.json` 随权重一起发布。

**Q：padding 用哪个 token？为什么经常把 pad_token 设成 eos_token？**
A：很多开源基座（如 Llama 系列）tokenizer 里没有 pad token。常见做法是 `tokenizer.pad_token = tokenizer.eos_token`。这时**必须**确认 label 的 padding 位被设为 `-100`，否则模型会被训练去生成 eos，导致输出极短或直接空回复。更稳妥的做法是往词表里新增一个专用 `<|pad|>` 并 resize embedding（见继续预训练卡片）。

**Q：怎么区分过拟合和灾难性遗忘？**
A：两者现象不同。过拟合是**训练集分布内的指标也变差或泛化变差**：eval loss 上升、生成多样性下降、对 paraphrase 后的同义提问鲁棒性变差。灾难性遗忘是**基座原有能力丢失**：通用任务（MMLU、GSM8K、 HumanEval）分数下降，而 SFT 目标任务分数正常。前者靠减 epoch / 加 dropout / 减 lr 解决，后者要靠混通用数据 replay 或降 lr / 上 LoRA。

**Q：packing 之后 loss 的分母要不要按样本数归一？**
A：朴素 packing 下分母是 batch 内所有有效 token 数，与 padding 方案一致，可以直接沿用。但 packing 会改变"一条样本"的物理边界，如果某条超长样本被切到两个 pack 里，需要保证它的 token 不被重复计数或截断。工程上更简单的做法是不要切分单条样本，超长样本单独成 pack。

## 关联知识点

- LoRA 与 QLoRA 原理（微调与参数高效）
- Prompt Tuning 家族与 Adapter（微调与参数高效）
- 继续预训练与领域适配（微调与参数高效）
- RLHF 三阶段流程（对齐与强化学习）
- 解码策略与采样参数（推理优化与部署）
