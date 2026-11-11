---
id: continue-pretraining
title: 继续预训练与领域适配：数据混合、学习率与防遗忘
module: 08-sft-peft
tags:
- 继续预训练
- 领域适配
- 灾难性遗忘
- 词表扩展
- replay
difficulty: 3
freq: 中
updated: 2026-08-28
summary: 领域适配（如医疗/法律/代码）常在基座上做"继续预训练（CPT）"。核心是数据混合比例（通用数据 replay 防遗忘）、低学习率与 warmup、以及灾难性遗忘的度量与缓解。还需要考虑词表扩展与
  embedding 初始化。
source:
- https://arxiv.org/abs/2106.09685
questions:
- 为什么 CPT 一定要混通用数据 replay？不复盘会怎样？
- replay 比例怎么定？有没有理论依据？
- 词表扩展后，新增 token 的 embedding 怎么初始化才训得动？
- 怎么度量"灾难性遗忘"？除了 replay 还有什么缓解手段？
- CPT 的学习率为什么要比预训练低？
---

## 一句话结论

- **不要纯领域数据**：必须混入一定比例（常 5%~30%）的通用预训练数据做 **replay**，否则通用能力会"被洗掉"（灾难性遗忘）。
- **学习率要低且 warmup**：CPT 阶段 lr 通常只有预训练的 1/10~1/3（如 1e-5~5e-5），配合短 warmup 与 cosine 衰减。
- **遗忘可度量**：在通用 benchmark（MMLU 等）上的掉点即遗忘信号；缓解用 replay / 参数冻结 / 低 lr / L2 正则。

## 原理推导

### 为什么纯领域数据会遗忘

预训练把"通用知识"编码进权重。CPT 若只看领域分布 `D_domain`，优化会沿 `∇L(D_domain)` 把权重推离"通用最优"，领域任务 loss 降了，但 `L(D_general)` 上升 → 通用能力退化。replay 把目标改成联合 `L = α·L(domain) + (1-α)·L(general)`，约束权重不偏离太远。

### 数据混合比例的实验结论

- 太多领域数据 → 遗忘严重；太少 → 领域增益不足。
- 经验区间：领域:通用 ≈ **7:3 到 9:1**，代码/数学等"硬领域"可更高；也常用"课程式"——先高比例通用、逐步提领域。
- 小模型 ablation 外推法（同 scaling law 思路）：在小模型上扫比例，选最优后迁移到大模型。

### 词表扩展

领域有大量 OOV（如中文医疗古文、代码标识符）时扩充词表：

- 新增 token 的 embedding 用**已有 embedding 的均值或同源 token 拼接初始化**，避免随机初始化训不动。
- 扩展后**词表对应的 embedding 与 lm_head 矩阵变大**，新增行需训练；旧行保持。注意：扩展会轻微改变 tokenization，需重新对齐旧权重对应行。

## 具体例子

**医疗 LLM 的 CPT 配方（示意）**：

- 数据：医学文献/指南 50B token + 通用语料 replay 15B token（约 3:1）。
- 优化：lr 3e-5，warmup 200 step，cosine 到 3e-6，batch 大、epoch 1~2。
- 监控：每几千 step 在 MMLU + 医学 QA 上测，若 MMLU 掉 >2 点则提高 replay 比例。
- 词表：从 32k 扩到 35k（+3k 医疗/古文 token），新 embedding 用相邻 token 平均初始化。

## 代码速记

```python
# 只训练扩展出的 token 对应的 embedding/lm_head 行（其余冻结可选）
new_ids = list(range(old_vocab, new_vocab))
for p in model.parameters():
    p.requires_grad = False
model.model.embed_tokens.weight.requires_grad = True
model.lm_head.weight.requires_grad = True
# 或只对 new_ids 行 unlock：用parameter hook 更精细
```

## 面试追问

1. 为什么 CPT 一定要混通用数据 replay？不复盘会怎样？
2. replay 比例怎么定？有没有理论依据？
3. 词表扩展后，新增 token 的 embedding 怎么初始化才训得动？
4. 怎么度量"灾难性遗忘"？除了 replay 还有什么缓解手段？
5. CPT 的学习率为什么要比预训练低？

## 关联知识点

- [SFT 全流程配方：数据构造、loss 掩码、packing 与超参经验值](sft-recipe)
- [LoRA 与 QLoRA：低秩分解与 4bit 训练](lora-and-qlora)
- [Tokenizer：BPE / WordPiece / Unigram 与词表设计](tokenizer)
