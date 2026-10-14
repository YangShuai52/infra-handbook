---
id: quantization
title: 量化：对称/非对称、GPTQ、AWQ 与 SmoothQuant
module: 15-inference-algorithms
tags:
- 量化
- GPTQ
- AWQ
- SmoothQuant
- INT8
- FP8
- 对称量化
- per-group
difficulty: 4
freq: 高
updated: 2026-08-28
summary: 量化把 fp16/bf16 权重/激活压成 INT8/INT4，省显存、提吞吐。核心权衡是"粒度 vs 误差"：per-group 比 per-tensor
  更准。GPTQ 用二阶信息逐列量化、AWQ 保护显著权重、SmoothQuant 把激活的量化难度迁移到权重。
source:
- https://arxiv.org/abs/2211.17192
questions:
- 对称和非对称量化的公式？什么时候用非对称？
- per-group 为什么比 per-tensor 准？代价是什么？
- GPTQ 和 AWQ 的核心思想差异？（二阶补偿 vs 保护显著权重）
- SmoothQuant 为什么能把激活难度迁移到权重？
- W4A16 是什么意思？为什么激活常保持 fp16？
---

## 一句话结论

- **对称量化**：`q = round(x / s)`，`s = max|x|/127`（INT8），零点为 0；非对称加零点 `z` 适配非零中心分布（激活常用）。
- **粒度**：per-tensor < per-channel < per-group(per-block)，越细越准、开销越大。
- **GPTQ**：基于 OBQ 的逐列二阶量化，用 Hessian 补偿截断误差。
- **AWQ**：发现"少量显著权重"主导精度，只缩放激活（等效保护这些权重），不反量化它们。
- **SmoothQuant**：把激活的离群难度乘到权重上（`s = max(x)/max(w)`），让两者都变好量化。

## 原理推导

### 对称 / 非对称量化

```
对称:   q = clip(round(x / s), -127, 127),   x̂ = q·s,  s = max|x|/127
非对称: q = clip(round((x - z)/s), 0, 255),  x̂ = q·s + z
        s = (x_max - x_min)/255,  z = x_min
```

- 权重近似零中心 → 对称足够；激活有偏移 → 非对称更好。
- 反量化误差 `|x - x̂|` 由缩放步长 `s` 决定，s 越小越准但动态范围越小。

### 粒度（granularity）

- **per-tensor**：整个张量一个 `s` → 一个离群值把 s 拉大、其余全量化粗糙。
- **per-channel / per-group**：每个输出通道或每 128 个元素一组一个 `s` → 各组自适配，误差大幅下降。INT4 普遍用 **per-group=128**。

### GPTQ（逐列二阶）

把 OBQ（Optimal Brain Quantization）扩展到 GPU 可并行：对每一输出列，量化当前权重后，用该列 Hessian 逆把误差**分配到后续未量化的列**上补偿。本质一阶/二阶信息最小化量化后输出的 `||Wx - Ŵx||²`。4-bit 下能保持接近 fp16 的精度。

### AWQ（Activation-aware Weight Quantization）

观察到：权重里只有 ~1% 的"显著权重"（对激活贡献大）决定精度。AWQ **不量化这些权重**，而是对激活做 per-channel 缩放 `x̃_j = x_j · s_j`，等效于把缩放吸收进权重 `ŵ = w·s`，使显著权重相对"变大"、被保留更多信息。比 GPTQ 更简单、不依赖校准的二阶、硬件友好。

### SmoothQuant（训练后，针对激活离群）

权重分布好量化、激活有极端离群值（少数 channel 极大）。令 `y = (W·(X·s))·(1/s) = Ŵ·X̃`，选 `s_j = max|X_j|^α / max|W_j|^(1-α)`（α≈0.5），把离群难度从激活迁移到权重，双方都进入 INT8 友好范围 → W8A8 几乎无损。

## 具体例子

- **LLM.int8()**：对离群特征（按 channel 绝对值最大的 0.1%）保持 fp16、其余 INT8，解决激活离群导致的崩塌。
- **4-bit 部署**：AWQ/GPTQ 把 7B 从 14GB(fp16) 压到 ~4GB(INT4)，单卡消费级显卡可跑。
- **W4A16**：权重 4-bit（AWQ），激活 fp16，推理时反量化计算——最常见部署形态。
- **FP8**：H100 原生 FP8 张量核，训练/推理都可近无损加速，无需复杂缩放（但有累加精度陷阱需小心）。

## 代码速记

```python
# 对称 per-group INT4 量化（示意）
def quantize_sym(w, group=128, bits=4):
    w = w.reshape(-1, group)
    s = w.abs().amax(1, keepdim=True) / (2**(bits-1)-1)
    q = (w / s).round().clamp(-(2**(bits-1)), 2**(bits-1)-1)
    return q.to(torch.int8), s                       # 反量化: q*s
```

## 面试追问

1. 对称和非对称量化的公式？什么时候用非对称？
2. per-group 为什么比 per-tensor 准？代价是什么？
3. GPTQ 和 AWQ 的核心思想差异？（二阶补偿 vs 保护显著权重）
4. SmoothQuant 为什么能把激活难度迁移到权重？
5. W4A16 是什么意思？为什么激活常保持 fp16？

## 关联知识点

- [解码策略：temperature / top-p / repetition](decoding-strategy)
- [投机解码：draft-then-verify 与加速比推导](speculative-decoding)
- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
