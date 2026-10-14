---
id: transformer-architecture
title: Transformer 结构与参数量计算
module: 03-transformer-architecture
tags:
- Transformer
- 结构
- 参数量
- Decoder-only
- 高频
difficulty: 2
freq: 高
updated: 2026-08-28
summary: 现代 LLM 的标准形态是 Decoder-only + Pre-LN RMSNorm + SwiGLU FFN + RoPE 且不共享 embedding，参数量几乎全部集中在
  L×(4d²+3·d·d_ff) 上。
source:
- https://arxiv.org/abs/1706.03762
questions:
- 为什么现在主流大模型都是 Decoder-only，而不是 Encoder-Decoder？
- Pre-LN 和 Post-LN 到底差在哪？
- embedding 和 lm_head 要不要共享权重？
- 加深还是加宽？
- 参数量里 n_h·d_h 与 d 不等会怎样？
---

## 一句话结论

现代开源 LLM 的结构已经高度收敛，事实标准是：

```
Embedding → [RMSNorm → Attention → 残差加 → RMSNorm → FFN → 残差加] × L → RMSNorm → LM Head
```

即 **Decoder-only + Pre-LN + RMSNorm + SwiGLU + RoPE + 无 bias + embedding 与 lm_head 不共享**。

参数量的结论更直接：**7B 模型中 attention 占约 1/3、FFN 占约 2/3、embedding 只占约 4%**。所以"词表太大导致模型大"这个直觉是错的，真正吃参数的是隐藏层宽度和层数。

## 原理推导

### 1. 单个 Decoder Layer 的数据流

设输入 `x ∈ R^{B×S×d}`（d = d_model = hidden_size）：

```
h = x + Attention(RMSNorm(x))      # 第一残差
y = h + FFN(RMSNorm(h))            # 第二残差
```

注意归一化放在**子层内部**（Pre-LN），而不是子层输出之后（Post-LN）。这一点决定了深层模型能不能训得动，详见 `归一化与激活函数` 一篇。

### 2. 残差流视角（理解 Transformer 的关键心智模型）

把最底层的 embedding 看作一条贯穿整个模型的"信息高速公路"**residual stream**，宽度是 d。每一层的 Attention 和 FFN 都不直接改写这条公路，而是：

- 从公路上**读**一个归一化后的副本
- 计算出一个增量 Δ
- 把 Δ **加**回公路

所以 `x_L = x_0 + Σ Δ_i`，层数增加等价于在同一条公路上做更多次"小步编辑"，而不是把信息压缩再重建。这解释了两件事：

1. 为什么可以随意插入/删除层做消融（公路宽度不变，接口一致）
2. 为什么 resid 分支上的数值会随层数累积变大——这也是 Pre-LN 需要配合合适初始化的原因

### 3. 参数量逐项公式

记 d = d_model，L = 层数，n_h = query 头数，n_kv = KV 头数，d_h = head_dim，d_ff = FFN 中间维，V = 词表大小。

**Attention 部分**（MHA，即 n_kv = n_h 且 n_h·d_h = d 时每个矩阵都是 d×d）：

```
W_q, W_k, W_v, W_o  各 d²  →  4d²
```

若使用 GQA / MQA（n_kv < n_h）：

```
W_q: d × (n_h · d_h)
W_k: d × (n_kv · d_h)
W_v: d × (n_kv · d_h)
W_o: (n_h · d_h) × d
```

**FFN 部分**（SwiGLU 有三个矩阵：gate、up、down）：

```
W_gate: d × d_ff
W_up:   d × d_ff
W_down: d_ff × d
合计:   3 · d · d_ff
```

注意：SwiGLU 比原始 FFN（2·d·d_ff）多一个矩阵，所以 LLaMA 把 d_ff 从 4d 降到 (2/3)·4d = 8d/3，使总参数量与"4d 的两矩阵 FFN"大致持平。

**Embedding 与 LM Head**：

```
不共享: 2 · V · d
共享:   V · d
```

**总参数量**：

```
N = L · (4d² + 3·d·d_ff) + 2·V·d        （MHA + 不共享 embedding）
```

### 4. 训练显存与算力

混合精度训练（bf16 参数 + fp32 主权重 + AdamW）每参数占 16 字节：

```
模型 bf16 参数   2 bytes
梯度 bf16        2 bytes
fp32 主权重      4 bytes
Adam m           4 bytes
Adam v           4 bytes
─────────────────────────
合计            16 bytes / param
```

训练算力（前向 2N + 反向 4N，N 为非 embedding 参数量，D 为 token 数）：

```
训练总 FLOPs ≈ 6 · N · D
单次前向     ≈ 2 · N · D
```

## 具体例子

以 **LLaMA-7B** 为例核对（d=4096, L=32, n_h=32, n_kv=32, d_h=128, d_ff=11008, V=32000）：

| 项目 | 计算 | 参数量 |
|---|---|---|
| Attention / 层 | 4 × 4096² | 67,108,864 |
| FFN / 层 | 3 × 4096 × 11008 | 135,266,304 |
| 单层合计 | | 202,375,168 |
| 32 层 | | 6,476,005,376 |
| Embedding + Head | 2 × 32000 × 4096 | 262,144,000 |
| **总计** | | **6,738,149,376 ≈ 6.74B** |

与官方公布的 6.7B 吻合。占比：**FFN 64.0% / Attention 31.7% / Embedding 3.9%**。

再算 LLaMA-13B（d=5120, L=40, d_ff=13824, V=32000）：

```
单层 = 4×5120² + 3×5120×13824 = 104,857,600 + 212,336,640 = 317,194,240
40 层 = 12,687,769,600
Embedding = 2×32000×5120 = 327,680,000
总计 ≈ 13.02B
```

**显存速算**：7B 模型

```
bf16 推理权重   6.74e9 × 2   ≈ 13.5 GB
AdamW 训练      6.74e9 × 16  ≈ 108 GB   → 80G A100 单卡放不下，必须 ZeRO / TP
```

**算力速算**：7B 模型训练 1T tokens

```
6 × 6.74e9 × 1e12 = 4.04e22 FLOPs
A100 bf16 峰值 312 TFLOPS，MFU 取 40% → 125 TFLOPS 有效
单卡耗时 = 4.04e22 / 1.25e14 ≈ 3.23e8 s ≈ 10.2 卡年
1024 卡约 3.6 天
```

## 代码速记

```python
import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        # 只在最后一维求 RMS，不做减均值、不做 bias
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class DecoderLayer(nn.Module):
    """Pre-LN Decoder Block：Norm 在子层内部"""
    def __init__(self, d, n_h, n_kv, d_ff):
        super().__init__()
        self.n_h, self.n_kv, self.d_h = n_h, n_kv, d // n_h
        self.q = nn.Linear(d, n_h * self.d_h, bias=False)
        self.k = nn.Linear(d, n_kv * self.d_h, bias=False)
        self.v = nn.Linear(d, n_kv * self.d_h, bias=False)
        self.o = nn.Linear(n_h * self.d_h, d, bias=False)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x):
        B, S, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, S, self.n_h, self.d_h).transpose(1, 2)
        k = self.k(h).view(B, S, self.n_kv, self.d_h).transpose(1, 2)
        v = self.v(h).view(B, S, self.n_kv, self.d_h).transpose(1, 2)
        # GQA：KV 头数不足时按整数倍复制，等价于 repeat_kv
        rep = self.n_h // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, -1)
        x = x + self.o(out)                 # 残差 1
        x = x + self.ffn(self.n2(x))        # 残差 2
        return x

def count_params(d=4096, L=32, n_h=32, n_kv=32, d_ff=11008, V=32000):
    d_h = d // n_h
    attn = d * (n_h * d_h) + 2 * d * (n_kv * d_h) + (n_h * d_h) * d
    ffn = 3 * d * d_ff
    return L * (attn + ffn) + 2 * V * d

print(f"{count_params()/1e9:.3f} B")   # 6.738 B
```

## 面试追问

**Q：为什么现在主流大模型都是 Decoder-only，而不是 Encoder-Decoder？**
A：四个原因。(1) 训练目标（逐 token 预测）与生成任务完全一致，zero-shot 迁移能力最好；(2) 相同参数/算力预算下，Decoder-only 的 scaling 曲线更优，Encoder 那部分参数相当于"分摊"了预算；(3) 工程上 KV Cache 复用、投机解码、chunked prefill 都只对自回归结构自然成立；(4) 双向注意力在理解类任务上有优势，但这部分能力可以通过指令微调补回来，性价比不如统一架构。

**Q：Pre-LN 和 Post-LN 到底差在哪？**
A：Post-LN 的归一化在残差加完之后（`y = Norm(x + F(x))`），残差支路被归一化，导致初始化时**靠近输出层的梯度被放大、靠近输入层的梯度被缩小**，深层模型必须精细 warmup 才不崩。Pre-LN 的残差支路是从输入直通输出的恒等路径，梯度可以直接无损回传到第一层，训练对 warmup 和 lr 都鲁棒得多。代价是 Pre-LN 的激活值方差会随层数累积变大（因为没有归一化压制），所以需要配合 scaled 初始化。

**Q：embedding 和 lm_head 要不要共享权重？**
A：共享能省下 V·d 参数（7B 模型约 1.3 亿，占 2%），且有一定的正则化效果（GPT-2、部分 Qwen 版本这么做）。不共享的理由是：embedding 和 lm_head 承担完全不同的功能——一个是查表输入，一个是产生 logits 分布，共享会让 embedding 的梯度同时被两端的任务拉扯，在大词表大 batch 下训练更容易不稳。LLaMA 系列全部不共享。

**Q：加深还是加宽？**
A：经验结论是在固定参数预算下，深度对"推理/多步能力"更有帮助，宽度对"知识容量和并行效率"更有帮助。实践中大家基本不做这个权衡，而是沿用已验证的深度/宽度比（LLaMA 系大致 d≈4096 配 32 层、d≈5120 配 40 层），真正的自由度留给 d_ff 和数据。

**Q：参数量里 n_h·d_h 与 d 不等会怎样？**
A：只要 W_o 的输出维是 d、W_q 的输入维是 d 就能跑通，两者不需要相等。现代实现（如 Qwen）显式配置 `head_dim` 而不从 d 整除，好处是 head_dim 可以稳定在 128（对 FlashAttention kernel 友好），同时 d 可以自由取值。

## 关联知识点
- Scaled Dot-Product Attention 与为什么要除以 √d（01-transformer）
- 归一化、残差与激活函数：RMSNorm / Pre-LN / SwiGLU（01-transformer）
- RoPE 旋转位置编码（01-transformer）
- 训练显存与算力估算模板（10-coding）
