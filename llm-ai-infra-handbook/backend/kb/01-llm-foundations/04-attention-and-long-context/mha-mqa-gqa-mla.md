---
id: mha-mqa-gqa-mla
title: MHA / MQA / GQA / MLA 对比与 KV Cache 压缩
module: 04-attention-long-context
tags:
- 注意力变体
- GQA
- MQA
- MLA
- KV Cache
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: GQA 通过让多个 query 头共享一组 KV 头把 KV Cache 压缩到 1/g 倍，质量损失极小；MLA 用低秩联合压缩把 KV 压到一个潜向量，且推理时可通过矩阵吸收避免解压。
source:
- https://arxiv.org/abs/2205.14135
questions:
- GQA 是怎么训的？直接在预训练里用，还是从 MHA 转过来？
- MQA 质量损失有多大？为什么 PaLM 用 MQA 而 LLaMA 用 MHA？
- MLA 的矩阵吸收在训练和推理时都能用吗？
- GQA 下 KV Cache 量化有什么特殊注意点？
- n_h 不能被 n_kv 整除怎么办？
---

## 一句话结论

这条演进线的唯一目标是**压 KV Cache**，因为长上下文推理时 KV Cache 才是显存和带宽的瓶颈，而它随序列长度线性增长、随 batch 线性增长，比固定不变的权重可怕得多。

| 方案 | KV 头数 | KV Cache 相对 MHA | 质量损失 | 代表模型 |
|---|---|---|---|---|
| MHA | n_h | 1× | 无（基准） | LLaMA-1/2、GPT 系列早期 |
| GQA | g（1 < g < n_h） | 1/(n_h/g) | 极小 | LLaMA-3、Qwen2/3、Mistral |
| MQA | 1 | 1/n_h | 较小但可测 | PaLM、Falcon、部分早期推理友好模型 |
| MLA | 潜向量（不存 KV） | ≈ 1/93（DeepSeek） | 极小 | DeepSeek-V2/V3/R1 |

**选型结论**：新模型默认 GQA（组数 g 通常取 8，即 32 头配 8 组）；追求极致长上下文吞吐用 MLA；MQA 只在极度带宽受限且能接受质量损失时用。

## 原理推导

### 1. 为什么是 KV 而不是 Q

自回归解码每步只产生**一个**新 token 的 query，但需要**全部**历史的 K 和 V。所以：

- Q 只需要当前步的，缓存毫无意义
- K、V 必须缓存，否则每步重算，复杂度从 O(S) 退化到 O(S²)

因此所有优化都指向 K/V 的头数或维度。

### 2. GQA 的数学形式

MHA 中每个头有独立的 W_k^i、W_v^i（i = 1..n_h）。GQA 把 n_h 个 query 头分成 g 组，每组 g_h = n_h/g 个 query 头共享一对 K/V 头：

```
组 j 的 key:   K^j = X · W_k^j,  j = 1..g
第 i 个 query 头属于组 j = i // g_h
Attention_i = softmax(Q_i K^jᵀ / √d_h) V^j
```

实现上等价于把 K/V 头**重复** g_h 次再走标准 MHA（`repeat_interleave`）或者用更高效的 reshape + broadcast（避免实际复制）。

注意 `repeat_interleave` 与 `repeat` 的区别：

```
repeat_interleave: [k1, k1, k2, k2]     # 头内连续，正确
repeat:            [k1, k2, k1, k2]     # 头间交错，错误
```

### 3. KV Cache 显存公式

通用公式（每层每 token）：

```
每 token 每层的 KV 字节数 = 2 × n_kv × d_h × bytes_per_elem
```

一个长度 S、batch B、层数 L 的完整缓存：

```
KV Cache = 2 × B × S × L × n_kv × d_h × bytes
```

MHA 时 `n_kv = n_h`；GQA 时 `n_kv = g`；MLA 时 `n_kv × d_h` 被替换成潜向量维度 `d_c`。

### 4. MLA 的低秩联合压缩

DeepSeek 的 MLA 思路：不缓存 K 和 V，而是缓存一个**压缩后的潜向量** c：

```
下投影（训练/推理都要算）:  c_{KV} = X · W^{DKV},   W^{DKV} ∈ R^{d × d_c},  d_c ≪ d_h·n_h
上投影（理论上需要）:       K = c_{KV} · W^{UK},  V = c_{KV} · W^{UV}
```

关键 trick：**推理时不需要显式上投影**。因为 attention 分数可以改写：

```
score = Q · (c·W^{UK})ᵀ = (Q·W^{UK}ᵀ) · cᵀ
```

把 `W^{UK}ᵀ` **吸收进 W_q**（`W_q' = W_q · W^{UK}ᵀ`），就可以直接用压缩向量 c 计算注意力分数，完全跳过解压 K 的步骤。同理：

```
out = attn · (c·W^{UV}) = (attn · c) · W^{UV}
```

把 `W^{UV}` 吸收进 `W_o`（`W_o' = W^{UV} · W_o`）。

**结论**：MLA 推理时缓存的是 d_c 维的潜向量，且矩阵乘可以直接吃这个潜向量，不需要还原 K/V。这既省显存又省带宽——这是 MLA 比"先压缩再解压"的方案高级的地方。

MLA 还额外对 Q 做低秩压缩（d_c' ≈ 1536），目的不是省缓存（Q 不缓存）而是**省训练时的激活显存**。

另一个细节：RoPE 与低秩压缩不兼容（旋转会破坏线性可交换性），所以 MLA 单独切出 `d_h^R = 64` 维走"解耦 RoPE"（decoupled RoPE），这部分不压缩、单独缓存。所以 MLA 的缓存实际是 `d_c + d_h^R`。

## 具体例子

**例 1：LLaMA-2-7B（MHA）的 KV Cache**

d=4096, L=32, n_h=n_kv=32, d_h=128，bf16（2 字节），B=1, S=4096：

```
KV = 2 × 1 × 4096 × 32 × 32 × 128 × 2
   = 2 × 4096 × 32 × 4096 × 2
   = 2,147,483,648 bytes = 2.0 GB
```

对比权重 13.5GB：4k 长度时 KV 只占 15%。但 S=32768 时 KV = 16GB，**超过权重**。B=32 时更是达到 512GB——这就是为什么高并发长上下文服务必须用 GQA/MLA + 量化 + 卸载。

**例 2：换成 GQA（g=8）**

```
KV = 2 × 1 × 4096 × 32 × 8 × 128 × 2 = 536,870,912 = 0.5 GB
压缩比 = 32/8 = 4×
```

**例 3：DeepSeek-V3 的 MLA 压缩比**

配置：n_h = 128, d_h = 128, d_c (KV 压缩维) = 512, d_h^R (解耦 RoPE) = 64, L = 61。

```
MLA 每 token 每层缓存:  d_c + d_h^R = 512 + 64 = 576  个数值
等效 MHA 每 token 每层: 2 × n_h × d_h = 2 × 128 × 128 = 32,768  个数值

压缩比 = 32768 / 576 ≈ 56.9×
```

若按"每层每 token 数值个数"算，官方口径常给出约 93×（因为还额外计入了 Q 的压缩和不同基数），量级在 **50~100×**。这就是为什么 DeepSeek 能以极低推理成本支撑 128k 上下文。

**例 4：并发能力对比（A100 80G，7B 模型）**

```
可用显存 ≈ 80 - 13.5(权重) - 2(激活/碎片) ≈ 64 GB

MHA, S=4096:  每请求 KV = 2.0 GB   → 最多 32 并发
GQA(g=8):     每请求 KV = 0.5 GB   → 最多 128 并发
GQA + FP8 KV: 每请求 KV = 0.25 GB  → 最多 256 并发
```

## 代码速记

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GQAAttention(nn.Module):
    def __init__(self, d, n_h, n_kv, d_h=None):
        super().__init__()
        self.n_h, self.n_kv, self.d_h = n_h, n_kv, d_h or d // n_h
        assert n_h % n_kv == 0, "query 头数必须是 KV 头数的整数倍"
        self.group = n_h // n_kv
        self.q = nn.Linear(d, n_h * self.d_h, bias=False)
        self.k = nn.Linear(d, n_kv * self.d_h, bias=False)
        self.v = nn.Linear(d, n_kv * self.d_h, bias=False)
        self.o = nn.Linear(n_h * self.d_h, d, bias=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.n_h, self.d_h).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_kv, self.d_h).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_kv, self.d_h).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        # 高效 GQA：不复制 KV，用 expand + reshape 让 matmul 自己 broadcast
        # (B, n_kv, S, d) -> (B, n_kv, 1, S, d) -> (B, n_kv, group, S, d) -> (B, n_h, S, d)
        k_exp = k.unsqueeze(2).expand(-1, -1, self.group, -1, -1).reshape(B, self.n_h, -1, self.d_h)
        v_exp = v.unsqueeze(2).expand(-1, -1, self.group, -1, -1).reshape(B, self.n_h, -1, self.d_h)
        # 等价但更省显存的写法: k.repeat_interleave(self.group, dim=1)
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=(kv_cache is None))
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o(out), new_cache

def kv_cache_bytes(B, S, L, n_kv, d_h, dtype_bytes=2):
    return 2 * B * S * L * n_kv * d_h * dtype_bytes

# LLaMA-2-7B, B=1, S=4096
print(kv_cache_bytes(1, 4096, 32, 32, 128) / 1024**3, "GB")   # 2.0
print(kv_cache_bytes(1, 4096, 32, 8, 128) / 1024**3, "GB")    # 0.5
```

## 面试追问

**Q：GQA 是怎么训的？直接在预训练里用，还是从 MHA 转过来？**
A：两种方式都可行，且都有成熟 recipe。直接预训练用 GQA 最简单（LLaMA-3、Qwen）；从已有的 MHA checkpoint 转 GQA 叫 **GQA uptraining**，做法是把同组内的多个 K/V 头取**平均**（或取第一个）合并成一个头，然后继续用原训练数据的 5%~10% 做短程微调，质量损失通常在 1% 以内。取平均比取第一个好，因为平均保留了组内所有头的信息。

**Q：MQA 质量损失有多大？为什么 PaLM 用 MQA 而 LLaMA 用 MHA？**
A：MQA（所有 query 头共享一对 KV）在长文本和需要精细位置信息的任务上损失可测，通常 PPL 劣化几个百分点。PaLM 选 MQA 是因为它服务于大规模在线推理，TPU 上的带宽成本是首要约束；LLaMA-1/2 定位是开源研究基座，更看重质量，所以保留 MHA。到了 LLaMA-3（8B 以上全部 GQA）说明业界已确认 GQA 是更优折中。

**Q：MLA 的矩阵吸收在训练和推理时都能用吗？**
A：推理时能，训练时不能。原因是**吸收会破坏位置编码和结构**：训练时需要对 K 施加 RoPE、需要显式的多头结构做 loss 和梯度；而吸收后 `W_q' = W_q W^{UK}ᵀ` 把两个矩阵合成了一个，形状虽然合法，但 RoPE 要在压缩前施加，且训练时还需要额外的表征约束。所以 MLA 训练走完整路径、推理走吸收路径，部署时需要做一次权重融合（absorb）。

**Q：GQA 下 KV Cache 量化有什么特殊注意点？**
A：GQA 让 KV 头数变少，每个头的"重要性"上升，量化误差的影响被放大。实践上：K 比 V 对量化更敏感（K 参与分数计算，误差直接进 softmax），所以常见做法是 K 用 per-channel、V 用 per-token，或者对 K 保留更高精度。另外必须先做异常值处理（如 KV cache 的 SmoothQuant / QuaRot 的 Hadamard 旋转），直接 INT8 量化 KV 在长上下文下 PPL 会明显崩。

**Q：n_h 不能被 n_kv 整除怎么办？**
A：标准 GQA 要求整除，否则无法均匀分组。若非要非整除，实现上可以退化为"复制后截断"或干脆用 MQA 式的单一 KV 头 + 注意力头数不变。实践中所有模型都保证整除（常见组合 32/8、64/8、128/8）。

**Q：GQA 对训练速度有影响吗？**
A：几乎没有。训练时序列是整块输入的，KV 不缓存，GQA 只是让 K/V 的投影矩阵输出维度从 n_h·d_h 降到 g·d_h，FLOPs 略微下降（attention 部分的投影少了一点），但 attention 分数矩阵仍是 n_h 个 S×S。所以训练收益很小，GQA 的价值几乎全在推理侧。

## 关联知识点
- KV Cache 原理、显存估算与 prefill/decode 瓶颈（02-attention）
- PagedAttention 与 vLLM 显存管理（02-attention）
- FlashAttention 与 IO 复杂度分析（02-attention）
- 量化：KV Cache 量化与 SmoothQuant（07-inference）
- 手写 Multi-Head Attention 与 KV Cache（10-coding）
