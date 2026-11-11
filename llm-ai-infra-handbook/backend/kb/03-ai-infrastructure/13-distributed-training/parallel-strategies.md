---
id: parallel-strategies
title: 并行策略：DP / TP / PP 与三维并行选型
module: 13-distributed-training
tags:
- 分布式训练
- 数据并行
- 张量并行
- 流水线并行
- Megatron
- 高频
difficulty: 4
freq: 高
updated: 2026-08-28
summary: TP 切层内矩阵（通信量大，只能节点内）、PP 切层（通信量小但有 bubble）、DP 切数据（通信梯度）；三维并行的选型顺序是先看单卡能放什么，再定
  TP/PP。
source:
- https://arxiv.org/abs/2104.04473
questions:
- 为什么 TP 每个 Transformer 层只需 2 次 all-reduce？
- TP 和 ZeRO-3 都切参数，有什么区别？
- PP 的 bubble 能完全消除吗？
- 微批大小怎么选？
- DP/TP/PP 能不能和 ZeRO 混用？
---

## 一句话结论

三种并行切的东西不同，通信代价也差一个数量级：

| 策略 | 切分对象 | 每卡显存 | 通信内容 | 通信量（每 step） | 适用 |
|---|---|---|---|---|---|
| DP (DDP) | batch | 完整副本 | 梯度 all-reduce | `2(N-1)/N × 参数量` | 模型放得下时首选 |
| TP | 层内矩阵 | 1/tp | 激活 all-reduce（每层 2 次） | `O(batch×seq×d×layers)` | **只能节点内**（NVLink） |
| PP | 层 | 1/pp | 激活（只有边界） | `O(batch×seq×d×stages)` | 跨节点友好 |
| SP/CP | 序列 | 激活 1/sp | 序列维的 ring/p2p | 视实现 | 超长上下文 |
| EP | 专家 | 专家 1/ep | token all-to-all | `O(k×d×tokens)` | MoE 专用 |

**核心选型口诀**：

```
能用 DP 就用 DP（通信最少、实现最简单）
放不下就上 TP（但 TP 只做在节点内，跨节点会被 IB 带宽卡死）
还放不下或模型超大才上 PP（PP 通信最少但有 bubble，要配合微批）
```

## 原理推导

### 1. 数据并行（DP / DDP）

每张卡持有完整模型副本，各自处理不同的数据分片，反向后同步梯度。

**朴素 DP**（Parameter Server）：所有卡把梯度发给 PS，PS 更新后广播回来。问题：PS 是带宽瓶颈。

**DDP**（Ring All-Reduce）：去中心化，通信量恒定：

```
每张卡发送  2(N-1)/N × Φ  bytes
每张卡接收  2(N-1)/N × Φ  bytes    （Φ = 参数量 × 梯度字节数）
```

关键性质：**单卡通信量几乎与卡数无关**（N→∞ 时趋于 2Φ）。这是 DDP 能扩展到千卡的原因。

**梯度同步的重叠**：DDP 的做法是把梯度按 bucket（默认 25MB）分组，反向传播过程中算完一个 bucket 就异步 all-reduce 一个，与后续层的反向计算重叠。这是 DDP 实用的关键。

**显存代价**：每张卡都要存完整的模型 + 优化器状态（16 bytes/param）。这就是 ZeRO 要解决的问题（见 `ZeRO 与 FSDP`）。

### 2. 张量并行（TP，Megatron-LM）

把层内的矩阵乘按维度切分到多张卡。

**FFN 的切分**（SwiGLU 为例，三个矩阵）：

```
列切分（Column Parallel）:
  W_gate = [W_gate_1, W_gate_2]     沿输出维切
  W_up   = [W_up_1,   W_up_2]
  每卡算出部分结果，无需通信

行切分（Row Parallel）:
  W_down = [W_down_1]
           [W_down_2]              沿输入维切
  每卡算部分和，最后需要 all-reduce 合并
```

**为什么这样切**：因为 FFN 是 `Down(SiLU(Gate(x)) · Up(x))`，中间激活的维度是 d_ff。若 Gate/Up 按列切（输出维），每卡得到激活的一半，不需要通信；然后 Down 按行切（输入维），每卡算出一个部分和，最后 all-reduce 一次即可。

**Attention 的切分**：

```
W_q, W_k, W_v: 列切分（按 head 切，每卡 n_h/tp 个头）→ 无通信
W_o:           行切分（输入是 n_h 份拼接）→ all-reduce 一次
```

**关键结论：每个 Transformer 层只需 2 次 all-reduce**（attention 的 W_o 一次、FFN 的 W_down 一次）。这个数字面试常考。

**通信量**（每层，TP=t）：

```
每次 all-reduce: 2(t-1)/t × (B × S × d) × bytes    （激活张量大小）
每层 2 次
```

以 B=8, S=4096, d=4096, bf16 为例：

```
单次: 2 × (8 × 4096 × 4096) × 2 = 5.37e8 bytes = 537 MB
每层 2 次 = 1.07 GB
32 层（若每层都通信）= 34 GB per forward
```

这个量非常大——**1 TB/s 的 NVLink 需要 34ms**，而跨节点 IB（200 Gb/s = 25 GB/s）需要 1.4 秒，完全不可接受。所以：

> **TP 几乎必须限制在单节点内（NVLink 域），通常 tp ∈ {1, 2, 4, 8}。**

**Sequence Parallel（SP）**：TP 的补充。TP 只切分了 attention 和 FFN 的矩阵，但 LayerNorm/Dropout/残差加的激活仍是完整的。SP 把这些沿序列维切分，配合 TP 的 all-reduce 换成 reduce-scatter + all-gather，显存再省 `1/tp`。

### 3. 流水线并行（PP）

把 L 层切成 p 个 stage，每张卡（或每组卡）持有一个 stage。

**朴素 PP 的 bubble 问题**：

```
时间轴:  |--Stage1 F---|--Stage2 F---|--Stage3 F---|  (空闲: Stage1 等待)
        |              |--Stage1 B---|--Stage2 B---|--Stage3 B---|
        GPU1: [F][idle][idle][B]
        GPU2: [idle][F][idle][B]
        GPU3: [idle][idle][F][B]
```

bubble（空闲）比例：

```
bubble_ratio = (p - 1) / m        p = stage 数, m = 微批数
```

例如 p=4, m=8：bubble = 3/8 = 37.5%。**所以要减小 bubble 必须增大微批数 m**。

**1F1B 调度**（Megatron 的核心）：

把执行分成三个阶段：
1. **Warmup**：依次启动前 p 个微批的前向
2. **Steady（1F1B）**：每个时刻，每张卡做 1 个前向 + 1 个反向，交替进行
3. **Cooldown**：完成剩余微批的反向

1F1B 把 bubble 比例降到：

```
bubble_ratio = (p - 1) / m      （同样是这个公式，但峰值显存大幅降低）
```

关键收益不是减少 bubble，而是**降低显存**：朴素 PP 需要缓存 m 个微批的激活，1F1B 只需缓存 p 个（与 m 无关）。所以 1F1B 让"用很大的 m 来压低 bubble"变得可行。

**PP 的通信量**：只在 stage 边界传递激活：

```
每微批每个边界: B_micro × S × d × bytes
p-1 个边界
```

比 TP 小一到两个数量级，所以 **PP 可以跨节点**。

**Interleaved 1F1B**：每个 stage 负责 2 个不连续的子块（如 stage 0 负责第 0~1 层和第 8~9 层），进一步压低 bubble（约减半），代价是通信次数翻倍。

### 4. 三维并行与选型

总并行度 = `dp × tp × pp = 总 GPU 数`。

**选型决策树**（实用版）：

```
1. 模型 + 优化器显存 > 单卡显存？
   ├─ 否 → 纯 DP（或 ZeRO-1/2 省优化器）
   └─ 是 → 进 2

2. 单节点（8 卡）内用 tp 能否让模型放得下？
   ├─ 能 → 用 tp（如 tp=8），跨节点用 dp/pp
   └─ 不能 → 进 3

3. 加 pp 分层，或用 ZeRO-3 切参数
   （ZeRO-3 与 tp 是竞争关系：都切参数，但 ZeRO-3 通信量大、tp 通信量小但只能节点内）

4. 微批数 m 够大吗（m ≥ 4×pp）？不够就降 pp 或加 ZeRO
```

**现代实践的一个共识**：

- **中小模型（≤13B）**：ZeRO-2/3 + DP 是最省事的选择，tp 可不用
- **大模型（70B+）**：节点内 tp=8，节点间 pp 或 ZeRO-3 切分
- **超长上下文**：序列并行 / Context Parallel 必开

## 具体例子

**例 1：通信量对比（7B 模型，B=8, S=4096, d=4096, bf16）**

```
DP (N=64 卡):
  梯度 all-reduce: 2 × (63/64) × 6.74e9 × 2 bytes = 26.6 GB per step
  与 batch 无关，只与参数量有关

TP (t=8, 每层 2 次 all-reduce):
  单次: 2 × (7/8) × (8 × 4096 × 4096) × 2 = 4.70e8 bytes
  每层: 9.39e8 bytes
  32 层: 30.1 GB per forward
  加反向（同样量级）: 约 60 GB per step

PP (p=4, 3 个边界):
  每微批每边界: (8/4) × 4096 × 4096 × 2 = 6.7e7 bytes = 67 MB
  3 边界 × 8 微批 × 2(前后向): 3.2 GB per step
```

**结论**：PP 的通信量比 TP 小约 20 倍，比 DP 小约 8 倍。但 PP 有 bubble，所以不是"通信少就一定好"。

**例 2：TP 跨节点为什么不行**

TP 每层要传 0.94 GB（见上）。假设单层前向+反向计算时间约 20ms：

```
节点内 NVLink (300 GB/s):  0.94 GB / 300 = 3.1 ms     → 通信占 15%，可接受
跨节点 IB (200 Gb/s = 25 GB/s): 0.94 / 25 = 37.6 ms   → 通信占 188%，完全卡死
```

**例 3：PP 的 bubble 与微批数**

```
p=8 stages:
  m=8:   bubble = 7/8 = 87.5%     ← 灾难
  m=32:  bubble = 7/32 = 21.9%
  m=64:  bubble = 7/64 = 10.9%
  m=128: bubble = 7/128 = 5.5%
```

所以 PP 一定要配足够大的 global batch（micro_batch × m × dp）。

**例 4：70B 模型的 3D 并行配置（256 卡 A100 80G）**

```
70B, AdamW: 70e9 × 16 bytes = 1.12 TB 优化器+参数
单卡 80GB → 至少需要 14 卡才能放下（不考虑激活）

配置: tp=8 (节点内), pp=8, dp=4  → 8×8×4 = 256 卡
  每卡参数+优化器: 1.12 TB / (8×8) = 17.5 GB
  剩余 62 GB 给激活（配合重计算）
  global batch = micro_bs × m × dp = 4 × 32 × 4 = 512 sequences
  bubble = 7/32 = 21.9%（用 interleaved 可降到约 11%）
```

## 代码速记

```python
# ---------- 1. 通信量计算 ----------
def ddp_bytes(num_params, n_gpus, grad_bytes=2):
    """Ring all-reduce: 每卡发送/接收 2(N-1)/N × Φ"""
    return 2 * (n_gpus - 1) / n_gpus * num_params * grad_bytes

def tp_bytes_per_layer(B, S, d, tp, act_bytes=2):
    """每层 2 次 all-reduce（W_o 和 W_down）"""
    single = 2 * (tp - 1) / tp * B * S * d * act_bytes
    return 2 * single

def pp_bytes(micro_bs, S, d, pp, act_bytes=2):
    """(pp-1) 个边界，每个传一次激活"""
    return (pp - 1) * micro_bs * S * d * act_bytes

def bubble_ratio(pp, micro_batches):
    return (pp - 1) / micro_batches

print(f"DDP 64卡: {ddp_bytes(6.74e9, 64)/1e9:.1f} GB/step")
print(f"TP=8 每层: {tp_bytes_per_layer(8, 4096, 4096, 8)/1e9:.3f} GB")
print(f"PP=4: {pp_bytes(2, 4096, 4096, 4)/1e6:.1f} MB/微批")

# ---------- 2. 列切分 / 行切分 ----------
import torch, torch.nn as nn
class ColumnParallelLinear(nn.Module):
    """按输出维切: W = [W1, W2]，每卡算 W_i x，无需通信"""
    def __init__(self, d_in, d_out, world_size, rank):
        super().__init__()
        assert d_out % world_size == 0
        self.local = nn.Linear(d_in, d_out // world_size, bias=False)
    def forward(self, x):
        return self.local(x)

class RowParallelLinear(nn.Module):
    """按输入维切: W = [W1; W2]，每卡算部分和，最后 all-reduce"""
    def __init__(self, d_in, d_out, world_size):
        super().__init__()
        assert d_in % world_size == 0
        self.local = nn.Linear(d_in // world_size, d_out, bias=False)
    def forward(self, x_local):
        out = self.local(x_local)
        import torch.distributed as dist
        dist.all_reduce(out)                    # 实际用 reduce-scatter + all-gather 更省
        return out

# ---------- 3. 3D 并行配置搜索 ----------
def suggest_3d(num_gpus, gpus_per_node, model_bytes, card_bytes=80e9,
               activation_headroom=0.4):
    """返回 (tp, pp, dp) 建议"""
    usable = card_bytes * (1 - activation_headroom)
    min_shard = model_bytes / usable            # 至少要切成这么多份
    tp = min(gpus_per_node, max(1, int(min_shard)))
    while gpus_per_node % tp: tp -= 1
    pp = 1
    while model_bytes / (tp * pp) > usable:
        pp *= 2
    if tp * pp > num_gpus:
        return None                             # 需要 ZeRO-3
    dp = num_gpus // (tp * pp)
    return {"tp": tp, "pp": pp, "dp": dp}

print(suggest_3d(256, 8, 70e9 * 16 + 70e9 * 2))
```

## 面试追问

**Q：为什么 TP 每个 Transformer 层只需 2 次 all-reduce？**
A：因为切分方式被精心设计过。Attention 的 W_q/W_k/W_v 按 head **列切**（每卡独立算自己的头，无通信），W_o **行切**（需要 all-reduce 合并）→ 1 次。FFN 的 W_gate/W_up **列切**（无通信），W_down **行切**（all-reduce）→ 1 次。合计每层 2 次。这个"列切-行切"配对是使用非线性（SiLU、attention）的前提：非线性必须在列切之后、行切之前，不能跨卡。

**Q：TP 和 ZeRO-3 都切参数，有什么区别？**
A：切分的粒度和通信模式不同。TP 把单个矩阵**永久地**切成 t 份，每张卡只存 1/t，前向时通过 all-reduce 重建完整激活——通信的是**激活**（每层 2 次，量大但延迟小、易重叠）。ZeRO-3 把参数**动态地** gather：需要某一层时临时 all-gather 完整参数，用完立即释放——通信的是**参数**（每层 1~2 次 all-gather + 反向的 reduce-scatter，总量约 3× 参数量）。所以 TP 通信量大但更适合低延迟的 NVLink，ZeRO-3 通信量更大但可以用 prefetch 重叠。实践上节点内用 TP、跨节点用 ZeRO-3 或 PP。

**Q：PP 的 bubble 能完全消除吗？**
A：不能完全消除，但能压得很低。1F1B 的 bubble 是 `(p-1)/m`，靠增大微批数 m 压低；interleaved 1F1B（每 stage 负责多个不连续块）能再减半；更激进的如 Zero Bubble 调度把反向拆成"输入梯度"和"权重梯度"两步分别调度，理论上可以把 bubble 压到接近 0，但实现复杂、显存代价更高。

**Q：微批大小怎么选？**
A：权衡 bubble 和 GEMM 效率。微批太小时：(1) bubble 比例不变但绝对时间占比高；(2) GEMM 矩阵太小，Tensor Core 打不满。微批太大时：激活显存不够。经验值：**micro_batch × seq_len ≈ 4096~16384 tokens**（如 seq 4096 时 micro_batch 取 1~4），且总微批数 m ≥ 4×pp。

**Q：DP/TP/PP 能不能和 ZeRO 混用？**
A：能，而且很常见。典型组合是 **ZeRO-1/2 + TP + PP**：ZeRO 切优化器状态和梯度（DP 维度上），TP 切节点内的矩阵，PP 切跨节点的层。注意 ZeRO-3（切参数）与 TP 有一定冲突——两者都在切参数，同时开会导致切分过细、通信模式复杂，实践中通常二选一（节点内 TP、跨节点 ZeRO-3）。

**Q：序列并行（SP）和 TP 是什么关系？**
A：SP 是 TP 的补充，不是独立的并行方式。TP 切分后，attention/FFN 的激活被切了，但 LayerNorm、Dropout、残差加这些逐元素操作的激活仍是完整的 `B×S×d`。SP 把这些沿序列维切成 `B×(S/tp)×d`，把 TP 的两个 all-reduce 换成 reduce-scatter + all-gather，这样整个 Transformer 层的激活全部被切分，显存再省 tp 倍。Megatron 的 SP 就是这个设计，代价是通信量不变（reduce-scatter + all-gather 的总量等于 all-reduce）。

**Q：超长上下文（如 128k）怎么并行？**
A：需要**序列维的并行**，因为激活显存随 S 线性增长，TP 切不动（TP 只切 d 维）。方案有两类：
(1) **Ring Attention / Context Parallel**：把序列切成 n 份分到不同卡，每张卡算本地 Q 的 attention，通过 ring 传递 K/V 块，通信量与计算完全重叠；
(2) **DeepSpeed-Ulysses**：在 attention 计算时用 all-to-all 把"序列维"换成"头维"，这样每张卡有全部序列但只有部分头，算完再 all-to-all 换回来。Ulysses 的通信量更小但要求头数能被并行度整除，Ring Attention 更通用但有延迟累积。

## 关联知识点
- ZeRO 与 FSDP（04-distributed）
- 通信原语与通信量推导（04-distributed）
- 训练显存估算与激活重计算（04-distributed）
- MoE 与专家并行 EP（03-pretrain）
- 长上下文训练（02-attention）
