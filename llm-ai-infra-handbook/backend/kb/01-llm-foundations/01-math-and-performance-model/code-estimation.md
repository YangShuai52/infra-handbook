---
id: code-estimation
title: 手撕估算题：显存 / 6ND FLOPs / 训练天数
module: 01-math-performance
tags:
- 手撕代码
- 估算
- 显存
- FLOPs
- MFU
- roofline
- 训练时间
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 面试必考的"算一笔账"：训练显存（模型+优化器+激活）、推理显存（权重+KV Cache）、训练 FLOPs≈6ND 与训练时间（含 MFU）、推理算力强度与
  roofline。全部给可运行脚本与真实算例。
source:
- https://doi.org/10.1145/1498765.1498785
questions:
- 训练 16 bytes/param 怎么拆的？为什么优化器占大头？
- 推理 KV Cache 公式的每一项是什么？为什么长上下文高并发会爆显存？
- 6ND 的 6 怎么来的（前向 2N + 反向 4N）？
- MFU 是什么？为什么训练 MFU 常只有 0.4~0.5？
- 怎么判断一个算子是 memory-bound 还是 compute-bound（roofline）？
---

## 一句话结论

- **训练显存**：≈ `16·P` 字节（fp16 参数/梯度各 2B + AdamW 优化器 12B）。
- **推理显存**：权重 `2·P` + KV Cache `2·2·L·S·B·d_kv·n_layers`（bf16）。
- **训练 FLOPs**：`C ≈ 6·N·D`（每 token 前向 2N + 反向 4N，N=参数，D=token 数）。
- **训练时间**：`T = 6ND / (GPU数 × 单卡 FLOPS × MFU)`，MFU 常取 0.4~0.5。
- **Roofline**：算力强度 `I = FLOPs/Byte`，低于带宽拐点走 memory-bound、高于走 compute-bound。

## 实现要点

- 所有估算用 **参数 P（模型参数量）** 与 **token 数 D** 表达，代入具体数字即可。
- KV Cache 公式里的 `2`（K 和 V）`×2`（字节，bf16）易漏，写清。
- MFU 是"实际吞吐 / 理论峰值"的比值，重训练比推理低。

## 输入输出示例

```python
estimate_train_memory(P=7e9)                     # ≈ 112 GB（未计激活）
estimate_train_time(P=7e9, D=1e12, gpus=64,
                    peak_tflops=312, mfu=0.45)    # ≈ 54 天
```

## 核心代码

```python
def estimate_train_memory(P, bytes_per_param=16):
    """训练单卡常驻显存（不含激活/碎片），P=参数量"""
    return P * bytes_per_param          # fp16 参数2 + 梯度2 + AdamW 12 = 16

def estimate_infer_memory(P, L, S, B, d_kv, n_layers, dtype_bytes=2):
    """推理显存：权重 + KV Cache（bf16 示例）"""
    weights = P * dtype_bytes
    kv = 2 * L * S * B * d_kv * n_layers * dtype_bytes   # K,V × 序列×batch×维×层
    return weights + kv

def estimate_train_flops(N, D):
    return 6 * N * D                    # 每 token 6N FLOPs

def estimate_train_time(N, D, gpus, peak_tflops, mfu=0.45):
    total = 6 * N * D
    per_gpu = peak_tflops * 1e12 * mfu   # FLOPS
    seconds = total / (gpus * per_gpu)
    return seconds / 86400               # 天

def roofline(flops, bytes_, peak_tflops, bandwidth_tbs):
    I = flops / bytes_                  # FLOPs per byte
    I_max = peak_tflops * 1e12 / (bandwidth_tbs * 1e12)
    achieved = min(peak_tflops * 1e12, I * bandwidth_tbs * 1e12)
    bound = "compute" if I >= I_max else "memory"
    return I, bound, achieved

# ---- 真实算例：7B 模型，A100 80G，fp16 ----
if __name__ == "__main__":
    P, D, gpus = 7e9, 1e12, 64
    print("训练常驻显存/卡:", estimate_train_memory(P)/1e9, "GB")     # ≈112 GB
    print("训练天数:", round(estimate_train_time(P, D, gpus, 312), 1)) # A100=312 TFLOPS → ~54 天
    # 推理 KV：L=32层, S=4096, B=1, d_kv=128
    print("推理 KV(单请求):", estimate_infer_memory(P,32,4096,1,128,32)/1e9, "GB")  # ~2.1 GB
```

**算例核对**：
- 7B 训练常驻 ≈ 16×7e9 = 112 GB（单卡放不下 → 需 ZeRO/FSDP）。
- 7B × 1T token，64×A100(312 TFLOPS)×MFU0.45：`6·7e9·1e12 ≈ 4.2e22` FLOPs；分母 `64·312e12·0.45 ≈ 9.0e15`/s → ≈ **54 天**。
- 推理 KV（单请求 4k、32 层、128 维）= `2·32·4096·1·128·32·2 ≈ 2.1 GB`，高并发时主导显存。

## 面试追问

1. 训练 16 bytes/param 怎么拆的？为什么优化器占大头？
2. 推理 KV Cache 公式的每一项是什么？为什么长上下文高并发会爆显存？
3. 6ND 的 6 怎么来的（前向 2N + 反向 4N）？
4. MFU 是什么？为什么训练 MFU 常只有 0.4~0.5？
5. 怎么判断一个算子是 memory-bound 还是 compute-bound（roofline）？

## 关联知识点

- [ZeRO 与 FSDP：优化器状态、梯度与参数的分片](zero-and-fsdp)
- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
- [显存与激活重计算：四大块逐项估算](memory-and-checkpointing)