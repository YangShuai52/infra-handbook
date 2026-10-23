---
id: mixed-precision-hardware
title: FP32、TF32、BF16、FP16、FP8 与 INT
module: 12-gpu-kernels-compilers
tags:
- GPU 编程、算子与编译器
- FP32
- AI-Infra
difficulty: 3
freq: 中
updated: '2026-09-04'
source:
- https://triton-lang.org/main/getting-started/tutorials/index.html
questions:
- 请从数据流和瓶颈角度解释「FP32、TF32、BF16、FP16、FP8 与 INT」的核心机制。
- 线上指标异常时，如何用可观测证据验证「FP32、TF32、BF16、FP16、FP8 与 INT」相关假设？
- 「FP32、TF32、BF16、FP16、FP8 与 INT」在质量、性能、成本和复杂度之间如何权衡？
summary: 数值格式在范围、精度、吞吐和带宽间取舍；累加精度通常高于乘法输入精度。
---

## 一句话结论

数值格式在范围、精度、吞吐和带宽间取舍；累加精度通常高于乘法输入精度。

## 原理与系统位置

- **核心对象**：FP32、TF32、BF16、FP16、FP8 与 INT。
- **分析维度**：线程层级、访存、tiling、fusion、编译和 profiling。
- **因果链**：先明确输入规模与状态，再写出计算、内存、通信或 I/O 路径，最后定位关键路径上的限制因素。

## 工程权衡

落地时必须同时核对正确性、质量、吞吐/延迟、资源成本与运维复杂度。单点峰值不能代表端到端收益；局部优化还可能把瓶颈推到队列、网络、存储或下游服务。

## 具体例子

做设计评审时，先列出 batch、序列长度、并发、数据精度和拓扑等输入，再估算资源账本；上线后以 occupancy、带宽、FLOPS、stall、launch 和数值误差 对照估算。若观测与模型不一致，应优先检查隐式同步、缓存命中、数据倾斜和测量口径。

## 观测与排障

先用端到端指标确认用户影响，再沿关键路径下钻。每次只改变一个变量，保留基线、版本和逐样本/逐请求证据，避免把相关性误判为因果。

## 面试追问

1. 请从数据流和瓶颈角度解释「FP32、TF32、BF16、FP16、FP8 与 INT」的核心机制。
2. 线上指标异常时，如何用可观测证据验证「FP32、TF32、BF16、FP16、FP8 与 INT」相关假设？
3. 「FP32、TF32、BF16、FP16、FP8 与 INT」在质量、性能、成本和复杂度之间如何权衡？
