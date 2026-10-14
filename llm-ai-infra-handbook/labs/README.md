# Labs

这里承载可复现、可测量的实践，不把一次性演示混入概念卡片。

- `foundations/`：张量形状、参数量、FLOPs、显存与 Roofline 估算。
- `training/`：数据、稳定性、多维并行、Checkpoint 与故障恢复。
- `kernels/`：CUDA/Triton 算子、数值正确性与性能剖析。
- `inference/`：KV Cache、量化、动态批处理与服务压测。
- `production/`：调度、SLI/SLO、追踪、容量规划与事故演练。

每个实验应记录环境、输入、基线、变量、结果、原始证据和结论边界。目录会在对应实验落地时逐步创建，避免空目录占位。
