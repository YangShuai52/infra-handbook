# EP 并行原理与 vLLM-Ascend 实现

本文整理 MoE 模型中 Expert Parallel（EP，专家并行）的基本原理、Router 与 Token Dispatch 数据流、TP/DP/EP 的关系，以及 vLLM-Ascend 中 AllToAll、MC2、FlashComm1 等实现路径。示例主要参考 Qwen3.5-397B-A17B 与 `vllm_ascend/ops/fused_moe/token_dispatcher.py`。

## 1. 为什么需要 EP

MoE 模型的总参数量可能很大，但每个 Token 只激活少量专家。例如 Qwen3.5-397B-A17B：

- 397B 表示整个模型总参数量；
- A17B 表示每个 Token 前向过程中大约激活 17B 参数；
- A17B 不表示加载时只需要存放 17B 权重。

以 W4A8 粗略估算，仅 397B 个 4-bit 权重就约为：

```text
397B × 4 bit ≈ 198.5 GB
```

还没有计入量化 Scale、非 4-bit 权重、对齐空间、KV Cache 和运行时 Workspace，因此单张 64 GB NPU 无法保存完整模型。

EP 的核心做法是：

> 非专家层按原有 TP/DP 方式执行；大量 MoE Expert 权重分散到多个 EP Rank，每个 Rank 只保存部分完整专家。

例如模型有 64 个专家、EP Size 为 16，连续分配时每个 Rank 保存 4 个专家：

```text
EP Rank 0  → Expert  0～3
EP Rank 1  → Expert  4～7
...
EP Rank 15 → Expert 60～63
```

注意，EP 通常是“按专家切分”，不是把单个专家矩阵沿 Hidden Size 切开。某个专家的完整权重一般位于其所属 EP Rank 上。

## 2. 一个 MoE 层的完整数据流

一个普通 MoE 层可以抽象为：

```text
输入 hidden_states [T, H]
          │
          ▼
Router Linear → router_logits [T, E]
          │
          ▼
select_experts
  ├── topk_ids     [T, K]
  └── topk_weights [T, K]
          │
          ▼
Dispatch：按专家路由 Token
  Permute → AllToAll → 本地专家排序
          │
          ▼
本 Rank 的 Expert FFN / Grouped MatMul
          │
          ▼
Combine：把结果送回 Token 原属 Rank
  撤销本地排序 → 反向 AllToAll → 恢复 Token 顺序
          │
          ▼
按 topk_weights 合并 K 路专家输出
          │
          ▼
输出 hidden_states [T, H]
```

其中：

- `T`：当前参与 MoE 计算的 Token 数；
- `H`：Hidden Size；
- `E`：全局专家数；
- `K`：每个 Token 选中的专家数。

## 3. Router 在哪里计算

### 3.1 Router 属于谁

Router 是每个 MoE 层中的门控网络，通常是一个线性层：

```text
router_logits = hidden_states @ router_weight
```

Router 权重通常在并行 Rank 间复制，而 Expert 权重才按照 EP 切分。每个持有一部分 Token 的 Rank，会为自己的 Token 在全部逻辑专家 `E` 上计算 Router Logits，并选择全局 Top-K 专家。

因此不能把 Router 简单理解成“位于某个特定 EP Rank 上”。同一组物理进程在不同算子阶段会扮演 TP Rank、DP Rank 或 EP Rank；Router 在本地 Token 上执行，专家归属由 `topk_ids` 决定。

### 3.2 Router 在 Token 切分前还是后

这取决于进入 MoE 层之前的数据布局。

普通 TP、未使用 Sequence Parallel 时：

1. Attention 的 Row Parallel 输出做 AllReduce；
2. 每个 TP Rank 都得到完整的 `[T, H]`；
3. Router 在完整 Token 集合上得到 `[T, E]`；
4. EP Prepare 再把 `hidden_states` 和 `router_logits` 沿 Token 维度做相同切分，避免重复路由相同 Token。

使用 Sequence Parallel 或 FlashComm1 的优化路径时：

1. 前序 TP 聚合可能由 AllReduce+Split 融合成 ReduceScatter；
2. 每个 Rank 直接得到本地 Token 分片 `[T/TP, H]`；
3. Router 只为本地 Token 计算 `[T/TP, E]`；
4. Prepare 不再重复切分。

所以“Router 前还是后切 Token”没有脱离执行路径的唯一答案。关键约束是：

> `hidden_states` 与对应的 `router_logits` 必须描述同一批 Token，并保持 Token 维度完全对齐。

### 3.3 为什么 Router Logits 也要切分

假设普通 TP 路径中每张卡都有相同的 `[T, H]`，Router 也会算出相同 `[T, E]`。如果只切 Hidden States、不切 Logits，会出现：

```text
hidden_states: [T/TP, H]
router_logits: [T, E]
```

两者第一维不对应，无法进行逐 Token 的 Top-K 选择。因而必须沿 Token 维度应用相同切分：

```text
hidden_states → [T/TP, H]
router_logits → [T/TP, E]
```

切 Router Logits 不是为了节省 Router 权重，也不是沿专家维度切 Router，而是为了保持 Token 对齐并消除 TP Rank 间的重复 Token。

## 4. `select_experts()` 与 Top-K 权重

典型调用为：

```python
topk_weights, topk_ids = select_experts(
    hidden_states=x,
    router_logits=router_logits,
    ...
)
```

虽然传入 `hidden_states`，专家选择的核心依据是 `router_logits`。具体实现会根据模型配置和硬件支持选择融合 NPU 门控算子或原生计算，并完成：

1. 对 Router Logits 使用 Softmax、Sigmoid 或模型指定的打分函数；
2. 可选执行分组 Top-K；
3. 可选加入专家修正 Bias；
4. 选择每个 Token 得分最高的 K 个专家；
5. 可选对选中分数重新归一化；
6. 乘以 `routed_scaling_factor` 等缩放系数。

输出含义：

```text
topk_ids[t, j]     = Token t 选择的第 j 个专家编号
topk_weights[t, j] = 该专家输出在最终合并时的系数
```

最终结果通常是：

```text
y_t = Σ_j topk_weights[t, j] × Expert[topk_ids[t, j]](x_t)
```

## 5. 为什么仍然需要 AllToAll

一个常见疑问是：TP Attention 结束后，每张卡不是都有完整序列吗？各卡算出的 Router 也相同，为什么还要 AllToAll？

原因在于：

1. TP AllReduce 让各 Rank 得到完整 Hidden 值，但不意味着每张卡拥有所有专家权重；
2. EP 下每个专家只位于其所属 Rank；
3. Token 选择的专家很可能在其他 Rank 上；
4. 必须把完整 Token Hidden 行发送给专家拥有者计算，再把结果送回。

如果让每个 TP Rank 对相同 Token 重复 Dispatch，会产生重复计算。普通实现会先沿 Token 维度切分，使每个 Token 只由一个源 Rank 发起专家路由；Sequence Parallel 路径则更早完成这一切分。

## 6. AllReduce 与 AllToAll 的区别

### 6.1 TP AllReduce

TP 的 Row Parallel 层中，各 Rank 得到的是同一个输出向量的部分和：

```text
rank0: partial_y0 [T, H]
rank1: partial_y1 [T, H]
...
```

AllReduce 执行逐元素求和：

```text
y = partial_y0 + partial_y1 + ...
```

之后每个 TP Rank 都持有相同的完整 `[T, H]`。这里不是沿 Hidden Size 拼接。

### 6.2 EP AllToAll

EP AllToAll 按目标专家所在 Rank 搬运 Token 行：

```text
[若干 Token, H] → 分发到不同 EP Rank
```

每一行仍包含完整 Hidden Size `H`，通信发生在 Token/路由副本维度，不是在 Hidden Size 上聚合。

两者的目的不同：

| 通信 | 主要用途 | 数据语义 |
|---|---|---|
| AllReduce | 合并 TP 部分和 | 同一位置逐元素求和 |
| ReduceScatter | 求和后再分片 | 常用于 TP+Sequence Parallel |
| AllGather | 收集各 Rank 分片 | 每个 Rank 获得所有分片 |
| AllToAllV | 交换不同数量的 Token 行 | 将 Token 发往专家所属 Rank |

## 7. AllToAllV Dispatch 的源码逻辑

vLLM-Ascend 的普通 EP 路径主要由 `TokenDispatcherWithAll2AllV` 实现。文件结构包括：

- `MoETokenDispatcher`：统一抽象接口；
- `TokenDispatcherWithMC2`：MC2 通信路径；
- `TokenDispatcherWithAllGather`：AllGather 路径；
- `TokenDispatcherWithAll2AllV`：变长 AllToAll 路径。

### 7.1 Dispatch 前处理

`_dispatch_preprocess()` 主要完成四件事。

第一，统计当前源 Rank 路由到每个全局专家的 Token 副本数：

```python
num_global_tokens_per_expert = torch.histc(topk_ids, bins=num_experts)
```

第二，根据专家归属计算：

```text
input_splits[destination_rank]
```

它表示本 Rank 要向每个目标 EP Rank 发送多少行。

第三，AllGather 各 Rank 的专家计数，截取本 Rank 负责的专家列，得到：

```text
output_splits[source_rank]
```

它表示本 Rank 将从每个源 Rank 接收多少行。同时汇总出每个本地专家的 Token 数，用于生成 Grouped MatMul 的 `group_list`。

第四，调用 `npu_moe_token_permute()`：

- 一个 Token 被展开为 K 个专家路由副本；
- 按全局专家编号排序；
- 连续专家归属于连续 Rank 时，也形成按目标 Rank 排列的连续数据段；
- 保存第一次排序的逆映射，供 Combine 恢复。

### 7.2 第一次 AllToAll

`async_all_to_all()` 最终调用类似：

```python
dist.all_to_all_single(
    output,
    input_,
    output_split_sizes=output_splits,
    input_split_sizes=input_splits,
    async_op=True,
)
```

这是 AllToAllV：每对 Rank 的发送数量可以不同。Token 路由天然不均匀，因此不能假设每个 Rank 收发相同数量。

若启用动态量化，相应 Scale 也需使用相同 Split 信息进行通信。

### 7.3 Dispatch 后处理

第一次 AllToAll 后，数据通常按源 Rank 分段，不一定按本地专家连续。因此 `_dispatch_postprocess()` 会执行第二次 Permute：

```text
按来源排列的 Token → 按本地 Expert 排列的 Token
```

并保存第二个逆映射。此后专家 FFN 使用 `group_list` 执行 Grouped MatMul。

### 7.4 Combine

专家计算完成后：

1. 使用第二个逆映射撤销本地专家排序；
2. 交换 `input_splits` 与 `output_splits`，执行反向 AllToAll；
3. 使用第一个逆映射恢复原始 Token 顺序；
4. 使用 `topk_weights` 合并一个 Token 对应的 K 路专家输出。

因此普通 EP 的两次 Permute 分别服务于：

```text
Permute 1：按目标 Rank/全局专家组织发送数据
Permute 2：按本地专家组织计算数据
```

Combine 则按相反顺序撤销它们。

## 8. 某个 Rank 只接收不发送可以吗

可以。AllToAllV 允许某个 Rank：

- 只发送、不接收；
- 只接收、不发送；
- 收发数量不相等；
- 本轮收发量全部为零。

例如：

```text
Rank 0 input_splits  = [0, 0, 0, 0]
Rank 0 output_splits = [3, 5, 0, 2]
```

表示 Rank 0 当前没有自己的 Token 要发出，但其他 Rank 有 10 份 Token 选择了 Rank 0 持有的专家。Rank 0 仍必须参加该 Collective。专家计算后，Combine 阶段通信方向反转，它会把结果发回原 Token Rank。

所有通信组成员都必须调用同一次 Collective。即使某 Rank 所有 Split 为零，也不能跳过，否则其他 Rank 可能等待并最终超时或死锁。

## 9. TP、DP、EP 的组合关系

三个维度含义不同：

- TP：切分 Attention、Dense 等张量计算；
- DP：不同 Rank 处理不同请求或 Token 批次；
- EP：切分 MoE 专家权重，并在 MoE 层交换 Token。

在 vLLM 的常见 MoE EP 布局中，启用 `--enable-expert-parallel` 后，参与同一模型执行组的 TP/DP Rank 可共同构成 EP 组，常见关系为：

```text
EP Size = TP Size × DP Size
```

具体仍应以当前版本启动日志打印的并行组为准，尤其是跨节点 External DP、PP 或定制部署框架场景。

### 9.1 `DP32 × TP1 × EP32` 为什么能装下 397B

传统 `DP32 × TP1` 表示 32 张卡各复制一份完整模型，397B W4A8 显然无法装入单张 64 GB 卡。

但开启 EP 后，准确的数据布局是：

```text
Attention、Norm、Embedding、共享部分：按 DP Rank 复制（TP1）
大量 Routed Expert 权重：在 EP32 上分片
```

于是每张卡保存：

```text
约 1/32 的 Routed Expert 权重
+ 一份非专家/共享权重
+ KV Cache
+ 运行时 Workspace
```

因此它不是 32 个彼此独立的完整模型副本。DP Rank 在非 MoE 层表现为数据并行，在 MoE 层同时充当 EP Rank，并通过 AllToAll/MC2 协作完成专家计算。

## 10. `--enable-expert-parallel` 默认是 EP 几

`--enable-expert-parallel` 是开启专家并行的开关，本身不直接携带一个 EP 数值。EP Size 由当前 TP、DP、PP 分组和版本实现推导。

例如常见单 Stage 配置：

```bash
--data-parallel-size 1
--tensor-parallel-size 16
--enable-expert-parallel
```

通常对应：

```text
EP Size = 1 × 16 = 16
```

而：

```bash
--data-parallel-size 32
--tensor-parallel-size 1
--enable-expert-parallel
```

通常对应 EP32。最终应检查启动日志中的 EP Group、EP Rank 和本地专家分配信息，不能只凭开关名称判断。

## 11. MC2、Fused MC2 与普通 AllToAll EP

### 11.1 普通 MC2

MC2 可以理解为昇腾上的 MoE 通信—计算协同路径。普通 AllToAll EP 将过程拆成：

```text
Permute → AllToAll → Expert FFN → 反向 AllToAll → Unpermute
```

MC2 使用类似 `npu_moe_distribute_dispatch` 和 `npu_moe_distribute_combine` 的专用算子，减少显式通信编排和同步开销，但 Expert FFN 仍处于 Dispatch 与 Combine 之间。

### 11.2 Fused MC2

Fused MC2 进一步融合：

```text
Dispatch + Expert FFN + Combine
```

具体使用 `dispatch_ffn_combine` 或版本对应的 MegaMoE 路径。它可能降低中间 Tensor、Kernel Launch 和同步开销，但有更严格的约束，例如：

- 芯片和 CANN/算子版本；
- EP Size；
- 量化格式及 Scale/Bias 布局；
- 模型 Expert 结构；
- MTP、动态 EPLB 等特性的组合限制。

所以 W4A8 是否支持 Fused MC2，不能只由“W4A8”三个字判断，必须结合 vLLM-Ascend 分支、CANN、Checkpoint 量化元数据和实际选中的算子路径验证。

### 11.3 Shared Expert Multistream

`multistream_overlap_shared_expert` 用于模型存在 Shared Expert 时，让共享专家计算与 Routed Expert 的通信/计算在不同 Stream 上重叠。

它不是开启 MC2 后自动获得的特性。普通 MC2 在满足实现条件时可能与它配合；Fused MC2 已把 Routed Expert 的 Dispatch、FFN、Combine 深度融合，在相关实现中通常会关闭 Shared Expert Multistream，两者不能简单叠加。

## 12. FlashComm1 与 EP

FlashComm1 主要是昇腾 TP/Sequence Parallel 通信优化，不等同于 MC2。

它可能把 MoE 前后的：

```text
TP AllReduce + Token Split
```

优化为更合适的 ReduceScatter/AllGather 数据流，使每个 Rank 更早获得 Token 分片，减少重复 Router 和显式切分开销。

因此：

- FlashComm1 优化 TP 与 Sequence Parallel 衔接；
- AllToAll/MC2 负责 Token 到 Expert Rank 的路由；
- 二者解决的问题不同，可以在兼容的执行路径中共同使用。

## 13. 与部署参数的关系

### 13.1 `--distributed-executor-backend mp`

该参数表示 vLLM 使用 Python Multiprocessing 管理本机 NPU Worker：

```text
主进程
├── Worker 0 → NPU 0
├── Worker 1 → NPU 1
└── ...
```

它是进程启动与管理方式，不是模型并行方式：

```text
TP/DP/EP → 决定权重和计算布局
HCCL     → 执行 NPU 集合通信
mp       → 管理本机 Worker 进程
```

多机时，`mp` 通常只负责节点内 Worker；跨节点进程仍需由部署脚本、External DP、Ray 或外部 Launcher 等方式协调。

### 13.2 `--max-model-len`

限制单条序列的最大累计长度，通常包括 Prompt Token 与生成 Token：

```text
prompt_tokens + generated_tokens ≤ max_model_len
```

### 13.3 `--max-num-batched-tokens`

限制调度器单次迭代处理的总 Token 预算，是同一轮中所有请求 Token 数的总和。开启 Chunked Prefill 时，它会限制一个长 Prompt 每轮最多能取得多少预算，所以在“只有一个长 Prefill 请求”时近似其最大 Chunk 大小，但它不是一个固定 Chunk Size；混合多个 Prefill/Decode 请求时预算会被共同占用。

### 13.4 请求频率 `req/s`

性能测试里的请求频率通常表示请求到达系统的 offered load，即压测器每秒尝试发起多少请求，不是服务实际完成的吞吐。实际吞吐可能低于请求频率，并伴随排队、TTFT 和端到端时延上升。

## 14. PD 分离中的 P/D 数量与 EP

PD 分离把 Prefill 和 Decode 放在不同服务组：

```text
请求 → Prefill 生成 KV Cache → 传输到 Decode → 逐 Token 生成
```

`2P1D` 通常表示两个 Prefill 逻辑实例/集群和一个 Decode 逻辑实例/集群，不必等于三台机器。一个逻辑实例可以跨多机，也可以内部包含多个 DP Replica。

例如 4 台机器、每台 16 卡：

```text
P0：1 台，DP2 × TP8 = 16 卡
P1：1 台，DP2 × TP8 = 16 卡
D0：2 台共同组成 DP32 × TP1 × EP32 = 32 卡
```

从 PD 逻辑拓扑看是 `2P1D`；从物理节点看是 2 台 P 节点加 2 台 D 节点；从模型执行看，P 侧有 4 个 TP8 数据并行 Replica，D 侧有 32 个 TP1 Rank，同时在 MoE 层组成 EP32。

### Hybrid KV Cache Manager

Qwen3.5 等模型可能混合 Full Attention 与 Gated DeltaNet/Linear Attention 层，不同层保存的状态不同：

```text
Full Attention → 标准 K/V Cache
Linear Attention/Mamba 类层 → SSM State、Conv State 等
```

`--no-disable-hybrid-kv-cache-manager` 是双重否定，表示启用 Hybrid KV Cache Manager。它按不同 Cache 类型分组分配、释放和传输，与 EP 权重切分没有直接关系。PD 场景中 P/D 两端及 KV Connector 必须对这些 Cache Group 的 Block Size、Stride 和状态布局保持一致。

## 15. 典型 Qwen3.5 EP 配置的解读

```bash
vllm serve /path/to/Qwen3.5-397B-A17B-w4a8 \
  --quantization ascend \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 128 \
  --max-model-len 6144 \
  --max-num-batched-tokens 16384
```

常见含义是：

```text
TP16：Attention/普通 Dense 层在 16 Rank 上进行张量并行
DP1：只有一个数据并行组
EP16：MoE 专家在同一组 16 Rank 上分布
```

若同时配置：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
```

则意图是：

- FlashComm1 优化 TP/Sequence Parallel 数据衔接；
- Fused MC2 尝试使用融合的 MoE Dispatch+FFN+Combine 路径。

是否真正生效要查看启动日志和运行时选路，不能仅根据环境变量存在就认定。不同版本可能推荐通过 `--additional-config` 配置，环境变量名称和优先级也可能变化。

## 16. 调试时应确认的证据

判断 EP 是否按预期工作，建议检查：

1. 启动日志中的 TP、DP、EP Size 和 Rank 分组；
2. 每个 Rank 加载的本地 Expert 数及 Expert ID 范围；
3. MoE 通信类型是 AllToAll、AllGather、MC2 还是 Fused MC2；
4. FlashComm1/Sequence Parallel 是否实际选中；
5. W4A8 Scale、Bias 和 Expert 权重布局是否满足融合算子要求；
6. P/D 两端并行配置与 KV Connector 描述是否一致；
7. 首个真实推理请求是否成功，不能只以服务启动完成作为验证；
8. Token 分布是否失衡，以及是否存在某些专家过热导致的通信和计算尾延迟。

## 17. 关键源码位置

以下路径基于 vLLM-Ascend 代码仓：

- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
  - `MoETokenDispatcher`：统一 Dispatch/Combine 接口；
  - `TokenDispatcherWithMC2`：MC2 路径；
  - `TokenDispatcherWithAllGather`：AllGather 路径；
  - `TokenDispatcherWithAll2AllV`：普通变长 AllToAll 路径。
- `vllm_ascend/ops/fused_moe/comm_utils.py`
  - `async_all_to_all()`：分配输出并调用 `dist.all_to_all_single()`。
- `vllm_ascend/ops/fused_moe/experts_selector.py`
  - `select_experts()`：Router 打分、Top-K、归一化与权重缩放。
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
  - 普通 TP/EP 路径中 Hidden States 与 Router Logits 的 Token 切分及恢复。
- `vllm_ascend/ops/register_custom_ops.py`
  - TP AllReduce/ReduceScatter 等自定义通信路径。
- `vllm_ascend/ascend_forward_context.py`
  - MoE 通信类型和运行时路径选择。

## 18. 总结

EP 的本质不是“每张卡都加载一份 MoE 模型”，而是：

> Router 在本地 Token 上从全局专家中选择 Top-K；Token 通过 AllToAll 或 MC2 被送到专家权重所在 Rank；专家输出再返回原 Rank，恢复 Token 顺序并按 Top-K 权重融合。

理解 EP 时抓住三条主线即可：

1. **权重布局**：Routed Experts 按 EP Rank 分布，非专家部分由 TP/DP 规则决定；
2. **数据布局**：通信搬运的是完整 Hidden 行，主要沿 Token/路由副本维度重排；
3. **通信闭环**：Dispatch 把 Token 送到专家，Combine 把输出送回来并完成加权。

