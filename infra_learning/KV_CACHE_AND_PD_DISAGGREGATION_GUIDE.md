# vLLM-Ascend KV Cache 与 PD 分离技术详解

本文档整理了 vLLM-Ascend 中 KV Cache 管理、PD 分离部署、流水线并行（PP）、KV Offloading、KV Cache 池化等核心特性的原理与流程，适用于深入理解 vLLM-Ascend 在 Ascend NPU 上的推理优化机制。

---

## 目录

- [1. Prefix Caching（前缀缓存）](#1-prefix-caching前缀缓存)
- [2. PD 分离架构](#2-pd-分离架构)
- [3. Pipeline Parallelism（流水线并行）](#3-pipeline-parallelism流水线并行)
- [4. Layerwise（逐层传输）](#4-layerwise逐层传输)
- [5. Recompute Scheduler（重算调度器）](#5-recompute-scheduler重算调度器)
- [6. KV Offloading（KV 卸载）](#6-kv-offloadingkv-卸载)
- [7. KV Cache Pool（KV 缓存池化）](#7-kv-cache-poolkv-缓存池化)
- [8. MooncakeConnector（P2P KV 传输）](#8-mooncakeconnectorp2p-kv-传输)
- [9. 特性对比总览](#9-特性对比总览)

---

## 1. Prefix Caching（前缀缓存）

### 原理

vLLM 把 KV cache 按 block（如 128 token）切分并 hash 索引。当一个请求的某段 token 序列（前缀）已经被之前任意请求计算过，且 block 仍在缓存池中，该段 KV block 直接复用，跳过 prefill 计算。

### 两种命中场景

1. **同一请求内部**：多轮对话中，第二轮请求 token = [system prompt + 历史对话 + 新问题]，前缀部分之前算过 → 命中，只算新问题部分
2. **不同请求之间**：两个请求共享相同 system prompt 或相同开头，第二个请求的那部分前缀直接复用第一个请求算过的 KV block

关键要求是 **token 序列完全一致**（token-level 精确匹配至 block 边界），不是字符串看起来一致。

### 存储管理

- **不是永久保存所有请求的 KV**：HBM 容量有限，用 LRU（Least Recently Used）淘汰最久未访问的 block
- **缓存池大小**：启动时按 `gpu_memory_utilization` 分配
- **block 状态**：分为「带数据可复用」和「空 block」两类，新请求优先用空 block，不够才驱逐 LRU 的带数据 block
- **关闭 prefix caching 时**：请求结束后 KV block 直接回到 free pool 并清零，无跨请求匹配

### 命中率影响因素

| 因素 | 影响 |
|------|------|
| 工作负载前缀重复度 | 高重复 → 高命中率；随机 benchmark → 0% 命中率 |
| 缓存池大小 | 池越大，LRU 驱逐越慢，复用窗口越长 |
| block_size | prompt 长度 < block_size 时无完整 block 可命中 |

### 310P 硬件注意点

- 310P（Atlas 300I Duo）HBM 较小（~32GB），单并发下短 prompt 一般能命中，长 prompt 或多轮后可能因 LRU 驱逐导致部分命中
- 310P 走 `vllm_ascend/_310p/model_runner_310p.py`，prefix caching 路径已支持
- vLLM V1 默认开启 prefix caching

---

## 2. PD 分离架构

### 核心思想

将推理的 **Prefill 阶段**和 **Decode 阶段**分离到不同节点：

- **P 节点（Prefiller）**：专注 prefill 吞吐，按长 prompt 优化算力配比
- **D 节点（Decoder）**：专注 decode 吞吐，按低延迟 token-by-token 优化
- **KV Cache 传输**：P 节点 prefill 完成后，KV cache 通过 P2P 网络传到 D 节点

### 数据流

```
用户请求 → Proxy → P 节点（prefill，生成 KV）
                  ↓ KV cache P2P 传输（仅传 KV，不传 prompt）
              D 节点（从已有 KV 开始 decode）
              ↓ 输出 token
           Proxy → 用户
```

### 关键点

- **P 节点只传 KV cache 给 D 节点，不传 prompt**：D 节点不再 prefill，KV cache 直接从 P 节点搬到 D 节点的 HBM
- **D 节点正常流程不 prefill**：请求带 `do_remote_prefill=True` 标记，D 节点标为 `WAITING_FOR_REMOTE_KVS`，等 KV 到位后直接 decode
- **两种 KV 传输模式**：
  - **Pull（MooncakeConnector）**：D 节点主动从 P 节点拉 KV cache
  - **Push（MooncakeLayerwiseConnector）**：P 节点逐层把 KV cache 推到 D 节点

### 配置要素

| 参数 | P 节点 | D 节点 |
|------|--------|--------|
| `kv_role` | `kv_producer` | `kv_consumer` |
| `max_num_batched_tokens` | 大（如 32768） | 小（如 256，够 decode + MTP） |
| PP / CPP | 可开启 | 当前必须为 1 |

---

## 3. Pipeline Parallelism（流水线并行）

### 基本概念

PP 将模型的 Transformer hidden layers 分区到多个 pipeline stage，每个 stage 由一个 TP group 执行，相邻 stage 间传 intermediate activations。

```
Input tokens → PP stage 0 (earlier layers) → activations → PP stage 1 (later layers, LM head) → output
```

### vLLM-Ascend PP 优化点

#### 3.1 Dynamic Chunked Pipeline Parallel (CPP)

**问题**：PP + Chunked Prefill 时，由于 Self-Attention 的 O(n²) 复杂度，等长 chunk 随前缀增长耗时递增，导致 pipeline bubble。

**方案**：
- **启动期 Profiling**：64 次真实 forward 采样，最小二乘拟合二次模型 `f(l) = a·l² + b·l + c`
- **运行时预测**：给定当前前缀长度 L，解方程 `f(L+x) - f(L) = T` 求最优 chunk size x
- **在线校准**：运行期累积样本后更新模型参数

**收益**（A3 实测）：
- DeepSeek-V3.1-W8A8 128k：TTFT 22.5s vs static chunk 27.0s（降 17%）
- Qwen3-235B 256k：TTFT 53.5s vs 61.4s（降 13%）

#### 3.2 PP 异步调度

- `AsyncIntermediateTensors` + `comm_handles`：异步 P2P tensor 收发与计算重叠
- Zero Bubble 优化：消除 spec decode + async scheduling 的调度空隙
- 注意：P 节点 PP prefill 不建议开 async-scheduling（对 prefill 收益小且可能退化）

#### 3.3 PP + P/D 分离集成

- PP 通常在 P 节点启用，D 节点 PP=1
- `MooncakeConnectorV1` 要求 D 侧 PP=1
- connector `kv_connector_extra_config.prefill.pp_size` / `pp_layer_partition` 必须两边对齐

#### 3.4 Stage 平衡调优

- 自动分区：余数层从倒数第二 stage 向低 rank 分配
- 自定义分区：`VLLM_PP_LAYER_PARTITION="32,29"`
- 调优循环：测量每 stage peak memory + step latency → 每次只挪 1 层 → 重测

### PP 当前限制

| 限制 | 影响 |
|------|------|
| PP + PCP 不兼容 | 长上下文 prefill 不能同时用 PP 和 context parallel |
| MooncakeConnectorV1 要求 D 侧 PP=1 | D 节点不能扩容到多 stage |
| MoE over RoCE 不能 PP + DP 同时 | 限制大规模 MoE 部署拓扑 |

---

## 4. Layerwise（逐层传输）

### 概念

相对「整块 bulk」而言，把 KV cache 的存/取拆分到**模型每一层**的粒度做流水线，让传输和计算重叠。

### 对比

**非 layerwise（bulk 模式）**：
```
P: [forward 完整 prefill] → [一次性传整请求所有层 KV] → P 释放
D: [等整请求 KV 全到] → [forward]
    ↑ 长串行 stall，KV 越大等待越久
```

**Layerwise 模式**：
```
P layer i:  [算 layer i attention] → [layer i KV 立刻推 D]
P layer i+1:                        [算 layer i+1 attention] → [layer i+1 KV 推 D]
                                ↑ 传输与下一层计算重叠
D layer i:  [等 layer i KV 到] → [算 layer i attention]
                              ↑ 等待分摊到每层，不再集中阻塞
```

### 叠加 Chunked Prefill 的行为

每层 attention forward 中都会触发 `save_kv_layer` / `wait_for_layer_load`，与 chunked prefill 开启与否无关。叠加后变成「逐层 × 逐 chunk」的更细粒度 KV 流水：

| 模式 | forward 次数 | 每次 forward 内 | save 触发粒度 |
|------|-------------|----------------|--------------|
| 非 chunked | 1 次 | L 层各 save 一次 | L 次 |
| chunked (N chunks) | N 次 | L 层各 save 一次 | N × L 次 |

### D 节点不能在单个 chunk 后 decode

- 每个 chunk 只传「已完成的整 block」（增量），`floor` 取整，chunk 边界处不满的 block 不传
- D 只在 `chunk_finish=True`（所有 chunk 算完）+ 最后一层传输完毕才被通知可 decode
- layerwise 的价值是**预填充 D 的 KV cache**（背景接收），而非让 D 提前 decode

### 收益本质

把「整请求一次性传输」的串行 stall **分摊到每一层**，借助 `layerwise_prefetch_layers`（默认 1）让传输与下一层 attention 计算重叠，长 prompt 场景下显著降低 TTFT/TPOT。

### 三个相关 connector 的 layerwise 形态

| Connector | Layerwise 行为 |
|-----------|---------------|
| AscendStoreConnector | `use_layerwise: true` 后逐层 save/load 到 memcache pool |
| MooncakeLayerwiseConnector | P 节点有独立线程逐层 push KV 到 D |
| SFA PD RD2H | P 不主动 push，仅逐层发通知让 D 主动读（pull 模式但层粒度） |

---

## 5. Recompute Scheduler（重算调度器）

### 背景

PD 分离部署中 D 节点按 decode 调参（`max_num_batched_tokens` 很小，如 256），不适合 prefill。当 D 节点 HBM KV cache 不足需要抢占请求时，如何处理被抢占的请求？

### 三种配置方案对比

#### 方案 1：不开 recompute_scheduler（不推荐）

D 节点用标准 vLLM `Scheduler`，走上游 `_preempt_request`：

```python
self.kv_cache_manager.free(request)        # 释放 KV（丢弃）
request.num_computed_tokens = 0            # 重置
self.waiting.prepend_request(request)      # 放回 waiting 队列头
```

被抢占请求在 D 节点**本地从头 prefill**。这是灾难：64k prompt 在 `max_num_batched_tokens=256` 的 D 节点上需要 250 个 forward step，期间完全阻塞其他请求 decode。

#### 方案 2：开 recompute_scheduler，不开 offload（fallback recompute）

D 节点用 `RecomputeScheduler`，抢占时把请求**踢回 Proxy → P 节点重 prefill**：

```
D 节点抢占请求 → _finish_recomputed_request
  - 请求标记 FINISHED_ABORTED
  - KV 丢弃
  - 返回 stop_reason="recomputed" 给 Proxy
        ↓
Proxy 收到 "recomputed"
  - 拼接 prompt + 已生成 token 作为新 prompt
  - 重新选 P 节点 + D 节点
        ↓
新 P 节点完整 prefill（prompt + 已生成 token）
        ↓
KV 传给新 D 节点 → 继续 decode
```

代价很高：整个 prompt + 已生成 token 全量重算 prefill，加上网络传输开销。

#### 方案 3：开 recompute_scheduler + 开 recompute offload（推荐）

D 节点同时挂 `MooncakeConnectorV1`（正常 P→D 传输）和 `RecomputeCPUOffloadConnector`（抢占保护）。

**完整流程**：

```
D 节点 decode 中，HBM 满
   ↓
1. RecomputeScheduler 选 req A 抢占
   ↓
2. connector.update_state_before_preempt(reqA)
   scheduler 侧: 在 CPU pool 预留 block，建 HBM→CPU 映射
   ↓
3. 标准 _preempt_request(reqA)
   - free HBM block (归还 free pool)
   - reqA.status = PREEMPTED
   - reqA.num_computed_tokens = 0
   - waiting.prepend(reqA)        ← 仍在 D 节点本地
   ↓
4. worker.handle_preemptions()
   NPU store_stream: HBM → CPU pinned memory (同步 D2H)
   ↓ store 完成
5. state.ready = True
   ↓
6. D 节点用释放的 HBM 服务其他 decode 请求
   ↓ (HBM 又有空了)
7. scheduler 从 waiting 取 reqA 重新调度
   ↓
8. connector.get_num_new_matched_tokens(reqA, 0)
   返回 (state.num_computed_tokens, True)
   ↓
9. allocate 新 HBM block 给 reqA
   connector.update_state_after_alloc() 建 CPU→新 HBM 映射
   ↓
10. forward 前 worker.start_load_kv()
    NPU load_stream: CPU → HBM (异步 H2D)
    ↓
11. 每层 attention 前 wait_for_layer_load()
    ↓
12. KV 全在 HBM → 从 reqA 抢占前位置继续 decode
```

**核心收益**：整个恢复路径完全在 D 节点本地完成，不回 P 节点，不走 Proxy：
- 恢复耗时：几十 ms~秒级（vs 方案 2 的几十秒）
- P 节点不参与，Proxy 不参与
- KV 在 D 节点 HBM ↔ CPU 之间搬运，用 NPU store/load stream 可与其他请求 decode 重叠

### 三方案对比

| 维度 | 不开 recompute | 开 recompute（无 offload） | 开 recompute + offload |
|------|---------------|--------------------------|----------------------|
| 抢占后请求去哪 | D 本地 waiting | 踢回 Proxy → 新 P | D 本地 waiting |
| D 节点 KV 处理 | free（丢弃） | 丢弃 | D2H 暂存 CPU |
| 恢复时 KV 来源 | D 本地重 prefill | P 节点重 prefill + 网络传 | CPU H2D 拉回 |
| 恢复耗时（64k） | 几十秒~分钟 | 几十秒 | 几十 ms~秒 |
| P 节点参与 | 不参与 | 重 prefill + KV 传输 | 不参与 |
| Proxy 参与 | 不参与 | 重新路由 | 不参与 |

---

## 6. KV Offloading（KV 卸载）

### 定义

把 KV cache 从昂贵的 NPU HBM 搬到更便宜的存储介质（CPU DRAM / 外部存储池），按需再搬回。所有 offloading 的技术手段相同（KV 搬出 HBM），但**目的不同**。

### 四种 KV Offloading

#### 6.1 KV Cache CPU Offload（扩展 prefix cache）

```
HBM KV pool (小) ←→ CPU DRAM block pool (大, LRU)
```

- **目的**：扩大 prefix cache 复用窗口（两级 LRU：HBM 是 L1，CPU 是 L2）
- **触发**：HBM 满时 inactive block 异步 D2H 到 CPU；命中时 H2D 拉回
- **配置**：`OffloadingConnector` + `NPUOffloadingSpec`
- **命中识别**：通过 block hash 在 CPU `CPUOffloadingManager` 中查找（`lookup` 方法），命中后 `prepare_load` 占住 block（ref_cnt++），异步 H2D 传输

#### 6.2 Recompute CPU Offload（抢占保护）

```
D 侧 HBM block (被抢占) → CPU DRAM (暂存) → D 侧 HBM (恢复时拉回)
```

- **目的**：被抢占请求的 KV 暂存 CPU，避免回 P 节点重 prefill
- **配置**：`RecomputeCPUOffloadConnector`，仅 PD 分离 D 节点
- **与 prefix cache 无关**：这是单请求的 KV 保护机制，不是跨请求复用

#### 6.3 Layerwise KV Cache Offload（Prefill 省 HBM）

```
NPU: layer0 buf | layer1 buf | layer2 buf (3 个共享 buffer 复用 61 层)
              ↕ 异步 save/load
Host pool: layer0 KV | layer1 KV | ... | layer60 KV
```

- **目的**：长 prompt prefill 时 HBM 装不下所有层 KV，用少量 buffer 轮转
- **机制**：多个逻辑层映射到少数物理 NPU buffer；算完一层 → 存 host → buffer 复用给下一层
- **配置**：`AscendStoreConnector` + `use_layerwise: true` + `layerwise_num_shared_buffers: 3`

#### 6.4 Sparse KV Cache Offload（Decode 省 HBM）

```
NPU D 侧:
  indexer cache (选 token) + top-k hot buffer (最近用的 KV)
Host D 侧:
  全量 KV cache
每步 decode: 命中 top-k 直接用 → miss 部分从 host 拉
```

- **目的**：D 侧 HBM 装不下长序列全量 KV，用稀疏注意力 + host offload 突破容量
- **配置**：`SfaRemoteD2HConnector` + `sparse_kv_offload_config.enabled: true`
- **适用**：GLM-5.2 / DeepSeek-V3.2 等支持稀疏注意力的模型

### 四种 offload 对比

| 名称 | 作用阶段 | 介质 | 解决什么问题 | 与 prefix cache 关系 |
|------|---------|------|------------|---------------------|
| KV Cache CPU Offload | 单机全程 | CPU DRAM | HBM 不够，延长 prefix 复用窗口 | ✅ 直接配套 |
| Recompute CPU Offload | PD 分离 D 侧抢占时 | CPU DRAM | 避免回 P 重 prefill | ❌ 无关 |
| Layerwise KV Cache Offload | Prefill 阶段 | Host pool | 突破 P 侧 HBM 容量 | ❌ 无关 |
| Sparse KV Cache Offload | Decode 阶段 | Host DRAM | 突破 D 侧 HBM 容量 | ❌ 无关 |

---

## 7. KV Cache Pool（KV 缓存池化）

### 定义

把 KV cache 存到一个**集群共享的外部存储池**，多个 vLLM 实例都能查/取，实现跨实例 prefix 复用。

```
vLLM 实例 1 (HBM)  ─┐
vLLM 实例 2 (HBM)  ─┼──→ Mooncake Store / Memcache / Yuanrong (集群共享池)
vLLM 实例 3 (HBM)  ─┘
```

### 与 KV Offloading 的关系

| 维度 | KV Offloading | KV Cache 池化 |
|------|--------------|--------------|
| 目的 | 给单机 HBM 扩容（容量问题） | 跨节点/跨实例共享 prefix（复用问题） |
| 共享范围 | 本机自用 | 多 vLLM 实例共享 |
| 介质 | 本机 CPU DRAM | 集群级存储 |
| 典型场景 | 单机长序列、PD 抢占保护 | 多副本 prefix cache 共享 |

**技术手段相同（KV 搬出 HBM），但视角不同**：offloading 是单机视角的「KV 搬到 CPU」，池化是集群视角的「KV 存到共享存储」。

### 三种 backend

| Backend | 介质 | 特点 |
|---------|------|------|
| mooncake | DRAM (+ SSD offload) | Mooncake Store 独立服务，P2P RDMA 传输，lease TTL 管理 |
| memcache | DRAM | 华为 Memcache Hybrid，支持 layerwise 逐层存取，需 MemFabric 依赖 |
| yuanrong | 分布式存储 | 跨更大规模集群 |

### 池化 + PD 分离协同

实际部署中池化常和 P→D 传输 connector 一起用（`MultiConnector`）：

```
P 节点:
  MooncakeConnectorV1     ← P→D 直接传本次请求 KV
  AscendStoreConnector    ← 把 KV 存到共享池（供其他实例未来复用）

D 节点:
  MooncakeConnectorV1     ← 从 P 拉本次请求 KV
  (可选) AscendStoreConnector ← 从池里拉其他实例已算过的 prefix
```

D 节点收到请求时按级别查找：**本地 HBM prefix cache → KV pool（AscendStoreConnector）→ 从 P 拉（MooncakeConnectorV1）**，逐级减少传输量。

---

## 8. MooncakeConnector（P2P KV 传输）

### 定义

`MooncakeConnectorV1` 是 vllm-ascend 中用于 **PD 分离部署时 P 节点 → D 节点 KV cache 直接 P2P 传输**的 connector，基于开源项目 [Mooncake](https://github.com/kvcache-ai/Mooncake) 的 `TransferEngine`。

### 核心机制

- **P2P RDMA 传输**：P 节点 NPU HBM → RDMA 网络 → D 节点 NPU HBM（不经中间存储）
- **Session-based**：每个请求建立一个 session，P 端写、D 端读
- **Pull 模式**：D 节点主动从 P 节点拉 KV

### 工作流程（Pull 模式）

```
1. 请求到 Proxy → 选 P 节点 + D 节点
2. P 节点 prefill 完 → KV 留本地 → 返回 Proxy
3. Proxy 转发到 D 节点（带 do_remote_prefill=True）
4. D 节点标记 WAITING_FOR_REMOTE_KVS → 预分配 HBM → 从 P 拉 KV
5. KV 传完 → 通知 P 释放 → D 节点 decode 第一个 token
```

### 三个核心类

| 类 | 职责 |
|----|------|
| MooncakeConnector | base 类，提供核心接口 |
| MooncakeConnectorScheduler | scheduler 侧，管理 KV 传输请求和完成状态 |
| MooncakeConnectorWorker | worker 侧，管理 KV cache 注册和实际传输 |

### 关键能力

| 能力 | 说明 |
|------|------|
| NPU 直传 | `kv_buffer_device: "npu"`，KV 直接在 NPU HBM 间传 |
| PP 支持 | 配 `prefill.pp_size` / `pp_layer_partition` |
| TP 不对称 | P TP ≥ D TP 且整除，自动 resharding |
| MLA / GQA | 都支持，有 `pd_head_ratio` 处理 head 分布差异 |
| prefix cache 协同 | D 节点先查本地 HBM，剩余才从 P 拉 |
| 失败处理 | `kv_load_failure_policy: recompute` 或 `fail` |

### 与其他传输 connector 对比

| Connector | 传输模式 | 介质 | 何时用 |
|-----------|---------|------|--------|
| MooncakeConnectorV1 | D 主动 pull | Mooncake P2P RDMA | 标准 PD 分离 |
| MooncakeLayerwiseConnector | P 逐层 push | Mooncake P2P + 独立线程 | 长 prompt PD 分离 |
| SfaRemoteD2HConnector | D 主动 pull（层粒度） | MemFabric | 稀疏注意力模型 |

### 与 MooncakeStoreConnector 的区别

| 名称 | 作用 | 何时用 |
|------|------|--------|
| MooncakeConnectorV1 | P→D P2P 直接传 KV（不存 pool） | PD 分离传输通道 |
| AscendStoreConnector (mooncake backend) | KV Cache Pool（跨请求/跨节点 prefix 复用） | 多节点共享前缀缓存 |

---

## 9. 特性对比总览

### KV 搬出 HBM 的所有形态

```
            ┌── KV Cache CPU Offload        ← 单机扩 prefix cache (OffloadingConnector)
            │
            ├── Recompute CPU Offload        ← 单机抢占保护 (RecomputeCPUOffloadConnector)
            │
KV offloading ── Layerwise KV Cache Offload ← 单机 P 侧容量扩展 (AscendStoreConnector + layerwise)
            │
            └── Sparse KV Cache Offload     ← 单机 D 侧容量扩展 (SfaRemoteD2HConnector)


            ┌── Mooncake backend  ← 跨节点 DRAM 池 (AscendStoreConnector)
KV 池化 ─────┼── Memcache backend  ← 跨节点 DRAM 池 + layerwise (AscendStoreConnector)
            └── Yuanrong backend  ← 跨节点分布式存储 (AscendStoreConnector)
```

### 传输优化对比

| 特性 | 优化对象 | 机制 |
|------|---------|------|
| CPP | P 节点内 PP stage bubble | 动态 chunk 预测，等化各 chunk 耗时 |
| Layerwise (传输) | P→D KV 传输串行 stall | 逐层 push/pull，与计算重叠 |
| Async Scheduling | PP stage 间通信 | 异步 P2P tensor 收发 |
| FlashComm | TP 内 collective 通信 | Allreduce 分解为 RS + AG |

### 抢占恢复路径对比

| 配置 | 抢占后行为 | 恢复耗时（64k prompt） | P 节点参与 | Proxy 参与 |
|------|-----------|---------------------|-----------|-----------|
| 不开 recompute | D 本地重 prefill | 几十秒~分钟 | 否 | 否 |
| 开 recompute，无 offload | 回 P 节点重 prefill | 几十秒 | 是 | 是 |
| 开 recompute + offload | CPU 暂存 → 拉回 | 几十 ms~秒 | 否 | 否 |

---

## 参考文档

- [Pipeline Parallelism](docs/source/user_guide/feature_guide/pipeline_parallel.md)
- [Dynamic Chunked Pipeline Parallel](docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md)
- [Recompute CPU Offload](docs/source/user_guide/feature_guide/recompute_cpu_offload.md)
- [KV Cache CPU Offload](docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md)
- [Layerwise KV Pool](docs/source/user_guide/feature_guide/layerwise_kv_pool.md)
- [Layerwise and Sparse KV Cache Offloading](docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)
- [KV Cache Pool](docs/source/user_guide/feature_guide/kv_pool.md)
- [Disaggregated Prefill Design](docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)
- [CPP Design Document](docs/source/developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md)
- [KV Cache Pool Guide](docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md)
