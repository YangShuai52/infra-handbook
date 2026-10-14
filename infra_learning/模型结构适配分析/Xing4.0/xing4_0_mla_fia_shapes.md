# Xing4.0-29B-A4B MLA FIA 入参与 Shape

## 范围与约定

本文根据当前仓库 `vllm_ascend/attention/mla_v1.py` 调用代码整理，仅考虑 TP，关闭 PCP、DCP、MTP、量化及 MLA prolog。描述现有实现与 310P 的目标输入需求，不表示这些接口已在 310P 验证可用。

## 310P 目标接口的 NZ 格式要求

本适配方案约定：310P FIA 的 Q/K/V 类浮点输入均采用 **NZ 物理格式**，包括独立的 `query_rope`、`key_rope`，以及 decode 的 latent KV cache。主干及缓存 dtype 采用 FP16。下文各阶段表格仍保留当前 MLA 代码的 dtype、布局和调用方式，作为迁移参考；其中 BF16、ND 描述不代表 310P 的目标格式。

| 阶段 | 输入 | 逻辑 shape | 310P 目标 dtype / 物理格式 |
| --- | --- | --- | --- |
| 当前 prefill | query / key / value | 各 [T,H,128] | FP16 / NZ |
| 当前 prefill | query_rope / key_rope | 各 [T,H,64] | FP16 / NZ |
| 历史 chunk | query / query_rope | [T,H,128] / [T,H,64] | FP16 / NZ |
| 历史 chunk | key / value / key_rope | [C,H,128] / [C,H,128] / [C,H,64] | FP16 / NZ |
| decode | query / query_rope | BNSD 逻辑表示 [B,H,1,512] / [B,H,1,64] | FP16 / NZ |
| decode | key / value | 同一份 latent cache，每个 block 含 Bs×1×512 个元素 | FP16 / NZ |
| decode | key_rope | RoPE cache，每个 block 含 Bs×1×64 个元素 | FP16 / NZ |

`TND`、`BNSD`、`BSND` 表示逻辑轴顺序；NZ 表示物理存储格式，两者不是互斥选项。上述 shape 用于表达计算维度，不是 NZ 缓冲的最终物理 shape。NZ 的分块方式、padding、stride、描述符和分配字节数必须按目标 310P FIA 接口确定，不应直接套用现有 910B MLA 的 NZ view。

Q/K/V 投影、RoPE 和历史 KV 展开后的结果，进入 FIA 前必须已经满足目标 NZ 格式；cache 写入、历史 gather 和 FIA 读取也必须使用一致的 NZ 地址规则。`.contiguous()` 只保证连续性，`view` 只改变张量解释，两者均不能替代 ND 到 NZ 的数据重排或格式转换。

此约定不扩展到整数 metadata、mask 或 LSE。`block_table`、长度等按接口各自要求传入；外部 LSE 合并输入继续按本文记录的 FP32 shape 描述，合并算子的物理格式需独立确认。

模型参数：32 attention heads，KV latent rank 512，Q NoPE head dim 128，RoPE head dim 64，V head dim 128。来源：[官方 config.json](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/blob/main/config.json)。

| 符号 | 含义 |
| --- | --- |
| P | TP 数，本文假设整除 32 |
| H | 每卡 query heads，32/P；本文选择无 head padding 的分支 |
| T | 本轮 prefill 的总 Q token 数，多请求按 T 轴拼接 |
| C | 当前被加载的历史 KV chunk 的总 token 数 |
| R | prefill 请求数 |
| B | decode 请求数，每请求一个 Q token，无图模式 padding |
| Nb | 本层缓存 block 数 |
| Bs | block size；当前 MLA backend 声明 128 |
| M | block table 每请求的最大 block 数 |

TND 中 T=总 token 数，N=head 数，D=head dim。BNSD/BSND 中 B=batch，S=每请求 Q 序列长度，decode 基线 S=1。NBSD 是 head 在前的输出布局。

注意：模型 config 的 `num_key_value_heads=32` 不代表压缩 cache 存储 32 份 latent。此处标准 MLA 压缩 KV head 数为 1，prefill 展开 K/V 后每卡才具有 H 个 heads。

## 1. Prefill：当前 Chunk

**算子名：`torch_npu.npu_fused_infer_attention_score`。**

代码：[`_forward_prefill`](vllm_ascend/attention/mla_v1.py#L1367)，调用位置约第 1424 行。以下为 `head_padding == 0` 的独立 RoPE 输入分支。

| 入参 | Shape / 值 | 当前 dtype / 格式 | 布局及含义 |
| --- | --- | --- | --- |
| query（第一个位置参数） | [T,H,128] | BF16，普通 ND 张量 | TND，Q NoPE |
| key（第二个位置参数） | [T,H,128] | BF16，contiguous ND | TND，当前 K NoPE，已由 latent 展开 |
| value（第三个位置参数） | [T,H,128] | BF16，contiguous ND | TND，当前 V，已由 latent 展开 |
| query_rope | [T,H,64] | BF16 | TND，旋转后的 Q RoPE；可能是切片 view |
| key_rope | [T,H,64] | BF16，contiguous ND | TND，旋转后的 K RoPE，从共享 head 广播 |
| num_heads | H | Python int | 每卡 Q heads |
| num_key_value_heads | H | Python int | 展开后的 K/V heads |
| input_layout | "TND" | Python str | 多请求沿 T 轴拼接 |
| atten_mask | [2048,2048] | INT8，ND，NPU | metadata 提供的压缩 causal mask，不是动态 [T,T] 矩阵 |
| sparse_mode | 3 | Python int | causal 压缩 mask 模式 |
| scale | self.scale | Python float | 上游模型传入的 attention scale，需保留 YaRN 修正，不能直接替换成 1/sqrt(192) |
| antiquant_mode | 0 | Python int | 非量化路径 |
| antiquant_scale | None | 无 | 非量化路径 |
| block_table | None | 无 | 本次使用展开后的当前 K/V，不直接读取分页 cache |
| block_size | 0 | Python int | 本次不走分页读取 |
| softmax_lse_flag | True | Python bool | 请求输出 LSE |
| actual_seq_lengths | 长度 R；各请求 Q 长度的累计和 | Python 整数列表 | TND 请求边界，最后一个值为 T |
| actual_seq_lengths_kv | actual_seq_lengths 的副本 | Python 整数列表 | 当前 chunk K/V 请求边界 |

| 输出 | Shape | 说明 |
| --- | --- | --- |
| attn_output | [T,H,128] | 当前 chunk attention 输出 |
| attn_lse | 每个 Q/head 一个值；代码兼容 [H,T] 并转为 [T,H,1] | 外部合并前转 FP32；具体返回 rank 以目标算子版本为准 |

当前代码将非 BF16 的五份 Q/K/V/RoPE 输入转换为 BF16。310P 目标需确认 FP16 对应能力，不能照搬该转换。

## 2. Chunked Prefill：每个历史 KV Chunk

**算子名：`torch_npu.npu_fused_infer_attention_score`。**

代码：[`_compute_prefill_context`](vllm_ascend/attention/mla_v1.py#L1243)，调用位置约第 1353 行。

历史压缩 KV 先通过 `DeviceOperator.kv_cache_load` 读取，再通过 `kv_b_proj` 展开 K/V；同一批当前 Q 对每个历史块分别计算。

| 入参 | Shape / 值 | dtype / 格式 | 布局及含义 |
| --- | --- | --- | --- |
| query | [T,H,128] | 当前路径 BF16 | TND，与当前 chunk 使用相同 Q NoPE |
| key | [C,H,128] | 由 cache dtype 与 kv_b_proj 路径决定，调用前 contiguous | TND，历史 K NoPE |
| value | [C,H,128] | 由 cache dtype 与 kv_b_proj 路径决定，调用前 contiguous | TND，历史 V |
| query_rope | [T,H,64] | 当前路径 BF16 | TND，与当前 chunk 使用相同 Q RoPE |
| key_rope | [C,H,64] | 缓冲按 Q RoPE dtype 分配，调用前 contiguous | TND，历史 K RoPE |
| num_heads / num_key_value_heads | H / H | Python int | 展开后的 heads |
| input_layout | "TND" | Python str | Q 与历史 KV 分别沿 T/C 轴拼接 |
| atten_mask | None | 无 | 历史 KV 均早于当前 Q，无需 causal mask |
| sparse_mode | 0 | Python int | noncausal |
| scale | self.scale | Python float | 与当前 chunk 相同 |
| antiquant_mode / antiquant_scale | 0 / None | int / 无 | 非量化路径 |
| softmax_lse_flag | True | Python bool | 输出 LSE |
| actual_seq_lengths | 长度 R 的 Q 累计长度列表 | Python 整数列表 | 最后一个值 T |
| actual_seq_lengths_kv | 长度 R 的当前历史块 KV 累计长度列表 | Python 整数列表 | 最后一个值 C；部分请求本块长度可为 0 |
| block_table / block_size | 本调用未显式传入 | 接口默认值 | 分页读取已在 FIA 调用前完成 |

输出：`chunk_out` 为 [T,H,128]；`chunk_lse` 每个 Q/head 一个值。两者转换 FP32 后加入合并列表。

历史路径没有统一将展开 K/V 转为 BF16 的独立步骤。因此 FP16 主干迁移时须同时检查 Q/K/V dtype 一致性，不能只修改当前 chunk 的转换。

## 3. Decode：非 NZ Cache 基线

**算子名：`torch_npu.npu_fused_infer_attention_score_v2`。**

代码：[`_forward_decode`](vllm_ascend/attention/mla_v1.py#L1642)，eager 调用位置约第 1880 行。图模式使用同算子的 `.out` 变体，并额外传 workspace 与输出缓冲。

| 入参 | Shape / 值 | dtype / 格式 | 布局及含义 |
| --- | --- | --- | --- |
| query | [B,H,1,512] | 主干浮点 dtype；310P 目标 FP16 | BNSD，Q NoPE 已投影到 latent 空间 |
| key | [Nb,1,Bs,512] | cache dtype，非 NZ 路径 | FIA 前实际 view 为 block/head/slot/dim；压缩 latent cache |
| value | 与 key 同一 tensor | 同 key | 用 latent 同时作为 K 和 V |
| query_rope | [B,H,1,64] | 主干浮点 dtype | BNSD，旋转后的 Q RoPE |
| key_rope | [Nb,1,Bs,64] | cache dtype，非 NZ 路径 | FIA 前实际 view；旋转后的 K RoPE cache |
| num_query_heads | H | Python int | 无 head padding 基线 |
| num_key_value_heads | 1 | Python int | 压缩 latent KV 共享 head |
| input_layout | "BNSD_NBSD" | Python str | Q 输入 BNSD，输出 NBSD |
| atten_mask | None | 无 | 单 Q decode |
| sparse_mode | 0 | Python int | 基础 decode 模式 |
| softmax_scale | self.scale | Python float | 注意此处参数名不同于 prefill 的 scale |
| block_table | [B,M] | metadata 的整数 NPU tensor，通常 INT32 | 每请求的逻辑 block 到物理 block 映射；具体 dtype 以 metadata 为准 |
| block_size | Bs，当前 backend 声明 128 | Python int | cache 分页大小 |
| actual_seq_qlen | None | 无 | 基础单 token decode 分支 |
| actual_seq_kvlen | 长度 B，各请求完整可见 KV 长度 | Python 整数列表 | 包含已写入的当前 token |

输出：attention latent output 为 [H,B,1,512]，随后 `W_UV` 投影成 [B,H,128]。eager 调用的第二个返回值被丢弃；不做外部 LSE 合并。

**布局纠正：** cache 的统一逻辑分配 shape 可为 [Nb,Bs,1,D]，但 `_forward_decode` 在传给 FIA 前明确 view 为 [Nb,1,Bs,D]。上表列的是算子实际入参 shape，不是分配接口 shape。不能将两者当作相同的地址布局直接互换。

## 4. Decode：启用 KV NZ 的分支

仍调用 `npu_fused_infer_attention_score_v2`，其他基础参数同上一节。

| 入参/输出 | Shape | 布局 |
| --- | --- | --- |
| query | [B,1,H,512] | BSND |
| query_rope | [B,1,H,64] | BSND |
| key / value | [Nb,1,32,Bs,16] | 源码以 D/16 分块的 NZ cache view；D=512 |
| key_rope | [Nb,1,4,Bs,16] | D=64，按 16 分块 |
| input_layout | "BSND_NBSD" | Q 输入 BSND，输出 NBSD |
| attention output | [H,B,1,512] | NBSD |

这是代码中的 cache 内存组织，不代表所有张量都可用通用 format_cast 得到。写 cache 算子、分配/绑定与 FIA 必须使用一致的物理地址规则。

## 5. LSE 合并接口

**算子名：`torch_npu.npu_attention_update`。**

代码：`mla_v1.py` 第 1364 行。仅历史 KV 被分块计算时调用；无历史 prefill 与基础 decode 不调用。

| 入参/输出 | Shape / 值 | dtype / 布局 |
| --- | --- | --- |
| 第一个位置参数：tuple(lse_list) | K+1 个 [T*H] tensor，K 为历史块数 | FP32，按 token/head flatten |
| 第二个位置参数：tuple(out_list) | K+1 个 [T*H,128] tensor | FP32，与 LSE flatten 顺序一致 |
| 第三个位置参数 | 0 | 原代码传值；本文不扩展解释其接口语义 |
| output_final | [T*H,128] | 恢复 [T,H,128]，随后恢复原输出 dtype |
| 第二个返回值 | 当前调用丢弃 | shape 不由本调用确认 |

列表第 0 项为当前 chunk 的 causal attention 结果，其后是历史块的 noncausal attention 结果。融合关系为 `L=logsumexp(L_i)`、`O=sum(exp(L_i-L)*O_i)`，是 TP 卡内融合。

## 6. TP=2 示例

| 阶段 | 关键入参 shape |
| --- | --- |
| 当前 prefill | Q/K/V [T,16,128]，Q/K RoPE [T,16,64] |
| 历史 chunk | Q [T,16,128]，K/V [C,16,128]，Q RoPE [T,16,64]，K RoPE [C,16,64] |
| Decode 非 NZ | Q [B,16,1,512]，Q RoPE [B,16,1,64]，latent cache [Nb,1,128,512]，RoPE cache [Nb,1,128,64] |
| Decode NZ | Q [B,1,16,512]，Q RoPE [B,1,16,64]，latent cache [Nb,1,32,128,16]，RoPE cache [Nb,1,4,128,16] |
| LSE 合并每组输入 | LSE [16T]，output [16T,128]，均 FP32 |

本表是从现有调用代码推导的接口需求，未进行 310P 单算子测试。若目标实现改变 padding、布局、block size 或算子版本，应同步更新本表。

## 7. MTP1、MTP2、MTP3：主模型验证与草稿生成

本节扩展前文关闭 MTP 的基线。这里 MTP1/2/3 分别指 `num_speculative_tokens=1/2/3`，即每轮提出 K=1/2/3 个候选 token；不是模型中 MTP 层数。Xing config 的 `num_nextn_predict_layers=1` 表示检查点包含一层 MTP，不保证目标 vLLM 已支持复用该层进行 K=2/3 的多步生成，仍需确认 Xing 草稿模型实现与数值语义。

必须分清两个计算阶段：主模型一次验证多个 token；草稿模型提出候选，可能先处理一批主模型输出，再逐步生成后续候选。不能把所有草稿调用一律当作单 Q decode。

### 7.1 主模型验证：FIA v2 的多 Q、分页 MLA 路径

当前 `_forward_decode` 在启用 speculative_config 时，针对 SpecDecoding、ChunkedPrefill、DecodeOnly 等状态切换为 `input_layout="TND_NTD"`，而不是基线的 `BNSD_NBSD`。Q token 展平，使用累计 Q 长度区分请求。输出 head 在前，可直接接后续 V up projection。

典型完整验证批次每请求 Q 长度为 K+1，包含一个基准输入 token 与 K 个候选输入 token。实际批次可能存在不同 Q 长度、提前结束和 padding，应以 metadata 为准。

| 项目 | MTP1：K=1 | MTP2：K=2 | MTP3：K=3 |
| --- | --- | --- | --- |
| 每请求典型 Q 长度 | 2 | 3 | 4 |
| 总有效 Q 数 Tv | 2B | 3B | 4B |
| query 逻辑 shape | [2B,H,512] | [3B,H,512] | [4B,H,512] |
| query_rope 逻辑 shape | [2B,H,64] | [3B,H,64] | [4B,H,64] |
| actual_seq_qlen，均匀批次示例 | [2,4,...,2B] | [3,6,...,3B] | [4,8,...,4B] |
| latent attention output | [H,2B,512] | [H,3B,512] | [H,4B,512] |
| V up projection 后 | [2B,H*128] | [3B,H*128] | [4B,H*128] |
| 本轮有效 slot 数 | 2B | 3B | 4B |

算子：`torch_npu.npu_fused_infer_attention_score_v2`；图模式为其 `.out` 变体。表中所有 Q/RoPE/cache 浮点输入按 310P 目标要求采用 FP16/NZ，TND/NTD 是逻辑轴顺序，不表示物理 ND 格式。

| 其余入参 | MTP 验证值 / shape | 说明 |
| --- | --- | --- |
| key / value | 同一 latent cache；非 NZ 参考 view [Nb,1,Bs,512]，现有 MLA NZ参考 view [Nb,1,32,Bs,16] | 310P 的最终 NZ 地址规则由目标接口定义 |
| key_rope | 非 NZ 参考 view [Nb,1,Bs,64]，现有 MLA NZ参考 view [Nb,1,4,Bs,16] | 与 latent cache 共用请求位置和 block table |
| num_query_heads / num_key_value_heads | H / 1 | 本节无 head padding |
| input_layout | "TND_NTD" | 输入 [Tv,H,D]，输出 [H,Tv,512] |
| sparse_mode / atten_mask | causal 分支为 3 / [2048,2048] INT8 mask | 每个验证 Q 只能看历史和本请求中不晚于自己的 token |
| actual_seq_qlen | 长度为请求数的累计 Q 长度列表 | 不是单独的 [2,2,...] 或 [3,3,...] 长度列表 |
| actual_seq_kvlen | 每请求验证时完整 KV 长度的列表 | 包括本轮已写入的验证输入；与 Q 长度联合确定 causal 对齐 |
| block_table | [B,M]，图模式可扩展为 padded 请求行数 | 一行对应一个请求，不是一行对应一个验证 token |
| block_size / softmax_scale | Bs / self.scale | 与基线相同 |

TP=2 时 H=16：MTP1 的 Q 为 [2B,16,512]，RoPE 为 [2B,16,64]，latent 输出 [16,2B,512]；MTP2/3 将 2B 替换为 3B/4B。

必须支持多 Q 的 causal 分页 attention；普通 IFA 的单 Q 接口不能直接覆盖此验证路径。mask 和 seq length 语义不能因为输入采用 NZ 而省略。

代码位置：

- `vllm_ascend/attention/mla_v1.py:1695`：speculative 分支，选择 TND_NTD、causal mask 和 actual_seq_lengths_q。
- `vllm_ascend/attention/mla_v1.py:1794`：构造 FIA v2 参数，将累计 Q 长度传给 actual_seq_qlen。
- `vllm_ascend/attention/mla_v1.py:1880`：eager FIA v2 调用。
- `vllm_ascend/attention/mla_v1.py:677`：build_decode_metadata，SpecDecoding 中请求数不等于 token 数。
- `vllm_ascend/attention/mla_v1.py:292`：decode_threshold=K+1，当前代码检查每请求阈值不超过 16。

### 7.2 草稿侧：优先 MTP1，再验证多步 MTP2/3

MTP1 不需要生成第二、第三个候选的后续单 token 草稿步，但首次草稿调用仍可能处理多 token 的主模型输出/上下文；其 Q 长度由草稿 metadata 决定，不应固定写成 B。

310P MRv2 已有通用草稿循环，K>1 时执行后续 draft decode 步。但它是框架参考路径，不表示 Xing MLA 已适配；当前 310P runner 仍拒绝 MLA，需先完成模型及 attention 接入。

| 草稿阶段 | MTP1 | MTP2 | MTP3 | FIA 输入需求 |
| --- | --- | --- | --- | --- |
| 首次草稿调用 | 提出首个候选 | 提出首个候选 | 提出首个候选 | Q 总数与每请求长度取自草稿 metadata；可为多 token |
| 后续候选生成 | 无后续步 | 通常再执行 1 个自回归步 | 通常再执行 2 个自回归步 | 活跃请求一 token/步：Q [Bd,H,512]、RoPE [Bd,H,64]；Bd 为本步活跃请求数 |
| 后续步 latent 输出 | 不适用 | [H,Bd,512] | 每步 [H,Bd,512] | 若使用当前 MLA speculative 分支，为 TND_NTD；不能自动套 BNSD |

首次草稿调用若被 metadata 分类为 prefill，则走前文 prefill FIA和可能的历史 LSE 合并；若分类为 speculative decode，则走上述 FIA v2 多 Q 路径。实际 Xing 接入必须记录首步 Q 长度、状态与调用分支。

代码位置：

- `vllm_ascend/_310p/worker/v2/spec_decode/mtp_speculator.py:75`：load_draft_model，加载具体草稿模型。
- `vllm_ascend/_310p/worker/v2/spec_decode/mtp_speculator.py:167`：草稿 attention metadata 构建。
- `vllm_ascend/_310p/worker/v2/spec_decode/mtp_speculator.py:255`：首次草稿生成包装。
- `vllm_ascend/_310p/worker/v2/spec_decode/mtp_speculator.py:288`：后续多步 decode；约第 310 行循环执行后续步骤。

### 7.3 Cache 写入、拒绝候选与图模式

主模型与草稿 attention 层应按各自 cache 绑定执行写入，不能假定共享同一层的 KV 数据。主模型验证时每个 Q token 有自己的 position/slot，KV写入的 token 轴由 B 扩展到 Tv；每 token 仍仅写 latent512与RoPE64，不新增候选轴或额外 KV heads。

拒绝候选后的物理位置可能留有临时 KV。正确性依赖调度器/草稿逻辑调整有效长度、下一轮 position与slot，并在必要时覆盖这些位置；不要求每轮清零整个 cache。应测试拒绝后残留数据不会再次进入可见 KV，尤其跨 block 边界时。

图模式会 padding 请求数和 token 数，加入无效 slot，并补齐累计 Q 长度及 block table。表中 2B/3B/4B 为无 padding、均匀有效批次示例，不是所有实际 launch shape。目标 310P 写 cache kernel 必须跳过无效 slot。

### 7.4 MTP 是否增加外部 LSE 合并

MTP 主模型验证的 FIA v2 一次读取该批 Q 对应的完整分页 KV，仍没有外部 `npu_attention_update`；多 Q 的 causal 归一化由 FIA 内部完成。MTP1/2/3 均如此。

如果草稿首步或混合批次中的 prefill 使用历史 KV 分块路径，仍按前文执行外部 LSE 合并。此时令 T 为该 prefill 分支实际 Q token 总数，每组 LSE [T*H]、output [T*H,128]，均 FP32；不能将这些 shape 替换为 decode latent输出宽512。

### 7.5 最小验证矩阵

优先 MTP1：均匀 Q 长度2、变长批次、接受/拒绝候选、分页边界、无效slot、eager/图模式输出一致性。再扩展 MTP2/3：Q 长度3/4、连续草稿步、部分接受后 position和cache复用、首次草稿多 Q、与prefill混合。所有验证均需核对 FP16/NZ 与参考模型精度。
