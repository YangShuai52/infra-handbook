# LLM 基础与 AI Infra 知识体系分类研究

> 研究日期：2026-09-04  
> 目标：为 `infra-handbook` 设计一套可长期扩展、适合开源协作、同时服务学习与工程实践的知识目录。  
> 证据原则：本稿只采用论文、标准/规范、项目官方文档与官方源代码仓库；不以博客转载、培训文章或二手综述作为分类依据。

## 1. 结论摘要

建议把仓库从当前“面试知识卡片集合”升级为一部沿着 **模型原理 → 模型生命周期 → AI 基础设施 → 应用系统 → 实践验证** 展开的工程手册。

最合适的一级结构不是按热门项目名称划分，也不是把所有内容平铺为几十个专题，而是五个稳定分区：

1. **LLM Foundations**：回答模型为何有效、一次前向/反向到底发生了什么。
2. **Model Lifecycle**：回答数据如何变成基座模型，模型又如何经过后训练、对齐和评估成为可用模型。
3. **AI Infrastructure**：回答计算、通信、存储、训练平台、推理运行时和在线服务如何共同承载模型。
4. **LLM Systems**：回答 RAG、Agent、安全与治理如何把模型接入真实环境。
5. **Labs & Reference**：用可运行实验、估算器、故障手册和术语表验证前四部分，而不是另建一套重复的“手撕代码知识树”。

推荐主目录共 20 个模块。它比当前 10 模块更细，但边界仍足够稳定：算法创新可以落入既有模块，框架更替不会迫使整个目录重构。

## 2. 分类方法与边界原则

### 2.1 三条正交轴

知识分类同时采用三条轴，但只让“学习依赖”决定目录顺序：

- **生命周期轴**：数据 → 预训练 → 后训练 → 评估 → 部署 → 运行。
- **系统栈轴**：模型算法 → 算子/编译器 → 单机运行时 → 分布式系统 → 集群平台 → 在线服务。
- **工程目标轴**：质量、吞吐、延迟、显存/内存、成本、可靠性、安全。

每篇卡片只设一个“主归属模块”，其他关系用交叉链接表达。例如 FlashAttention 主归属“注意力与长上下文”，其 CUDA 实现链接到“GPU 算子与编译器”；PagedAttention 主归属“推理运行时”，不再与注意力数学混放。

### 2.2 关键模块边界

- **模型评估 vs 系统基准**：准确率、偏好、安全性属于模型评估；TTFT、TPOT、吞吐、功耗和并发曲线属于系统基准。
- **推理算法 vs 在线服务**：量化、投机解码、KV Cache 属于单模型推理；动态批处理、路由、自动扩缩容、背压和多租户属于在线服务。
- **分布式训练 vs 训练平台**：DP/TP/PP/EP、集合通信属于训练计算图；数据供给、Checkpoint、弹性恢复、实验追踪和作业编排属于训练平台。
- **RAG vs Agent**：RAG 的核心状态是语料、索引和证据；Agent 的核心状态是任务、工具、计划、记忆和执行轨迹。
- **安全评估 vs 安全工程**：越狱成功率等测量属于评估；权限隔离、秘密管理、供应链、审计与事故响应属于安全工程。
- **原理卡 vs 工具说明**：稳定原理放主章节；快速变化的框架/API 放 `reference/implementations/`，并注明版本和核验日期。

## 3. 推荐目录结构

```text
infra-handbook/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── content/
│   ├── 00-start-here/
│   │   ├── roadmap.md
│   │   ├── learning-paths.md
│   │   ├── glossary.md
│   │   └── system-map.md
│   │
│   ├── 01-llm-foundations/
│   │   ├── 01-math-and-performance-model/
│   │   ├── 02-language-modeling-and-tokenization/
│   │   ├── 03-transformer-architecture/
│   │   ├── 04-attention-and-long-context/
│   │   └── 05-modern-architecture-extensions/
│   │
│   ├── 02-model-lifecycle/
│   │   ├── 06-data-engineering/
│   │   ├── 07-pretraining-and-scaling/
│   │   ├── 08-supervised-and-parameter-efficient-tuning/
│   │   ├── 09-alignment-and-reasoning-training/
│   │   └── 10-model-evaluation-and-safety/
│   │
│   ├── 03-ai-infrastructure/
│   │   ├── 11-accelerator-and-server-architecture/
│   │   ├── 12-gpu-programming-kernels-and-compilers/
│   │   ├── 13-distributed-training/
│   │   ├── 14-training-platform-and-reliability/
│   │   ├── 15-inference-algorithms/
│   │   ├── 16-inference-runtime-and-serving/
│   │   └── 17-cluster-scheduling-observability-and-cost/
│   │
│   └── 04-llm-systems/
│       ├── 18-rag-and-retrieval-systems/
│       ├── 19-agent-and-context-systems/
│       └── 20-security-governance-and-supply-chain/
│
├── labs/
│   ├── foundations/
│   ├── training/
│   ├── kernels/
│   ├── inference/
│   └── production/
├── reference/
│   ├── formulas/
│   ├── checklists/
│   ├── failure-playbooks/
│   ├── implementations/
│   └── evidence-ledger.md
├── assets/
│   └── diagrams/
└── tools/
    ├── validate_frontmatter.py
    ├── check_links.py
    └── build_search_index.py
```

`backend/kb/` 可以暂时继续作为应用的运行时知识库；内容迁移稳定后，再让应用从 `content/` 构建索引，避免长期维护两份正文。

## 4. 模块定义、核心主题与一手依据

### Part I：LLM Foundations

#### 01. 数学基础与性能模型

**回答的问题**：张量运算、概率目标与硬件性能之间如何建立可计算的直觉？

核心主题：

- 向量/矩阵、张量形状、矩阵乘法与 Einstein notation。
- 概率、最大似然、交叉熵、KL 散度、梯度与反向传播。
- 参数量、激活、优化器状态、FLOPs、带宽和通信量估算。
- 算术强度、compute-bound / memory-bound、Roofline 模型。
- 延迟、吞吐、利用率、扩展效率的基本定义。

一手依据：[Roofline 性能模型原始论文](https://doi.org/10.1145/1498765.1498785)、[CUDA Programming Guide：编程模型与 GPU 内存层次](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。

#### 02. 语言建模与 Tokenization

**回答的问题**：文本如何变成训练信号和模型可以处理的离散序列？

核心主题：

- 自回归、掩码语言建模、序列到序列目标及其适用边界。
- 词级、字符级、BPE、Unigram、byte-level 与无 OOV 设计。
- 词表大小对参数、序列长度、多语言和代码/数字表示的影响。
- BOS/EOS/PAD、聊天模板、loss mask、packing。
- logits、softmax、temperature、top-k/top-p 和停止条件。

一手依据：[Neural Machine Translation of Rare Words with Subword Units（BPE）](https://arxiv.org/abs/1508.07909)、[SentencePiece](https://arxiv.org/abs/1808.06226)、[GPT-3 的自回归语言建模设定](https://arxiv.org/abs/2005.14165)。

#### 03. Transformer 架构

**回答的问题**：标准 Transformer 与现代 decoder-only LLM 的计算图如何构成？

核心主题：

- Encoder、Decoder、decoder-only 与 causal mask。
- scaled dot-product attention、multi-head attention 与 FFN。
- residual path、Pre-LN/Post-LN、LayerNorm/RMSNorm。
- ReLU/GELU/SwiGLU、参数分布与初始化。
- 位置编码：绝对位置、RoPE、ALiBi。
- 一次 forward/backward 的张量形状和显存生命周期。

一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)、[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

#### 04. 注意力与长上下文

**回答的问题**：注意力的二次复杂度、KV Cache 和上下文扩展怎样影响质量与效率？

核心主题：

- MHA、MQA、GQA 的表示能力与 KV 容量权衡。
- prefill / decode 的不同计算特征与 KV Cache 公式。
- FlashAttention 的 IO-aware tiling 与 online softmax。
- 稀疏、滑窗、线性注意力及其正确比较方式。
- RoPE 外推、Position Interpolation、ALiBi 与长上下文评估。
- context parallel 与 sequence parallel 的算法侧动机；系统实现归入模块 13。

一手依据：[Fast Transformer Decoding / MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[ALiBi](https://arxiv.org/abs/2108.12409)、[Extending Context Window with Position Interpolation](https://arxiv.org/abs/2306.15595)。

#### 05. 现代架构扩展

**回答的问题**：当标准 dense Transformer 不满足容量或序列效率需求时，架构如何扩展？

核心主题：

- 稀疏 Mixture-of-Experts、router、top-k routing、负载均衡。
- Expert Parallel 与 all-to-all 的算法来源；通信实现归入模块 13。
- 状态空间模型与 Transformer/SSM 混合架构。
- Encoder-Decoder、Prefix LM、multimodal projector 的边界性介绍。
- 架构比较必须同时报告参数总量、激活参数、训练 tokens 和推理代价。

一手依据：[Switch Transformers](https://arxiv.org/abs/2101.03961)、[GShard](https://arxiv.org/abs/2006.16668)、[Mamba](https://arxiv.org/abs/2312.00752)。

### Part II：Model Lifecycle

#### 06. 数据工程

**回答的问题**：原始数据如何变成可追溯、可复现、污染可控的训练语料？

核心主题：

- 数据获取、许可、来源和版本化；数据/模型卡。
- 语言识别、质量过滤、敏感信息与有害内容处理。
- exact/fuzzy/semantic dedup、MinHash/LSH、跨 split 去重。
- 数据混合、采样权重、课程学习与多语言平衡。
- benchmark contamination 与训练/评估边界。
- 数据流水线吞吐、shard、shuffle、streaming 与确定性。

一手依据：[Datasheets for Datasets](https://arxiv.org/abs/1803.09010)、[DataComp-LM](https://arxiv.org/abs/2406.11794)、[The FineWeb Datasets](https://arxiv.org/abs/2406.17557)、[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)。

#### 07. 预训练与 Scaling

**回答的问题**：在给定数据、计算和时间预算下，如何稳定训练一个基座模型？

核心主题：

- 参数、数据、计算的 scaling law 与 compute-optimal 配置。
- 优化器、学习率、warmup、weight decay、gradient clipping。
- fp32/fp16/bf16/fp8、loss scaling、累积误差与训练稳定性。
- dense 与 MoE 的训练预算、路由稳定性和容量因子。
- loss spike、梯度/激活监控、消融实验和小模型外推。
- 训练 FLOPs、利用率（MFU/HFU）与碳/能源记录。

一手依据：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)、[Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/abs/2203.15556)、[Adam](https://arxiv.org/abs/1412.6980)、[Mixed Precision Training](https://arxiv.org/abs/1710.03740)。

#### 08. 监督微调与参数高效微调

**回答的问题**：如何在保持基座能力的同时，以可控成本适配任务和对话格式？

核心主题：

- instruction 数据构造、去重、质量审核与 chat template。
- SFT loss mask、packing、sequence length 与超参选择。
- 全量微调、LoRA、QLoRA、prompt/prefix tuning 的适用边界。
- 领域继续预训练、数据 replay 与灾难性遗忘。
- adapter 合并、量化基座兼容性和可部署产物。
- 训练前后必须使用固定评估集与回归门槛。

一手依据：[LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)、[Prefix-Tuning](https://arxiv.org/abs/2101.00190)。

#### 09. 对齐与推理训练

**回答的问题**：如何把语言模型训练成遵循偏好、约束和可验证目标的策略？

核心主题：

- RLHF：demonstration、preference、reward model、policy optimization。
- Bradley-Terry、奖励模型校准、长度偏置和 reward overoptimization。
- PPO/KL control 与 reference policy。
- DPO 及其假设、beta、离线偏好数据的局限。
- 可验证奖励、结果监督/过程监督和 reasoning training。
- reward hacking、模式坍塌和能力回退的监控。

一手依据：[InstructGPT](https://arxiv.org/abs/2203.02155)、[PPO](https://arxiv.org/abs/1707.06347)、[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)。

#### 10. 模型评估与安全评测

**回答的问题**：如何以可复现的方法判断模型能力、可靠性和安全性是否变化？

核心主题：

- task taxonomy：知识、推理、代码、数学、长上下文、多语言。
- exact match、pass@k、pairwise preference、calibration 和置信区间。
- prompt/template 敏感性、随机性、版本固定和污染检查。
- 人评、LLM-as-a-Judge、position/verbosity/self bias。
- 事实性、groundedness、鲁棒性、越狱和红队评估。
- evaluation harness、回归门槛、结果与模型/代码/数据版本绑定。

一手依据：[MMLU](https://arxiv.org/abs/2009.03300)、[HELM](https://arxiv.org/abs/2211.09110)、[HumanEval](https://arxiv.org/abs/2107.03374)、[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)、[NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)。

### Part III：AI Infrastructure

#### 11. 加速器、服务器与互连

**回答的问题**：GPU/AI accelerator、主机和网络拓扑如何决定可实现的训练与推理性能？

核心主题：

- SM/warp/SIMT、Tensor Core、数据类型与理论峰值。
- register、shared memory、L1/L2、HBM、NUMA 与 pinned memory。
- PCIe、NVLink/NVSwitch、InfiniBand/RoCE 的带宽/延迟/拓扑。
- GPUDirect RDMA、DMA、host-device 与 peer-to-peer 数据路径。
- 单机多卡、机架和集群拓扑；拓扑感知放置。
- 功耗、散热、故障域和硬件可观测性。

一手依据：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[NVIDIA GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)、[NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)。

#### 12. GPU 编程、算子与编译器

**回答的问题**：模型中的数学操作如何变成高效且数值正确的设备代码？

核心主题：

- kernel launch、grid/block/warp、coalescing、bank conflict、occupancy。
- GEMM、softmax、normalization、attention 的数据复用。
- kernel fusion、tiling、double buffering 与 async copy。
- numerical stability、reduction order、determinism 和精度验证。
- CUDA、Triton、PTX 与图编译/算子编译的层次。
- profiler 驱动的优化：先定位瓶颈，再改变实现。

一手依据：[CUDA Programming Guide：GPU memory spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#gpu-device-memory-spaces)、[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)、[FlashAttention](https://arxiv.org/abs/2205.14135)。

#### 13. 分布式训练

**回答的问题**：当模型或 batch 超过单卡能力时，状态和计算应怎样切分与通信？

核心主题：

- Data/Tensor/Pipeline/Sequence/Context/Expert Parallel。
- 3D/4D parallelism、parallel mesh 与拓扑映射。
- AllReduce、ReduceScatter、AllGather、AllToAll、P2P。
- ZeRO/FSDP 的参数、梯度和优化器状态分片。
- activation checkpointing、offload、microbatch、1F1B 和 bubble。
- 通信量公式、overlap、straggler 与 scaling efficiency。

一手依据：[Megatron-LM](https://arxiv.org/abs/1909.08053)、[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)、[ZeRO](https://arxiv.org/abs/1910.02054)、[PyTorch FSDP 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)、[NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

#### 14. 训练平台与可靠性

**回答的问题**：如何把一次分布式训练变成可恢复、可复现、可审计的长期生产作业？

核心主题：

- dataset shard、prefetch、shuffle、CPU/存储/网络供给和 GPU 饥饿。
- 作业配置、容器镜像、依赖锁定、随机种子和环境快照。
- distributed checkpoint、异步保存、resharding 和恢复演练。
- elastic launch、节点失败、hang 检测、超时与最小故障域。
- loss/gradient/MFU/data throughput 监控和异常归因。
- 实验追踪、artifact lineage、产物验证和发布门槛。

一手依据：[PyTorch Distributed Checkpoint 官方文档](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)、[PyTorch Elastic 官方文档](https://docs.pytorch.org/docs/stable/distributed.elastic.html)、[Kubernetes Jobs 官方文档](https://kubernetes.io/docs/concepts/workloads/controllers/job/)。

#### 15. 推理算法

**回答的问题**：在固定模型语义或可接受质量损失下，如何减少生成所需的计算、内存与带宽？

核心主题：

- prefill/decode、KV Cache 容量、算术强度和 roofline 分析。
- weight-only、weight-activation、KV quantization；校准和误差评估。
- GPTQ/AWQ/SmoothQuant 的不同假设和部署约束。
- speculative decoding、draft model、acceptance rate 与收益上限。
- prefix caching、chunked prefill、KV offload/compression。
- TTFT、TPOT、ITL、吞吐与质量之间的优化目标冲突。

一手依据：[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[SmoothQuant](https://arxiv.org/abs/2211.10438)、[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)。

#### 16. 推理运行时与在线服务

**回答的问题**：如何把推理算法组合成支持真实并发流量的高效服务？

核心主题：

- model loading、sharding、memory pool、CUDA graph 和执行计划。
- PagedAttention、continuous/in-flight batching 与请求生命周期。
- admission control、queueing、cancellation、timeout 和 backpressure。
- data/tensor/pipeline/expert parallel inference 与多副本扩展。
- prefill-decode disaggregation、prefix-aware routing 与 cache locality。
- streaming API、模型版本、灰度、回滚和一致性边界。

一手依据：[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)、[Orca：迭代级调度](https://www.usenix.org/conference/osdi22/presentation/yu)、[NVIDIA TensorRT-LLM 官方文档](https://docs.nvidia.com/tensorrt-llm/)、[NVIDIA Triton Inference Server 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)。

#### 17. 集群调度、可观测性与成本

**回答的问题**：如何让多模型、多租户 AI 工作负载在集群上稳定、可测且经济地运行？

核心主题：

- GPU device plugin、资源声明、quota、priority、preemption、gang scheduling。
- topology-aware scheduling、fragmentation、MIG/共享与隔离。
- SLI/SLO：availability、TTFT/TPOT 分位数、queue time、error rate。
- traces/metrics/logs、请求到 GPU kernel 的关联与高基数控制。
- load test、开放环/关闭环、warmup、稳定态和可复现报告。
- cost/token、GPU-hour、能耗、利用率、容量规划和性能回归。

一手依据：[Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)、[Kubernetes Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/)、[OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)、[OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)、[MLPerf Inference](https://docs.mlcommons.org/inference/index_gh/)、[MLPerf Training](https://mlcommons.org/benchmarks/training/)。

### Part IV：LLM Systems

#### 18. RAG 与检索系统

**回答的问题**：如何让生成结果建立在可更新、可引用、可度量的外部证据上？

核心主题：

- ingestion、解析、切分、metadata 与 provenance。
- sparse/dense/hybrid retrieval、embedding、reranker、RRF。
- Flat/IVF/PQ/HNSW/DiskANN 与 recall-latency-memory 权衡。
- query rewrite、multi-hop、filter、context packing 与 citation。
- retrieval、context、generation 三段式评估和失败归因。
- index freshness、权限过滤、删除传播和 prompt injection 防护。

一手依据：[Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)、[DPR](https://arxiv.org/abs/2004.04906)、[HNSW](https://arxiv.org/abs/1603.09320)、[DiskANN](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)。

#### 19. Agent 与上下文系统

**回答的问题**：如何让模型在受控环境中规划、调用工具、管理状态并完成多步任务？

核心主题：

- tool schema、structured output、validation 和错误恢复。
- ReAct、plan-and-execute、reflection 的状态机视角。
- 短期上下文、长期记忆、压缩、检索和 state persistence。
- 单 Agent / 多 Agent 的适用条件、协调开销和终止条件。
- sandbox、least privilege、human approval 与可审计执行轨迹。
- task success、step success、cost、latency、安全和可重复性评估。

一手依据：[ReAct](https://arxiv.org/abs/2210.03629)、[Toolformer](https://arxiv.org/abs/2302.04761)、[Model Context Protocol Specification](https://modelcontextprotocol.io/specification/)。

#### 20. 安全、治理与供应链

**回答的问题**：如何覆盖数据、模型、依赖、运行时和 Agent 行为的全生命周期风险？

核心主题：

- threat modeling：训练数据、模型权重、提示、工具、输出和基础设施资产。
- prompt injection、insecure output handling、excessive agency 与数据泄漏。
- 数据/模型投毒、后门、模型窃取、成员推断和资源滥用。
- secrets、身份、最小权限、网络/容器隔离、审计和 incident response。
- dependency/model artifact provenance、SBOM、签名、哈希和可复现构建。
- model/data cards、license、隐私、风险登记与发布审批。

一手依据：[NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework)、[NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)、[NIST Adversarial Machine Learning Taxonomy](https://csrc.nist.gov/pubs/ai/100/2/e2025/final)、[OWASP Top 10 for LLM and GenAI](https://genai.owasp.org/initiatives/top-10-for-llm-and-genai/)、[SLSA Specification](https://slsa.dev/spec/)、[SPDX Specification](https://spdx.github.io/spdx-spec/v3.0/)。

## 5. 现有 10 模块的迁移映射

### 可直接保留并重命名

- `01-transformer` → `03-transformer-architecture`。
- `04-distributed` → `13-distributed-training`。
- `05-peft` → `08-supervised-and-parameter-efficient-tuning`。
- `06-alignment` → `09-alignment-and-reasoning-training`。

### 应拆分后迁移

- `02-attention`：attention 数学、GQA、长上下文、FlashAttention 进入模块 04；KV Cache、PagedAttention 进入模块 15/16。
- `03-pretrain`：tokenizer 进入模块 02；数据工程进入模块 06；scaling、优化器与训练稳定性进入模块 07；MoE 架构进入模块 05，Expert Parallel 实现进入模块 13。
- `07-inference`：解码、量化、投机解码进入模块 15；continuous batching、推理引擎、PD 分离进入模块 16。
- `08-rag-agent`：RAG、embedding、vector index 进入模块 18；Agent、工具与上下文状态进入模块 19。
- `09-eval`：能力、幻觉、LLM-as-Judge 进入模块 10；生产安全控制进入模块 20。
- `10-coding`：不再作为平行理论模块；代码拆入对应章节的 lab，公式与速查模板进入 `reference/formulas/`。

### 当前最明显的知识空白

1. 缺少 GPU/服务器/网络拓扑的底层性能模型（模块 11）。
2. 缺少 CUDA/Triton/算子融合与数值正确性（模块 12）。
3. 分布式训练缺少拓扑映射、通信重叠、Expert/Context Parallel（模块 13）。
4. 缺少训练数据供给、Checkpoint、弹性恢复和故障诊断（模块 14）。
5. 推理内容尚未把算法、runtime 和在线 serving 分层（模块 15–17）。
6. 缺少集群调度、SLO、可观测性、基准方法和 cost/token（模块 17）。
7. RAG/Agent 缺少权限、安全边界、状态机和可复现评估（模块 18–20）。
8. 缺少开源所需的来源台账、许可、贡献规范和自动质量检查。

## 6. 扩充优先级

### P0：先建立 AI Infra 主干（建议首批 45–60 篇）

优先补模块 01、11–17、20。原因是现有内容的 Transformer、训练、对齐、RAG 已有基础，而硬件到生产服务的完整因果链几乎为空。首批至少覆盖：

- Roofline、GPU memory hierarchy、Tensor Core 与数据类型。
- PCIe/NVLink/InfiniBand/RoCE、集合通信与拓扑映射。
- CUDA execution/memory、coalescing、tiling、fusion、profiling。
- DP/TP/PP/EP/CP、ZeRO/FSDP、activation checkpointing。
- data loader、distributed checkpoint、elastic recovery、hang/straggler。
- quantization、speculative decoding、KV Cache、PagedAttention。
- continuous batching、queueing、backpressure、PD disaggregation。
- Kubernetes GPU 调度、SLI/SLO、OpenTelemetry、MLPerf、cost/token。
- prompt injection、least privilege、artifact provenance。

### P1：重构并加深模型生命周期（建议 35–45 篇）

拆分现有模块 02、03、05、06、09，补齐数据 provenance、污染、混合精度、训练稳定性、偏好数据质量、评估统计与复现。迁移期间必须保留旧 `id` 或提供 alias，避免学习数据库中的卡片记录失联。

### P2：强化应用系统与实践（建议 25–35 篇）

扩充 RAG 评估、索引运维、Agent 状态机、工具权限、多 Agent 的适用边界；把现有手写代码改成可以运行、带输入输出断言和性能测量的 labs。

### 暂不进入主线

- 每个新模型的新闻式追踪；应放 `reference/implementations/` 或模型档案。
- 厂商产品功能罗列；只保留能够映射到稳定原理的实现案例。
- 只有 API 截图而无机制、指标或复现实验的内容。
- 多模态生成、推荐系统等大分支；后续可作为扩展 Part，不应稀释当前 LLM + AI Infra 主线。

## 7. 开源内容质量标准

建议每篇正文统一包含以下字段和章节：

```yaml
---
id: stable-slug
title: 中文标题
module: 13-distributed-training
level: foundation | intermediate | advanced
status: draft | reviewed | verified
updated: 2026-09-04
prerequisites: [01-roofline, 13-collectives]
tags: [tensor-parallel, all-reduce]
sources:
  - type: paper | official-doc | specification | source-code
    url: https://...
    accessed: 2026-09-04
---
```

正文结构建议固定为：

1. **问题与结论**：这项机制解决什么问题，适用边界是什么。
2. **机制与推导**：张量形状、公式、状态变化或时序。
3. **资源账本**：参数、激活、KV、FLOPs、字节、通信量、延迟。
4. **实现映射**：至少指向一个官方实现/文档，并注明版本。
5. **失败模式**：现象 → 可能机制 → 验证指标 → 修复方向。
6. **最小实验**：可运行命令、固定输入、期望输出和环境信息。
7. **追问与交叉链接**：连接前置知识和系统上下游。
8. **来源**：关键事实逐条可追溯；实验值与论文值不得混写。

合并门槛：

- 核心技术主张至少有一个一手来源；易变事实必须标版本和核验日期。
- 所有量化结论必须写假设、公式、单位和可复算示例。
- 性能对比必须报告硬件、软件版本、batch/sequence、精度、并发和统计方法。
- 明确区分“论文结论”“官方实现行为”“作者实验结果”和“经验建议”。
- 链接检查、frontmatter 校验、重复 `id` 检查、代码/lab 冒烟测试进入 CI。
- 贡献者不得直接复制论文或官方文档长段落；使用自己的推导和表述，并保留引用。

## 8. 推荐的学习路径

- **LLM 初学者**：01 → 02 → 03 → 04 → 06 → 07 → 08 → 10。
- **训练系统工程师**：01 → 11 → 12 → 13 → 14 → 07 → 10。
- **推理/服务工程师**：01 → 03 → 04 → 11 → 12 → 15 → 16 → 17。
- **RAG/Agent 工程师**：02 → 03 → 10 → 15 → 16 → 18 → 19 → 20。
- **面试冲刺**：每个模块先读“资源账本”和“失败模式”，再完成对应 lab 与追问。

## 9. 最终建议

仓库的长期差异化不应只是“知识点更多”，而应是：

> 每个概念都能从模型行为一路追到张量、算子、通信、运行时策略、生产指标与验证证据。

因此，后续扩充应优先形成三条贯穿全书的因果链：

1. **OOM 链**：模型结构 → 激活/KV/优化器账本 → 分片/重计算 → runtime memory manager → 指标与诊断。
2. **性能链**：算术强度 → kernel → collective → batching/scheduling → TTFT/TPOT/吞吐 → cost/token。
3. **质量与风险链**：数据 → 训练目标 → 对齐 → evaluation → serving context/tool → 安全控制与审计。

这三条链既能支持系统学习，也能自然生成面试题、实验、故障排查手册和检索问答，是从当前“LLM 八股 Agent”演进为开源 `infra-handbook` 的最稳固主线。
