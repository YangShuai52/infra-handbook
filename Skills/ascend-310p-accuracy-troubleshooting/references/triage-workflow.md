# Ascend 310P 精度问题定位流程

仅在需要规划诊断、缩小变量、commit/PR 二分或验证修复时读取本文件。

## 1. 建立问题陈述

收集并明确：

- 有问题的 vLLM、vLLM-Ascend 版本或 commit；
- 最近一次精度正常的对应版本或 commit；
- 两侧 vLLM-Ascend 分支、源码路径，以及用户给出的日期区间；
- CANN、驱动、固件、镜像、Python 依赖与硬件拓扑；
- 模型、权重格式/量化方式及权重身份；
- 完整服务启动命令和日志；
- 数据集、benchmark commit、精度配置与评分后处理；
- 最小 curl（若已有）、期望输出、实际输出和复现概率；
- 用户已经确认的定位结论。

缺失信息不会自动阻塞所有工作：先列出现有证据和假设，优先尝试从已知失败样本构造单 curl，并用最能区分原因的低成本实验补齐。不得把猜测写成事实。

## 2. 固化 good / bad 基线

分别构建 last-known-good 与 first-known-bad 环境，并制作差异表。至少包含：

| 维度 | Good | Bad | 是否已控制 |
|---|---|---|---|
| vLLM commit | | | |
| vLLM-Ascend commit | | | |
| CANN/驱动/固件 | | | |
| Python 包 | | | |
| 模型权重 | | | |
| 启动参数 | | | |
| benchmark/数据集 | | | |
| 生成与后处理参数 | | | |
| 硬件/TP/并发 | | | |

先在两侧运行相同的最小请求，再运行原始数据集。若 good 也失败，优先检查复现漂移；若 bad 无法稳定失败，增加重复次数并确认并发/图捕获等状态因素。

## 3. 排除评测层伪回归

依次核对：

- host、port、served model name 是否一致；
- 权重是否完全相同；
- prompt/chat template、thinking 模式是否一致；
- `temperature`、seed（版本支持时）、`max_tokens`、EOS 策略是否一致；
- 输出是否被截断，reasoning 是否被错误纳入评分；
- evaluator 后处理和数据集答案是否发生变化；
- 失败来自 HTTP/超时/重试，还是模型答案本身变化。

保存 raw output，必要时人工核验少量差异样本。只有确认模型输出路径发生有意义变化后，才继续归因到 serving 特性或底层计算。

## 4. 先分析日期或 commit 区间

如果用户提供了两个日期或两个 commit ID，先思考这一区间内哪些代码变化可能影响 310P，再安排定位实验：

1. 确认比较的是哪个仓库、分支和提交拓扑；日期必须转换为该分支上实际存在的边界 commit；
2. 查看区间的 commit/PR 列表和 diff 统计，再聚焦 310P、平台适配及当前模型路径；
3. 结合症状检查图执行、量化/融合、MTP、prefix cache、scheduler、KV/mamba cache、TP/通信、dtype/shape、采样与后处理等相关调用链；
4. 记录每个候选变更的“影响机制、与症状的对应关系、支持/反对证据、验证方式”；
5. 按解释力和验证成本排序，先测试最有区分度的候选。

分析时同时查看 vLLM-Ascend 与其配对 vLLM 的变化，避免遗漏上游接口或默认值改变。日期只是筛选窗口，不代表窗口内所有提交都在目标分支，也不能仅凭文件名、commit message 或 PR 标题定责。静态分析的产物是候选假设，不是根因结论。

可使用的只读检查示例：

```bash
git log --oneline --decorate <good>..<bad>
git diff --stat <good>..<bad>
git diff <good>..<bad> -- <suspected-paths>
git log --since=<date-a> --until=<date-b> --oneline -- <suspected-paths>
```

## 5. 特性叠加拆分

从用户配置回退到最小可靠配置，再一次增加一个特性。候选变量包括：

- eager 与图模式、不同 capture sizes；
- 量化与非量化、`fuse_norm_quant`；
- MTP；
- prefix cache；
- 单请求与多并发、不同 batch size；
- TP 数量与拓扑；
- mamba cache dtype；
- function call / tool parser；
- thinking mode 与 chat template；
- 其他新增优化、融合算子或编译选项。

推荐用矩阵记录每个实验：

| Run | 基线 commit | 唯一改动 | 最小样本 | 全集分数 | 日志/产物 | 结论 |
|---|---|---|---|---|---|---|

若特性之间存在交互，先做单因素，再只对有证据的组合做 2×2 实验。不要同时关闭所有优化后就直接把问题归因给“图模式”或“量化”。

## 6. 从特性映射到代码与 PR

当某个开关稳定控制失败后：

1. 搜索该配置的解析、默认值、分支和调用链；
2. 找到 good/bad 区间内触碰相关路径的 vLLM-Ascend 和 vLLM commits/PR；
3. 检查接口适配、shape/dtype、缓存状态、图捕获、分布式同步、量化 scale、后处理等边界；
4. 优先测试最可能且改动最小的候选，不凭 PR 标题直接定责。

需要区分 vLLM 上游变化与 vLLM-Ascend 适配变化。二者 commit 必须按已验证兼容关系配对，不要任意交叉组合后把安装失败或 ABI 问题算成精度回归。

## 7. Commit 二分

只有在 good/bad 可稳定判定且自动判据可靠时使用 `git bisect`。先定义：

- good 与 bad 边界 commit；
- 每个候选对应的兼容 vLLM commit；
- 构建/启动流程；
- 最小测试命令；
- 明确的 pass/fail 阈值；
- flaky 时的重复次数和停止条件。

每次只改变待二分仓库的 commit。保存构建日志、实际 HEAD、依赖指纹、原始响应和判定。构建失败不能自动视为 bad；应标记 skip 或先处理兼容关系。

若无法可靠自动化，按候选 PR 和时间区间做手动二分，并完整记录每次选择。不要自动删除源码树或构建目录；优先使用独立工作区或容器。

## 8. 无法二分时进行 tensor dump 对比

出现以下情况时，可改用 good/bad tensor dump 定位：commit 无法兼容构建、vLLM 与 vLLM-Ascend 跨仓库配对使边界不连续、中间提交不可运行、问题存在波动而无法形成可靠 pass/fail，或已知功能开关能稳定控制问题但无法进一步二分。

### 对齐两侧执行条件

dump 前必须保证两侧使用相同的：

- 模型权重、tokenizer、prompt 和实际 input token IDs；
- generation 参数、随机种子（若支持）及相同的 prefill/decode 阶段和 decode step；
- batch 中的样本顺序、sequence length、padding 和并发条件；
- TP 数量、rank 对应关系、设备拓扑和 collective 边界；
- dtype、量化配置、图/eager 模式、融合开关和编译配置；
- KV cache、mamba cache、prefix cache 等初始状态。

先用同一侧重复 dump 判断自身波动。如果相同环境两次运行的差异已经达到 good/bad 差异量级，先处理不确定性，不能直接归因代码。

### 逐级缩小 dump 范围

1. 先 dump 少量粗粒度边界，例如 embedding 输出、若干 transformer block 输出、最终 norm 和 logits；不要一开始全量保存所有 tensor。
2. 找到“上一检查点仍一致、下一检查点开始显著偏离”的区间，用中点层继续缩小。
3. 定位到首个偏差 layer 后，再比较该层的 attention、MLP、norm、residual、量化/反量化及通信前后边界。
4. 最后只在可疑算子附近 dump 输入、输出及必要的 scale/zero-point 等辅助 tensor，确认偏差由该算子产生，而不是从上游传入。
5. 图融合可能改变可观察边界。必要时使用语义等价的逻辑边界比较，并分别验证图模式和 eager 模式；dump 本身若使问题消失，应记录其侵入性并换用更窄的采集点。

### 每个 tensor 必须带上下文

为每份 dump 记录：good/bad 标识、仓库 branch/commit、启动命令、请求或样本 ID、prefill/decode 与 step、layer/operator 名、TP rank、shape、stride、dtype、device、量化参数，以及 NaN/Inf 数量、min/max/mean、文件校验值。文件名必须能唯一映射相同逻辑 tensor，不能把不同 rank 的 shard 或融合前后不同语义的 tensor 直接比较。

dump 数据可能很大，也可能包含可还原用户输入的信息。使用用户确认的输出目录和样本，先估算范围，不覆盖已有产物，不自动清理 dump 文件。

### 比较与判定

至少计算 `max_abs_error`、`mean_abs_error`、相对误差、余弦相似度、NaN/Inf 分布，并按 dtype、量化方式和算子累积特性设置容差。低精度或量化路径不要求 bitwise 一致；容差应参考 good 环境重复运行噪声和偏差传播趋势。

“首个偏差点”应满足：其输入在容差内一致，输出首次稳定超出容差，并且该差异能够向下游传播并解释最终 logits/答案变化。若输入已经偏离，继续向上游查找；若输出差异不影响下游或最终结果，不应把它定为根因。

tensor dump 的结论应包括首个偏差 tensor、两侧误差指标、对应源码路径/算子、上游一致性证据、下游传播证据和下一项验证实验。可通过关闭相关融合/特性、替换参考实现或打最小补丁进一步验证。

## 9. 根因与修复验证

根因至少需要满足：

- 变更/特性与失败稳定相关；
- 代码机制能够解释观察到的输出差异；
- revert、最小补丁或等价开关能恢复结果；
- 没有更简单的环境、评测或配置差异解释现象。

修复仅在用户要求时实施。回归验证包含：

- 可稳定区分 good/bad 的单 curl 或其他最小失败样本，并优先验证；
- 原始数据集与原始并发/特性配置；
- last-known-good 对照；
- 受影响特性的关闭与开启两侧；
- 相邻模型、shape、batch、TP 或 dtype 场景（按代码影响范围选择）；
- 性能与内存基本检查，避免精度修复造成明显退化。

复杂精度回归应在最小判据通过后执行；最小 curl 通过但复杂回归失败时，继续定位并发、batch、状态缓存、图捕获或评测层触发条件，不能把最小用例通过当成问题已经完全解决。

## 10. 无法定根因时的输出

不要只说“未复现”或“仍在定位”。给出：

- 已验证事实；
- 被排除的假设及证据；
- 尚未控制的差异；
- 当前最可能的 1–3 个原因；
- 下一项最有区分度的实验、预期结果及各结果对应的后续方向。
