# ONNX、OM 与 ONNX Runtime GenAI 软件栈调研

更新日期：2026年11月5日

## 1. 定位与关系

**ONNX 和 OM 是模型表示或部署产物；ONNX Runtime 和 ONNX Runtime GenAI 是执行模型的软件。四者处于不同层次。**

| 名称 | 定位 | 核心作用 | 实际价值 |
|---|---|---|---|
| ONNX（Open Neural Network Exchange，开放神经网络交换格式） | 模型交换格式与计算图表示 | 描述算子、连接关系、张量类型和权重 | 连接训练框架与不同推理工具，减少模型交换成本 |
| OM（Offline Model，离线模型） | 昇腾离线编译模型 | 保存面向目标环境编译、优化后的模型部署内容 | 将模型转换与部分优化工作前置，便于在目标设备加载执行 |
| ONNX Runtime | 通用模型推理运行时 | 加载模型、优化计算图、调度算子，通过执行提供程序接入硬件 | 为应用提供相对统一的模型执行接口 |
| ONNX Runtime GenAI | 基于 ONNX Runtime 的生成式推理库 | 管理分词、生成循环、采样和键值缓存 | 减少应用开发者实现大模型生成流程的工作 |

ONNX 文件可以被 ONNX Runtime 等推理工具加载，也可以作为转换工具的输入，进一步生成 OM。模型格式通用不意味着任意模型都能在任意后端运行。[ONNX 规范](https://onnx.ai/onnx/repo-docs/IR.html)、[ONNX Runtime 硬件接入架构](https://onnxruntime.ai/docs/execution-providers/)

## 2. ONNX 与 OM：核心区别

| 维度 | ONNX | OM |
|---|---|---|
| 所处阶段 | 模型导出与交换阶段的计算图表示 | 面向目标环境编译后的部署产物 |
| 优化方式 | 推理工具加载后仍可进行图优化、算子融合或后端编译 | 生成模型时完成相应编译与优化，具体执行方式取决于模型和工具链 |
| 可移植性 | 可被多种工具读取，仍受算子版本和扩展算子支持约束 | 与目标芯片、软件版本和编译配置相关 |
| 修改与分析 | 便于查看计算图、检查节点和转换模型 | 修改模型结构通常需要从源模型重新编译 |
| 动态形状 | 格式可以表达动态维度 | 可配置动态档位或动态形状范围，受目标环境约束 |

**ONNX 转 OM 是模型编译，不是修改文件后缀。** 转换工具需要检查算子支持、处理输入形状，并进行目标相关优化。OM 也不能简单理解为每个细节都完全固定的机器码文件：它仍由昇腾运行时加载和执行。

### 动态形状如何理解

ONNX 可以用符号维度表示可变输入，例如大模型词元输入为 `[batch, sequence_length]`，图像输入为 `[batch, 3, height, width]`。但模型内部计算和硬件后端也必须支持相应变化；仅把输入维度改成变量，不会自动消除模型中写死的尺寸。[动态维度规范](https://onnx.ai/onnx/repo-docs/IR.html)

OM 同样支持动态输入，主要有两种方式：

| 方式 | 编译与运行方式 | 示例 |
|---|---|---|
| 动态档位 | 编译时配置若干形状，运行时选择已配置档位 | 批量大小配置为 `1、2、4、8`，不直接包含 `3` |
| 动态形状范围 | 构建时配置维度范围，运行时设置实际形状 | 某个维度在配置范围内变化，仍需满足算子和模型约束 |

固定形状 OM 不能在运行时随意改变尺寸；动态 OM 也不是无限制接受任意形状。具体能力取决于芯片、CANN 版本、算子和构建配置。[昇腾动态输入说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/aclcppdevg_000040.html)

## 3. 三条执行路径与软件栈

### 路径一：ONNX Runtime 运行 ONNX 模型

```text
业务应用：输入准备、结果处理
                ↓
ONNX Runtime ← ONNX 模型
计算图优化、模型执行、算子调度
                ↓
执行提供程序
对接对应硬件后端，承接支持的节点或子图
                ↓
硬件加速库、驱动 → 目标芯片
```

执行提供程序是 ONNX Runtime 的硬件接入机制。它可以将受支持的计算交给目标后端；后端不支持的部分是否回退到中央处理器，取决于配置和模型。回退可能带来设备间数据搬运和性能损失。[执行提供程序说明](https://onnxruntime.ai/docs/execution-providers/)

### 路径二：ONNX 编译成 OM 后执行

```text
模型准备阶段
训练模型 → 导出 ONNX → 昇腾模型转换工具编译 → OM

设备运行阶段
业务应用：输入准备、结果处理
                ↓
昇腾计算语言接口（AscendCL）：加载 OM、设置输入、执行模型
                ↓
CANN 运行时与计算能力 → 驱动 → 昇腾芯片
```

这条路径将模型转换与编译放在部署准备阶段。运行时应用直接加载 OM，不必通过 ONNX Runtime；动态输入需要使用与模型构建方式相匹配的设置接口。[昇腾动态输入执行指南](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/aclcppdevg_000040.html)

### 路径三：ONNX Runtime GenAI 运行生成式模型

```text
聊天助手、知识库、Agent 等业务应用
                ↓
ONNX Runtime GenAI
分词、生成循环、采样、键值缓存管理、逐词元输出
                ↓
ONNX Runtime ← 适配的 ONNX 模型与配置
                ↓
执行提供程序 → 硬件加速库、驱动 → 目标芯片
```

ONNX Runtime GenAI 控制生成过程，ONNX Runtime 执行每一步计算。二者不替代厂商加速库和驱动。[官方生成接口说明](https://onnxruntime.ai/docs/genai/)

## 4. ONNX Runtime GenAI 的功能与价值

大语言模型通常通过“处理提示词 → 生成一个词元 → 带着历史状态继续计算”的循环产生回答。ONNX Runtime GenAI 将这一流程封装为可供应用调用的库。

| 功能 | 具体作用 |
|---|---|
| 分词与解码 | 将文本转换成词元编号，将输出词元还原成文本 |
| 生成控制 | 组织预填充和逐词元解码，根据结束标记或长度条件停止 |
| 搜索与采样 | 根据输出分数选择下一个词元，提供贪心搜索、束搜索及 Top-K、Top-P 等策略 |
| 键值缓存管理 | 保存注意力历史状态，减少重复计算 |
| 流式输出 | 在生成过程中逐步将结果交给应用 |
| 格式约束 | 约束生成格式，辅助结构化输出和工具调用 |

它的直接价值是让开发者把时间用于应用交互和业务逻辑，无须从头实现生成循环。工具调用中的实际工具执行、业务权限和任务规划仍由上层应用负责。[功能说明](https://onnxruntime.ai/docs/genai/)

部署时通常需要一套适配好的模型目录：模型计算图、权重、分词器以及 `genai_config.json`。配置文件描述模型结构、输入输出映射、缓存与生成参数；多模态模型可能包含多个计算图。已有 ONNX 文件不等于可以直接交给它运行。[模型配置说明](https://onnxruntime.ai/docs/genai/reference/config.html)

## 5. 昇腾的支持情况

**昇腾存在 ONNX Runtime 的 CANN 执行提供程序。** 官方文档说明该适配由华为开发，目前归入社区维护列表；其路径是：

**应用 → ONNX Runtime → CANN 执行提供程序 → CANN、驱动 → 昇腾芯片。**

因此，昇腾既有 ONNX Runtime 接入方式，也有提前编译 OM 后通过 AscendCL 执行的方式。CANN 执行提供程序本身还提供子图相关选项及离线模型输出选项，不能把两种入口理解为底层完全互不相关的计算体系。[CANN 执行提供程序](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)

**ONNX Runtime 的支持与 ONNX Runtime GenAI 的支持需要分别判断。** 当前 ONNX Runtime GenAI 官方硬件支持矩阵没有列出 CANN／昇腾，不能由前者的存在推导出后者已完成适配。针对 310P 等具体芯片，也需要核对对应软件版本、算子和模型，不能将 CANN 后端的存在等同于任意模型均可用。[GenAI 支持矩阵](https://github.com/microsoft/onnxruntime-genai)

## 6. 端侧推理如何选择

| 部署需求 | 优先评估路线 | 选择依据 |
|---|---|---|
| 图像分类、检测、分割等模型，需要覆盖多种硬件 | ONNX + ONNX Runtime | 使用统一调用接口，复用不同执行提供程序 |
| 硬件与模型相对固定，采用昇腾部署工具链 | ONNX → OM → AscendCL | 将目标相关转换与优化前置，按构建配置部署 |
| 在本地应用内集成受支持的大语言或多模态生成模型 | ONNX Runtime GenAI + ONNX Runtime | 复用生成控制与缓存管理，减少开发工作 |
| 使用专用神经网络处理器 | 厂商已验证的模型与运行路径 | 先确认模型、量化方式、动态形状和算子支持，再决定是否采用 ONNX Runtime |

**三者没有脱离硬件和模型条件的统一性能排名。** ONNX 是模型交换基础，OM 是昇腾部署产物，ONNX Runtime GenAI 是生成控制软件；实际效果应看完整执行路径。视觉任务重点比较端到端时延、帧率与精度；大模型交互重点比较首字响应、生成速度、内存和功耗，并在实际输入长度下判断动态形状是否满足需求。
