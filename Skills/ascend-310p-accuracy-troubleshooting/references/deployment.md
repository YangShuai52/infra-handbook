# Ascend 310P vLLM / vLLM-Ascend 部署

仅在需要搭建、核验或切换诊断环境时读取本文件。命令来自项目资料，具体分支、依赖和参数可能随版本变化；执行前核对目标版本的仓库说明和本地环境。

## 先确认是否已有定位环境

开始搭建前先询问用户：

1. 是否已经在目标容器内搭建好配套的 vLLM 和 vLLM-Ascend；
2. 若已搭建，vLLM-Ascend 的源码安装绝对路径是什么；
3. 当前 vLLM-Ascend 使用哪个分支和 commit。不要默认使用 `main`，用户可能定位维护分支、发布分支或特性分支；
4. 若未搭建，期望使用哪个 vLLM-Ascend 分支、commit 或版本，以及目标容器如何进入。

若环境已搭建，先进入用户提供的 vLLM-Ascend 源码目录，确认目录和可编辑安装关系：

```bash
cd <vllm-ascend-source-path>
git branch --show-current
git rev-parse HEAD
git status --short
pip show vllm-ascend vllm
```

后续服务化启动、日志复现、源码修改和 debug 均以该 vLLM-Ascend 源码目录为工作目录。不要根据包名猜测源码路径，也不要擅自切换分支或覆盖未提交改动。

## 网络与安装授权

首次需要联网或改变诊断环境时，一次性说明范围并取得用户授权。若用户已经明确允许目标服务器/容器联网、clone、切换诊断 commit 或安装配套组件，不再为下列每一步重复询问：

- clone/fetch vLLM、vLLM-Ascend、ais_bench 及其 submodule；
- 在干净的独立 clone/worktree 中 checkout 用户给出的分支或 commit；
- 安装或更新该诊断环境中的 vLLM、vLLM-Ascend、ais_bench 及仓库声明的普通 Python/build 依赖；
- 为 good/bad/bisect 创建隔离源码副本或可编辑安装。

每次操作仍需记录仓库 URL、HEAD、安装命令、包路径和依赖变化。已有源码含未提交改动时不得直接 checkout；使用新 worktree/clone。一次性授权不包含驱动、固件、CANN、NNAL、PTA/`torch_npu`、批量删除、抢占设备或修改无关系统环境；这些继续遵守后文边界。若执行平台本身要求网络权限审批，直接发起所需工具审批，不再额外进行同内容的对话确认。

## 容器内判断现有版本是否可用

若用户尚未搭建环境，先在目标容器内检查，而不是直接重装：

```bash
pip show vllm vllm-ascend
pip list | grep -E 'vllm|triton'
python -c "import vllm; print(vllm.__version__, vllm.__file__)"
```

如能定位 vLLM-Ascend 源码或包路径，再核对它的分支/commit、`.github/vllm-main-verified.commit`、可编辑安装路径和简单 import。将“可用”定义为：分支符合用户要求、vLLM 配对兼容、包实际从预期位置加载，并能进入后续最小服务验证；不能只看版本字符串。

如果现有 vLLM/vLLM-Ascend 对目标分支不可用，先说明不兼容证据和计划安装的配对版本。用户尚未授权诊断环境变更时只询问一次；已有上述授权时直接执行配对安装，不重复确认。需要卸载时记录原包版本、路径和回滚方式，再卸载两个包：

```bash
pip uninstall vllm-ascend vllm
```

卸载后按用户指定的 vLLM-Ascend 分支/commit 安装，并使用该源码中的 verified commit 配对 vLLM。不要自动卸载 CANN、驱动、固件或其他无关依赖。

## 部署前记录

先记录以下信息，避免安装过程破坏可比较的环境：

```bash
python --version
pip show vllm vllm-ascend
pip show torch-npu
pip freeze
npu-smi info
```

同时记录 vLLM、vLLM-Ascend branch/commit，vLLM-Ascend 源码路径，CANN、PTA/`torch_npu`、驱动、固件版本，容器镜像标签/摘要、310P 卡数及拓扑。若命令在当前环境不可用，记录等价信息，不要为收集信息而改变环境。

## 版本匹配是首要约束

vLLM 与 vLLM-Ascend 必须使用兼容版本。先检出用户指定的 vLLM-Ascend 分支或 commit，不要默认 `main`；再从该检出版本读取它验证过的 vLLM commit：

```bash
cd vllm-ascend
cat .github/vllm-main-verified.commit
```

然后在 vLLM 仓库检出该 commit：

```bash
cd ../vllm
git fetch origin
git checkout <verified-vllm-commit>
```

精度回归定位期间不要自行升级到最新提交；分别固定 last-known-good 与 first-known-bad 的完整 commit。

## 安装模式

### vLLM 与 vLLM-Ascend 均源码安装

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
export VLLM_TARGET_DEVICE=empty
pip install -v -e .

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout <requested-branch-or-commit>
git submodule update --init --recursive
pip install -v -e .
```

检出版本的动作应在安装前完成。

### 只源码安装 vLLM-Ascend

适用于基础镜像中的 vLLM 已与目标 vLLM-Ascend 匹配：

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout <requested-branch-or-commit>
git submodule update --init --recursive
pip install -v -e .
```

### 镜像内可编辑安装，用于调试

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout <requested-branch-or-commit>
pip install -v -e . --no-deps --no-build-isolation
```

这能减少重复搭建依赖的时间，但前提是镜像依赖与目标源码兼容。C++ 代码改动后需重新编译/安装。

## Python 镜像源示例

仅在组织网络策略允许时使用：

```bash
pip install -v -e . \
  -i http://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --trusted-host mirrors.aliyun.com
```

镜像源可能影响解析到的包版本。精度对比中应保存 `pip freeze`，确保两侧除目标变量外一致。

## 310P 特有检查

项目资料指出 310P 不支持 Triton。源码安装后检查：

```bash
pip list | grep -i triton
```

若发现 `triton` 或 `triton_ascend`，先确认当前目标版本和 310P 支持矩阵。卸载会改变环境，必须得到用户明确授权后才执行：

```bash
pip uninstall triton_ascend triton
```

服务启动时通常还需禁用 310P 不支持的 `npugraph_ex`；详见 [accuracy-testing.md](accuracy-testing.md)。

## 构建残留与底层组件变更边界

构建残留可能导致非源码变更引起的编译错误。不得自动批量删除 `csrc/build`、`csrc/output`、`csrc/build_out`、`csrc/third_party/ascend_protobuf` 或 `csrc/third_party/abseil-cpp`。先记录构建日志和这些路径，再让用户手动清理或使用新的源码副本/容器。

### 驱动禁止变更

本 skill 的定位流程禁止升级、降级、重装或卸载 Ascend 驱动。即使证据指向驱动差异，也只能记录当前版本、复现证据和所需兼容关系，将其作为外部约束报告；不得执行或指导当前流程执行驱动变更。

### CANN 或 PTA 变更必须确认

CANN、310P ops、NNAL 或 PTA（通常表现为 `torch_npu`）属于高影响环境组件。只有 controlled comparison、tensor dump、组件替换实验的既有结果或明确调用链证据已经把问题定位到 CANN/PTA 时，才可以提出更新、降级、重装或卸载建议。版本看起来不匹配、存在较新版本或一般性兼容猜测都不足以触发变更。

执行前必须向用户提供并获得明确确认：

- 当前组件名称、版本、安装路径和来源；
- 拟变更的组件及精确目标版本，不得使用含糊的“最新版”；
- 指向 CANN/PTA 的定位证据，以及为何代码或配置层规避不足；
- 受影响的容器/环境、预计中断和磁盘影响；
- 备份或回滚方案；
- 单 curl、tensor 对比和复杂精度回归计划。

用户未确认时停止组件变更，继续提供不修改 CANN/PTA 的诊断结果、临时规避方案或下一项验证实验。获得确认后仍应优先在可回滚的独立容器中操作，并按目标版本的官方安装说明执行；不要把资料示例中的 `9.1.0` 当成固定要求。

## 部署完成后的核验

保留上述原始安装方式。安装完成后必须用运行时证据证明源码、包和解释器一致；无法证明时 fail closed，不得进入服务化和精度测试。

每次首次安装、切换 commit 后重装、C++/自定义算子重编译或依赖修复后，都必须重新执行本节。使用与安装相同的 Python 解释器，不要混用裸 `pip`、其他虚拟环境或另一个 Python：

```bash
DIAG_PYTHON="$(command -v python3 || command -v python)"
"$DIAG_PYTHON" -m pip --version
"$DIAG_PYTHON" -m pip show vllm vllm-ascend torch-npu
```

### 1. 证明源码配对正确

分别在 vLLM 和 vLLM-Ascend 源码目录记录：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

读取当前 vLLM-Ascend checkout 的 `.github/vllm-main-verified.commit`，验证其内容与实际 vLLM `HEAD` 完全相等。若文件缺失、commit 无法解析或两者不等，状态为 blocked；不得仅凭两个包都能 import 就继续精度测试。只有用户明确要求非 verified 配对实验时才允许偏离，并将其标记为受控变量。

### 2. 证明运行时加载了这些源码

执行真实 import，不用 `find_spec()` 代替：

```bash
"$DIAG_PYTHON" - <<'PY'
import json
import sys
import torch
import torch_npu
import vllm
import vllm_ascend

print(json.dumps({
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_path": torch.__file__,
    "torch_npu_path": torch_npu.__file__,
    "vllm_version": getattr(vllm, "__version__", "unknown"),
    "vllm_path": vllm.__file__,
    "vllm_ascend_path": vllm_ascend.__file__,
}, indent=2, sort_keys=True))
PY
```

将 `pip show` 的 `Location`/`Editable project location`、模块绝对路径与预期源码目录逐项比较。可编辑安装必须实际指向刚才记录 commit 的源码树；路径落到旧 clone、镜像自带 site-packages 或另一个 Python 环境时状态为 blocked，先按原安装流程修复，不能启动服务。

### 3. 证明依赖和 310P 后处理可解释

保存安装前后的 `pip freeze` diff，并运行：

```bash
"$DIAG_PYTHON" -m pip check
"$DIAG_PYTHON" -m pip list | grep -i triton || true
```

若已按“310P 特有检查”获得授权并卸载 Triton，再确认 `triton`、`triton_ascend`/`triton-ascend` 均未残留。`pip check` 若只报告目标版本把 Triton 声明为通用依赖，可记录为 310P 元数据例外，并以真实 import 和服务 smoke 决定是否可用；其他缺失或冲突均阻断测试。不得为了让 `pip check` 变绿而自动装回 310P 不支持的组件。

同时确认 CANN、PTA/`torch_npu`、驱动、固件和镜像身份与安装前记录一致；出现意外变化时停止，不把它带入 good/bad 精度对比。

### 4. 输出结构化验收结论

保存一份紧凑的安装验收记录，至少包含：

- `status`: `ready` 或 `blocked`；
- Python 可执行文件和 pip 路径；
- vLLM-Ascend HEAD、verified vLLM commit、实际 vLLM HEAD；
- `pip show` 元数据和三个关键模块的实际加载路径；
- import smoke、依赖检查和 Triton 检查结果；
- CANN、PTA、驱动、固件、镜像与 NPU 拓扑指纹；
- blocked 的准确原因和对应原始日志路径。

只有 `status=ready` 才进入 [accuracy-testing.md](accuracy-testing.md) 的服务化和精度验证。该状态证明当前安装与预期源码一致，但不替代后续真实服务启动、分级 curl 和用户指定目标数据集的回归。

最终至少确认：

- `pip show` 指向预期源码路径；
- 两个仓库 commit 与 `.github/vllm-main-verified.commit` 的关系明确；
- CANN、PTA/`torch_npu`、驱动、固件与镜像标签已记录；
- 没有意外依赖升级；
- 相同模型权重可在 good/bad 环境中读取；
- 后续启动命令使用当前版本实际支持的参数。
