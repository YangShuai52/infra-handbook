# Ascend 310P 精度复现与测试

仅在需要启动服务、制作最小 curl 或运行 ais_bench 时读取本文件。

## 选择运行卡

若用户指定了物理卡号，按用户指定执行并记录，不要自行改选。若用户没有指定：

1. 先检查当前 shell 是否已有 `ASCEND_RT_VISIBLE_DEVICES`，不要静默覆盖继承的设备约束；
2. 使用设备查询命令检查所有可见 310P 的健康状态、占用进程和可用显存；
3. 根据模型预估显存和 `-tp` 数量选择足够的空闲卡。空闲卡至少应无其他活动计算进程、状态正常且可用显存满足服务启动需求；
4. TP 场景选择数量等于 TP degree 的可用卡，并考虑实际拓扑；不要把不同 rank 的需求挤到同一张物理卡；
5. 启动前再次确认占用，避免检查与启动之间卡已被其他任务使用。

检查示例：

```bash
echo "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
npu-smi info
```

确认物理卡号后，只在当前服务 shell 中设置：

```bash
# 单卡示例
export ASCEND_RT_VISIBLE_DEVICES=<physical-device-id>

# TP 多卡示例
export ASCEND_RT_VISIBLE_DEVICES=<physical-id-0>,<physical-id-1>
```

环境变量设置后，进程内可见设备编号可能重新映射；记录物理卡号、变量值和服务使用的逻辑 rank 映射。不要把该变量写入用户 profile、镜像或全局启动脚本。

如果没有满足条件的空闲卡，报告各卡的健康状态、占用和可用显存并等待用户处理。不得终止其他任务、抢占占用卡或为了腾卡修改其他进程。

## 启动服务

先用当前版本的 `vllm serve --help` 核对参数。以下是诊断模板，不是可直接照抄的固定命令：

```bash
cd <vllm-ascend-source-path>
# 用户未指定卡时，使用上一步选择的物理卡号
export ASCEND_RT_VISIBLE_DEVICES=<selected-physical-device-ids>
vllm serve <weight-path> \
  --host 127.0.0.1 \
  --port 1025 \
  --served-model-name <model-name> \
  --trust-remote-code \
  --max-model-len <context-length> \
  --gpu-memory-utilization 0.6 \
  --max-num-seqs 32 \
  --mamba-ssm-cache-dtype float16 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":false}}'
```

重要说明：

- `enable_npugraph_ex=false` 是项目资料给出的 310P 约束。
- `gpu-memory-utilization=0.6` 是 RC 设备精度/性能测试经验值，不是普适最优值。低内存模型可按实际 OOM 情况调整，但调整必须进入对比记录。
- `max-num-seqs` 影响并发和内存；精度定位先从低并发、确定性最小用例开始，再恢复用户并发。
- `mamba-ssm-cache-dtype` 只应在模型/版本需要时使用。
- `cudagraph_capture_sizes` 与 TP、并发和版本有关。图模式本身也是需要隔离的变量。
- `--dtype float16`、`-tp <n>`、`--enable-auto-tool-choice` 与 `--tool-call-parser qwen3_coder` 按模型和复现场景启用，不要无条件添加。

保存选中的物理卡号、`ASCEND_RT_VISIBLE_DEVICES`、完整启动命令、环境变量和从进程启动到服务就绪的日志。
服务拉起、复现和源码 debug 均从用户确认的 vLLM-Ascend 源码目录执行；先记录 `pwd`、当前分支和 `HEAD`，避免在错误源码副本中定位。

## 分级 curl 复现

curl 的 host、port 和 model 必须与服务启动参数一致。

```bash
curl -sS -X POST http://127.0.0.1:1025/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "请说明爱因斯坦的出生日期、出生地、两项主要科学贡献及其影响。使用四个编号段落，每段至少两句，总字数 150–250 字。"}],
    "model": "<model-name>",
    "max_tokens": 512,
    "stream": false,
    "temperature": 0.0
  }'
```

该示例至少要求完整句子和多个事实字段，不要退化为 `1+1` 或单 token 输出。它仍只是第一层。为了形成可比证据，保存请求 JSON、HTTP 状态、原始响应、服务日志和重复次数。

定位时按成本递增构造三层最小测试，通常不需要先跑全量：

1. **结构化事实 curl**：答案可人工核验，要求多个字段或 80–300 字输出，`max_tokens` 至少覆盖 128–512 token；重复 3 次检查确定性。
2. **长序列 curl**：构造或选取 2K–8K token 确定性上下文，要求从首部、中部、尾部抽取信息并生成 256–1024 token 输出。若问题涉及 MTP、prefix cache、图模式或状态缓存，加入共享长前缀的连续请求、不同输出长度、batch 增删/重排或小并发，确保执行到相关 decode/cache 路径。
3. **真实失败样本 curl**：从 TextVQA 等原始评测失败样本中选 1–10 个，保留实际图片/base64、问题、chat template、生成参数与预期答案。优先用这些样本验证 good/bad、特性开关和补丁。

长序列内容必须有确定的可验证答案，避免用开放式创作判断精度。记录 prompt token 数、completion token 数、重复次数、首个错误 token/字段；需要触发 batch 状态问题时记录请求到达和完成顺序。

当用户给出的精度信息不足时，不要等待完整数据集配置才开始：先运行上述分级 curl，并确认它是否能稳定体现 expected/actual 差异。若能：

1. 固定请求体、模型、生成参数、环境和重复次数；
2. 在 good/bad 环境中验证该 curl 具有区分力；
3. 将其作为特性拆分、commit 定位和修复验收的首要判据；
4. 修复后先通过该 curl，再运行原始 ais_bench 或其他复杂精度回归。

若简单 curl 无法复现，不应宣称问题不存在，也不要立即运行全量。先增加输入/输出长度、共享 prefix、重复 decode、batch 状态变化、小并发或真实失败样本；再运行 10–100 条有代表性的固定小样本集。只有修复候选已通过最小与小样本验证，才进入全量回归。

## ais_bench 环境

按 benchmark 仓库文档从仓库根目录安装，不要为省时使用 `--no-deps` 后再把缺失 Python 包误判成数据集缺失：

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark
pip3 install -e ./ --use-pep517
```

安装后先核验 evaluator 运行环境：

```bash
pip show ais_bench_benchmark datasets mmengine
pip check
python -c "import ais_bench, datasets, mmengine; print(ais_bench.__file__, datasets.__version__, mmengine.__version__)"
```

`ModuleNotFoundError: No module named 'datasets'` 表示 Hugging Face `datasets` Python 依赖未安装，不表示 TextVQA 数据集不存在。网络/安装已获一次性授权时，按 benchmark 的 requirements/安装元数据自动补齐依赖，不重复询问；不要逐个猜包安装，也不要无依据升级 NumPy、Torch、CANN 或 PTA。

镜像可能已经安装 `ais_bench_benchmark`，先检查：

```bash
pip show ais_bench_benchmark
```

记录 benchmark commit、数据集版本/校验值和配置文件改动。数据集一般放在 `ais_bench/dataset/` 下；不要在 good/bad 环境使用不同的数据副本。

## vLLM API 配置要点

普通精度测试通常编辑 `ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`，function call 使用 `vllm_api_function_call_chat.py`，流式性能使用 `vllm_api_stream_chat.py`。

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import (
    extract_non_reasoning_content,
)

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="<weight-path>",
        model="<model-name>",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="127.0.0.1",
        host_port=1025,
        url="",
        max_out_len=4096,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=False,
            chat_template_kwargs={"enable_thinking": False},
        ),
    )
]
```

对比前核对：

- `path` 指向相同权重，必要时记录权重校验值；
- `model`、`host_ip`、`host_port` 与服务一致；
- `max_out_len` 不截断预期答案；
- `batch_size` 是需要单变量验证的并发因素；
- 精度测试通常 `temperature=0.0`、`ignore_eos=False`；
- 性能测试常用 `ignore_eos=True`，不能把该设置直接用于精度结论；
- thinking/chat template 和后处理保持一致，并保存后处理前后的结果。

## 常用数据集命令

```bash
# GSM8K
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_noncot_chat_prompt --mode all --debug

# SuperGLUE BoolQ
ais_bench --models vllm_api_general_chat --datasets SuperGLUE_BoolQ_gen_0_shot_noncot_str --mode all --debug

# C-Eval
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_noncot_chat_prompt --mode all --debug --merge-ds

# TextVQA
ais_bench --models vllm_api_general_chat --datasets textvqa_gen --mode all --debug
ais_bench --models vllm_api_general_chat --datasets textvqa_gen_base64 --mode all --debug

# BFCL 精简集 / 全量集
ais_bench --models vllm_api_function_call_chat --datasets BFCL_gen_simple --debug
ais_bench --models vllm_api_function_call_chat --datasets BFCL_gen_all --debug
```

TextVQA 图片应有 3165 张（按原始资料），但数据集上游内容可能更新；使用当前数据集说明和校验结果确认完整性。非 base64 模式可能要求在服务启动工作目录下存在 `data/textvqa`。记录实际路径，不要盲目复制或覆盖用户数据。

运行命令前必须 `cd` 到 benchmark 仓库根目录，并使用文档中的模型配置名。除非用户原始基线明确包含额外参数，不要擅自添加 `--num-warmups 0`、自建模型配置名或改变 batch/output 设置。需要诊断配置时复制为新文件并记录 diff；最终全量回归恢复用户原始 `vllm_api_general_chat` 配置和原始命令。

## 定位与全量回归边界

- **定位阶段**：分级 curl、1–10 个失败样本、10–100 条固定小样本。避免反复跑 3165 张全量 TextVQA。
- **修复验收阶段**：最小样本和固定小样本通过后，必须执行原始 TextVQA 全量命令，保留总分、失败样本、raw output、评分文件和运行日志。
- 全量结果必须与标准值和波动阈值比较。例如标准 82、允许 1% 波动时，明确阈值采用百分点还是相对百分比，并沿用用户/项目口径。
- 全量未完成时只能报告“最小修复已验证”，不能报告“精度问题已修复完成”。仅当设备、数据或外部环境形成明确阻塞时暂停，并给出续跑命令与产物位置。

## 精度结果的最低证据标准

一次可审计的测试至少保留：

- 服务完整启动命令和日志；
- benchmark commit、配置、命令、数据集身份；
- 原始输出、后处理输出、评分文件；
- 总分及失败样本列表；
- 重复运行的稳定性；
- good/bad 环境中除目标变量外的差异。

如果总分变化来自少量样本，优先将这些样本转成最小 curl，检查 token 截断、EOS、思考内容清理、模板和评分后处理，再进入底层算子分析。
