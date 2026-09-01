# Ascend 310P 精度复现与测试

仅在需要启动服务、制作最小 curl 或运行 ais_bench 时读取本文件。

## 启动服务

先用当前版本的 `vllm serve --help` 核对参数。以下是诊断模板，不是可直接照抄的固定命令：

```bash
cd <vllm-ascend-source-path>
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

保存完整启动命令、环境变量和从进程启动到服务就绪的日志。
服务拉起、复现和源码 debug 均从用户确认的 vLLM-Ascend 源码目录执行；先记录 `pwd`、当前分支和 `HEAD`，避免在错误源码副本中定位。

## 最小 curl 复现

curl 的 host、port 和 model 必须与服务启动参数一致。

```bash
curl -sS -X POST http://127.0.0.1:1025/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "爱因斯坦生日是哪天"}],
    "model": "<model-name>",
    "max_tokens": 2048,
    "stream": false,
    "temperature": 0.0
  }'
```

为了形成可比证据，保存请求 JSON、HTTP 状态、原始响应、服务日志和重复次数。

当用户给出的精度信息不足时，不要等待完整数据集配置才开始：从用户描述、错误样本或已有评测结果中选择一个代表性输入，构造单 curl，并确认它是否能稳定体现 expected/actual 差异。若能：

1. 固定请求体、模型、生成参数、环境和重复次数；
2. 在 good/bad 环境中验证该 curl 具有区分力；
3. 将其作为特性拆分、commit 定位和修复验收的首要判据；
4. 修复后先通过该 curl，再运行原始 ais_bench 或其他复杂精度回归。

若单 curl 无法复现，不应宣称问题不存在。保留原始并发、batch、图模式、prefix cache、MTP 或评测后处理等可能触发条件，并逐步制作仍然足够小的复现。

## ais_bench 环境

基础安装示例：

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark
pip3 install -e ./ --use-pep517
```

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

## 精度结果的最低证据标准

一次可审计的测试至少保留：

- 服务完整启动命令和日志；
- benchmark commit、配置、命令、数据集身份；
- 原始输出、后处理输出、评分文件；
- 总分及失败样本列表；
- 重复运行的稳定性；
- good/bad 环境中除目标变量外的差异。

如果总分变化来自少量样本，优先将这些样本转成最小 curl，检查 token 截断、EOS、思考内容清理、模板和评分后处理，再进入底层算子分析。
