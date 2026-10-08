# MiniMax H3 在 Ascend 310P 上的安装与部署指南 (TP4)

## 一、环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | Atlas 300I DUO (310P)，4 张卡，每卡 48GB |
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.10.0+cpu（容器内置） |
| torch_npu | 2.10.0.post2（容器内置） |
| vllm-ascend | 容器内置或源码安装 |
| MindIE-SD | **不安装**（310P 不支持其算子，回退到 torch_npu 原生算子） |
| ffmpeg/ffprobe | PATH 可用 |
| 模型权重 | MiniMaxAI/MiniMax-H3（FL2VA 分区，~134GB） |

> TP4 下每卡权重约 34GB，加上激活/VAE/临时缓冲，48GB 卡显存紧张。如果 OOM，参考第四节降级方案。

---

## 二、容器准备

### 2.1 启动容器

使用 vllm-ascend 310P 镜像：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:latest-310p
export CONTAINER_NAME=h3-310p

docker run -it -u root --name ${CONTAINER_NAME} \
  --privileged=true \
  --shm-size=2000g \
  --net=host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /home:/home \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /path/to/MiniMax-H3:/workspace/MiniMax-H3 \
  ${IMAGE} \
  /bin/bash
```

> 将 `/path/to/MiniMax-H3` 替换为宿主机上 H3 权重的实际路径。如果没有预下载，容器内可通过 `hf download` 下载。

### 2.2 进入容器后初始化环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# 验证 NPU 可见
npu-smi info
python -c "import torch_npu; print('torch_npu:', torch_npu.__version__)"
python -c "import vllm_ascend; print('vllm_ascend:', vllm_ascend.__version__)"
```

确认 4 张卡可见：

```bash
python -c "import torch_npu; print('device_count:', torch_npu.npu.device_count())"
# 应输出 4
```

---

## 三、安装步骤

### 3.1 安装 ffmpeg

```bash
apt-get update && apt-get install -y ffmpeg
ffprobe -version  # 确认可用
```

### 3.2 MindIE-SD（跳过，310P 不需要）

**310P 不安装 MindIE-SD。** 原因：

1. MindIE-SD 的 AscendC 算子（`laser_attention`、`ada_block_sparse_attention` 等）不支持 310P 架构，编译会失败
2. MindIE-SD 的 attention 算子底层依赖 FIA（`npu_fusion_attention`），310P 不支持 FIA
3. vllm-omni 的所有 norm/rope/attention 路径都有 `find_spec("mindiesd") is None` 时的回退，不装也能跑

不装 MindIE-SD 时的回退链：

| 算子 | 无 MindIE-SD 时走 | 310P 可用 |
|------|------------------|----------|
| RMSNorm | `torch_npu.npu_rms_norm` | ✅ |
| AdaLayerNorm | `torch_npu.npu_layer_norm_eval` | ✅ |
| RoPE | `torch_npu.npu_rotary_mul` | ✅ |
| Attention | `TORCH_SDPA` → `F.scaled_dot_product_attention` | ✅ |
| SwiGLU | `torch_npu.npu_swiglu` | ✅ |
| indexed_modulation | torch eager（`not HAS_TRITON`） | ✅ |

### 3.3 安装 vllm-omni（含 310P 适配修改）

```bash
cd /workspace
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

# 如果已有本地修改过的 vllm-omni，直接使用；否则需要手动应用 310P 适配补丁
# （见下方 "310P 适配补丁" 章节）

# 设置目标设备为 NPU
export VLLM_OMNI_TARGET_DEVICE=npu

# 源码安装，复用容器中已有的 torch/torch_npu
pip install -e . --no-build-isolation

# 卸载 triton（310P 不支持）
pip uninstall -y triton triton-ascend 2>/dev/null || true

cd /workspace
```

### 3.4 验证安装

```bash
# 验证 mindiesd 不可用（预期，310P 不安装）
python -c "from importlib.util import find_spec; print('mindiesd:', find_spec('mindiesd'))"
# 应输出 None

# 验证 vllm_omni 可 import
python -c "import vllm_omni; print('vllm_omni OK')"

# 验证 NPU 平台和 310P 检测
python -c "
from vllm_omni.platforms import current_omni_platform
print('is_npu:', current_omni_platform.is_npu())
from vllm_omni.platforms.npu._310p import is_310p
print('is_310p:', is_310p())
"

# 验证 HAS_TRITON 为 False
python -c "
from vllm.triton_utils import HAS_TRITON
print('HAS_TRITON:', HAS_TRITON)
"
# 应输出 False
# 应输出 False
```

---

## 四、310P 适配补丁

vllm-omni 需要以下 4 处修改才能在 310P 上运行。如果使用已修改的代码仓则跳过；否则手动应用。

### 4.1 `vllm_omni/diffusion/layers/indexed_modulation.py`

将 `from vllm.triton_utils import tl, triton` 改为 `HAS_TRITON` 守卫；4 个 Triton kernel 放入 `if HAS_TRITON:` 块；4 个公开函数的回退条件从 `x.is_cpu` 改为 `x.is_cpu or not HAS_TRITON`。

### 4.2 `vllm_omni/platforms/npu/platform.py`

3 处加 `is_310p()` 守卫：
- mindiesd import 时跳过 310P
- 默认后端选择时 310P 返回 `TORCH_SDPA`
- `supports_diffusion_dense_flash_attention` 在 310P 上返回 False

### 4.3 `vllm_omni/diffusion/attention/backends/flash_attn.py`

`forward_npu` 加 310P 分支，新增 `forward_fa_310p` 方法：
- packed varlen 序列用 `torch_npu._npu_flash_attention_unpad`
- 非 packed 回退 `F.scaled_dot_product_attention`

### 4.4 `vllm_omni/platforms/npu/_310p/patch/minimax_h3.py`（新建）

310P H3 patch 注册点，在 `_310p/patch/__init__.py` 中注册 `MiniMaxH3DiTModel` arch。

---

## 五、下载模型权重

```bash
# 方式一：huggingface-cli 下载（需 HF 授权）
pip install -U huggingface_hub
hf auth login  # 输入 HuggingFace token
hf download MiniMaxAI/MiniMax-H3 --local-dir /workspace/MiniMax-H3

# 方式二：仅下载 FL2VA 分区（节省磁盘，~134GB）
export MODEL=/workspace/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/**" \
  --local-dir "${MODEL}"
```

> TP4 仅跑 T2VA/FL2VA 任务时只需 FL2VA 分区。如需 Ref2VA 需额外下载 Ref2VA/transformer。

---

## 六、启动服务 (TP4)

### 6.1 基础配置（推荐首次使用）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PORT=8091
export MODEL=/workspace/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --diffusion-attention-backend TORCH_SDPA \
  --enforce-eager
```

**参数说明**：

| 参数 | 值 | 说明 |
|------|------|------|
| `--task-type fl2va` | fl2va | 仅加载 FL2VA 分区，支持 T2VA 和 FL2VA |
| `--num-gpus 4` | 4 | 使用 4 张 310P |
| `--tensor-parallel-size 4` | 4 | TP4 切分（56 heads / 4 = 14 heads/卡） |
| `--diffusion-attention-backend TORCH_SDPA` | SDPA | 310P 无 FIA，用 PyTorch SDPA |
| `--enforce-eager` | - | 310P 不支持 torch.compile，禁用编译 |

### 6.2 显存不够时的降级配置

TP4 每卡权重 ~34GB，48GB 卡可能 OOM。启用 CPU offload：

```bash
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --diffusion-attention-backend TORCH_SDPA \
  --enforce-eager \
  --enable-cpu-offload
```

### 6.3 性能优化配置（FLASH_ATTN + 310P 原生算子）

如果 SDPA 性能可接受且验证正确后，可切换到 FLASH_ATTN 后端（走 `_npu_flash_attention_unpad`）：

```bash
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --diffusion-attention-backend FLASH_ATTN \
  --enforce-eager
```

> FLASH_ATTN 后端在 310P 上会走 `forward_fa_310p`，使用 `torch_npu._npu_flash_attention_unpad` 处理 packed varlen 序列，比 SDPA 不需要 materialize 完整 mask，显存更省、性能更好。

---

## 七、发送推理请求

### 7.1 T2VA：文本生成视频+音频

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

# 小分辨率冒烟测试（推荐首次验证）
curl -sS -X POST "${API_URL}" \
  -F 'prompt=A cat walking in a snowy forest, with crunching footsteps.' \
  -F 'width=672' \
  -F 'height=384' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=5' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}' \
  -o t2va_test.mp4

# 标准分辨率（验证通过后使用）
curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

### 7.2 FL2VA：首帧生成视频+音频

```bash
export FIRST_FRAME=/path/to/first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A man stands beside a yellow car at night. The car drives away.' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -o fl2va.mp4
```

### 7.3 验证输出

```bash
# 检查 MP4 是否可解码
ffprobe t2va_test.mp4

# 应看到类似输出：
# Stream #0:0: Video: h264 ... 672x384 ... 24 fps
# Stream #0:1: Audio: aac ... 32000 Hz, stereo, fltp
```

---

## 八、显存监控与排障

### 8.1 监控显存

```bash
# 实时监控每张卡
watch -n 1 'npu-smi info | grep -E "Chip|Memory|310P"'

# 或查看详细
npu-smi info -l  # 列出所有卡
npu-smi info -t board -i 0  # 查看卡 0 详情
```

### 8.2 常见问题

#### OOM：CUDA/NPU out of memory

TP4 每卡 34GB 权重 + 激活可能超 48GB：

```bash
# 方案 1：加 CPU offload
--enable-cpu-offload

# 方案 2：降分辨率
-F 'width=672' -F 'height=384'

# 方案 3：降帧数
-F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}'

# 方案 4：减步数（牺牲质量）
-F 'num_inference_steps=5'
```

#### import 失败：triton 相关错误

```bash
# 确认 triton 已卸载
pip uninstall -y triton triton-ascend

# 确认 HAS_TRITON 为 False
python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"
```

#### mindiesd 相关错误

310P 不安装 MindIE-SD，如果代码中仍有 mindiesd 相关报错：

```bash
# 确认 mindiesd 未安装
python -c "from importlib.util import find_spec; print(find_spec('mindiesd'))"
# 应输出 None

# 如果误装了，卸载
pip uninstall -y mindiesd
```

#### 注意力报错：FIA not supported

```bash
# 确认使用了 TORCH_SDPA 或适配后的 FLASH_ATTN
# 检查 is_310p() 返回 True
python -c "from vllm_omni.platforms.npu._310p import is_310p; print(is_310p())"
```

#### TP4 不整除 heads

```
ValueError: num_attention_heads must be divisible by tensor_parallel_size: 56 % X != 0
```

56 的因数：1, 2, 4, 7, 8, 14, 28, 56。TP4 = 4，`56 % 4 == 0` ✅。

---

## 九、310P 适配注意事项

1. **Triton 完全不可用** — 310P 的 CANN 不提供 Triton-Ascend，所有 `@triton.jit` kernel 需有 `HAS_TRITON` 守卫和 torch eager 回退。

2. **FIA 不支持** — 310P 不支持 `npu_fusion_attention`（FIA），MindIE-SD 的 attention 算子不可用，必须用 `torch_npu._npu_flash_attention_unpad` 或 `F.scaled_dot_product_attention`。

3. **仅支持 TP** — 310P 不支持 Ulysses SP / Ring Attention / Pipeline Parallelism。`--usp` 和 `--ring` 必须保持默认值 1。

4. **无 torch.compile** — 310P 不支持 `torch.compile`，必须加 `--enforce-eager`。

5. **RC 设备** — 310P 是 RC 设备，host/device 共享内存，CPU offload 的 H2D 成本较低。

6. **BF16 待验证** — `fused_qk_norm_rope` 的 NPU 路径要求 BF16，不支持时自动回退到 eager。

7. **MindIE-SD 编译目标** — 构建 MindIE-SD wheel 时必须设 `ASCEND_COMPUTE_UNIT="ascend310p"`，否则编译的算子不含 310P 实现。

8. **不改 MindIE-SD 和 vllm-ascend** — 适配工作仅修改 vllm-omni，通过回退路径绕过 310P 不支持的算子。

---

## 十、快速验证清单

```bash
# 1. 环境检查
npu-smi info                                    # 4 张卡可见
python -c "import torch_npu; print(torch_npu.npu.device_count())"  # 输出 4

# 2. 软件检查
python -c "import vllm_omni; print('OK')"        # vllm_omni 可 import
python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"  # False

# 3. 310P 检测
python -c "from vllm_omni.platforms.npu._310p import is_310p; print(is_310p())"  # True

# 4. 启动服务（小分辨率冒烟）
# 见第六节 6.1

# 5. 发送请求
# 见第七节 7.1 小分辨率测试

# 6. 验证输出
ffprobe t2va_test.mp4  # H.264 + AAC
```
