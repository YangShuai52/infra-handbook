set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"   # 默认仅本机访问（当前为私有使用阶段）；需要局域网访问时 HOST=0.0.0.0 ./run.sh

# ---- 环境准备：venv 不存在或依赖缺失时自动创建并安装 ----
if [ ! -x ".venv/bin/python" ] || ! .venv/bin/python -c "import fastapi, uvicorn, httpx, yaml, markdown" 2>/dev/null; then
  echo "▶ 首次运行：创建虚拟环境并安装依赖…"
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r backend/requirements.txt
fi

echo "▶ 启动 LLM & AI Infra Handbook  (http://${HOST}:${PORT})"
cd backend
exec ../.venv/bin/uvicorn app:app --host "${HOST}" --port "${PORT}"
