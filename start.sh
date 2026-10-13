#!/usr/bin/env bash
# ==============================================================================
# RAG Agent Studio 一键启动：FastAPI 后端 + Vite 前端
#
#   ./start.sh            开发模式（后端热重载 + vite dev server）
#   ./start.sh --prod     构建前端并由后端单端口托管
#
# 在已经用 deploy/server.sh 装成 systemd 服务的服务器上，直接执行 ./start.sh 等同于
# deploy/server.sh start（启动服务 + Nginx，设为开机自启），不需要 Node / npm。
#
# 端口：BACKEND_PORT（默认 8000）/ FRONTEND_PORT（默认 5173），被占用时自动顺延。
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT/frontend"
VENV="$ROOT/.venv"
MODE="${1:-dev}"

# 服务器部署：这个目录已注册为 systemd 服务时交给部署脚本，不走开发模式
if [ "$MODE" = "dev" ] && grep -qsx "WorkingDirectory=$ROOT" /etc/systemd/system/rag-studio.service; then
  exec "$ROOT/deploy/server.sh" start
fi

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
say() { echo -e "${BLUE}[$1]${NC} $2"; }

[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

free_port() {  # 从 $1 开始找第一个未被监听的端口
  local port="$1"
  while lsof -iTCP:"$port" -sTCP:LISTEN -P -n >/dev/null 2>&1; do port=$((port + 1)); done
  echo "$port"
}

# 1. 配置文件
if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  say 配置 "已生成 .env（留空即离线体验模式；也可以在前端「模型配置」页配置模型）"
fi

# 2. Python 依赖
if [ ! -x "$VENV/bin/python" ]; then
  say 后端 "创建虚拟环境并安装依赖…"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" && uv pip install --python "$VENV/bin/python" -e "$ROOT[dev]"
  else
    python3 -m venv "$VENV" && "$VENV/bin/pip" install --upgrade pip && "$VENV/bin/pip" install -e "$ROOT[dev]"
  fi
elif ! "$VENV/bin/python" -c "import fastapi, sqlalchemy, uvicorn" >/dev/null 2>&1; then
  say 后端 "补装 Web 依赖…"
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$VENV/bin/python" -e "$ROOT[dev]"; else "$VENV/bin/pip" install -e "$ROOT[dev]"; fi
fi

# 3. 前端依赖
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  say 前端 "npm install…"
  (cd "$FRONTEND_DIR" && npm install)
fi

BACKEND_PORT="$(free_port "${BACKEND_PORT:-8000}")"
trap 'kill 0' EXIT
cd "$ROOT"

if [ "$MODE" = "--prod" ]; then
  say 前端 "构建生产包…"
  (cd "$FRONTEND_DIR" && npm run build)
  echo -e "${GREEN}✓ 访问 http://localhost:${BACKEND_PORT}（API 文档 /docs）${NC}"
  "$VENV/bin/uvicorn" app.main:app --host 0.0.0.0 --port "$BACKEND_PORT"
else
  FRONTEND_PORT="$(free_port "${FRONTEND_PORT:-5173}")"
  "$VENV/bin/uvicorn" app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload --reload-dir app --reload-dir prism &
  (cd "$FRONTEND_DIR" && VITE_API_TARGET="http://localhost:${BACKEND_PORT}" npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" --strictPort) &
  echo -e "${GREEN}====================================================${NC}"
  echo -e "${GREEN}✓ 前端控制台: http://localhost:${FRONTEND_PORT}${NC}"
  echo -e "${GREEN}✓ 后端 API 文档: http://localhost:${BACKEND_PORT}/docs${NC}"
  echo -e "${YELLOW}按 Ctrl+C 退出并关闭所有端口${NC}"
  echo -e "${GREEN}====================================================${NC}"
  wait
fi
