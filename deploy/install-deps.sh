#!/usr/bin/env bash
# 服务器上安装运行环境：uv 管理的 Python 3.13 + CPU 版 torch + 与本地一致的依赖版本（deploy/constraints.txt）。
# 可重复执行；已下载的包走 uv 缓存。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UV="${UV:-/root/.local/bin/uv}"
[ -x "$UV" ] || { /usr/bin/python3.11 -m pip install -q --user uv; }
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"          # 放在项目目录里，服务账号才读得到
export UV_DEFAULT_INDEX=http://mirrors.cloud.aliyuncs.com/pypi/simple/ UV_INSECURE_HOST=mirrors.cloud.aliyuncs.com UV_HTTP_TIMEOUT=300
[ -x .venv/bin/python ] || "$UV" venv --python 3.13 --python-preference only-managed .venv
echo "[$(date +%T)] torch (cpu)"
"$UV" pip install --python .venv/bin/python --compile-bytecode "torch==2.14.1" --default-index https://download.pytorch.org/whl/cpu
echo "[$(date +%T)] project deps"
"$UV" pip install --python .venv/bin/python --compile-bytecode -c deploy/constraints.txt -e '.[models,lancedb,lake,multimodal]'
# MinerU 4.x 基础包：没有 GPU 时小模型走 ONNX（CPU），不需要 [torch] 附加依赖
echo "[$(date +%T)] mineru"
"$UV" pip install --python .venv/bin/python --compile-bytecode -c deploy/constraints.txt 'mineru>=4.0,<5'
# 模型包（版面 / OCR / 公式 / 表格，约 0.8GB）下载到服务账号可写的目录；已下载过会直接跳过。
# standard / advanced 档还需要 VLM（约 2GB 模型、8GB 内存），按需执行：mineru-kit models download --tier standard
# 目录与 deploy/rag-studio.service 里的 MINERU_HOME / MODELSCOPE_HOME 保持一致
MINERU_ENV=(MINERU_HOME="$ROOT/.cache/mineru" MODELSCOPE_HOME="$ROOT/.cache/modelscope_home")
DOWNLOAD=("$ROOT/.venv/bin/mineru-kit" models download --tier basic --source modelscope)
mkdir -p "$ROOT/.cache/mineru" "$ROOT/.cache/modelscope_home"
if id ragstudio >/dev/null 2>&1; then
  chown -R ragstudio:ragstudio "$ROOT/.cache/mineru" "$ROOT/.cache/modelscope_home"
  runuser -u ragstudio -- env "${MINERU_ENV[@]}" "${DOWNLOAD[@]}"
else
  env "${MINERU_ENV[@]}" "${DOWNLOAD[@]}"   # 服务账号由 server.sh start 创建并接管 .cache
fi
echo "[$(date +%T)] INSTALL_DONE"
