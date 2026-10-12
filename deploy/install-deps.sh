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
echo "[$(date +%T)] INSTALL_DONE"
