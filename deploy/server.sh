#!/usr/bin/env bash
# ==============================================================================
# RAG Agent Studio 服务器端一键脚本（以 root 执行）
#
#   ./deploy/server.sh start     安装 systemd / Nginx 配置 → 启动 → 设为开机自启 → 等待就绪
#   ./deploy/server.sh stop      停止（不取消开机自启）
#   ./deploy/server.sh restart   重启后端并重载 Nginx
#   ./deploy/server.sh status    服务状态、端口、内存
#   ./deploy/server.sh logs      跟随后端日志（Ctrl+C 退出）
#   ./deploy/server.sh passwd [用户名]   开启网页访问密码并设置 / 修改密码（默认用户名 admin）
#   ./deploy/server.sh open      取消网页访问密码，任何人打开网址即可访问
#
# 首次部署先执行 ./deploy/install-deps.sh 安装依赖。
# ==============================================================================
set -euo pipefail

# 经 /usr/local/bin/rag-studio 这个软链接调用时也要找到真实的项目目录
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
UNIT=rag-studio
RUN_USER=ragstudio
HTPASSWD=/etc/nginx/rag-studio.htpasswd
AUTH_INC=/etc/nginx/rag-studio-auth.inc
HEALTH=http://127.0.0.1:8000/api/v1/system/status
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
say() { echo -e "${GREEN}[rag-studio]${NC} $*"; }
warn() { echo -e "${YELLOW}[rag-studio]${NC} $*"; }

[ "$(id -u)" = "0" ] || { echo "请用 root 执行"; exit 1; }

install_configs() {
  [ -x "$ROOT/.venv/bin/uvicorn" ] || { echo -e "${RED}没有找到运行环境，请先执行 $ROOT/deploy/install-deps.sh${NC}"; exit 1; }
  id "$RUN_USER" >/dev/null 2>&1 || useradd --system --home-dir "$ROOT" --shell /sbin/nologin "$RUN_USER"
  # 代码保持 root 所有，服务账号只能写数据和模型缓存
  mkdir -p "$ROOT/data" "$ROOT/.cache"
  chown -R "$RUN_USER:$RUN_USER" "$ROOT/data" "$ROOT/.cache"
  [ -f "$ROOT/.env" ] && chown "root:$RUN_USER" "$ROOT/.env" && chmod 640 "$ROOT/.env"
  sed "s|@ROOT@|$ROOT|g" "$ROOT/deploy/rag-studio.service" > "/etc/systemd/system/$UNIT.service"
  install -m 644 "$ROOT/deploy/nginx-rag-studio.conf" /etc/nginx/conf.d/rag-studio.conf
  # 设过密码就要求登录，没设就直接放行
  if [ -s "$HTPASSWD" ]; then
    printf 'auth_basic "RAG Agent Studio";\nauth_basic_user_file %s;\n' "$HTPASSWD" > "$AUTH_INC"
  else
    echo "# 未设置访问密码：不做登录验证" > "$AUTH_INC"
  fi
  systemctl daemon-reload
}

wait_ready() {
  say "等待后端就绪（首次启动要加载向量模型，可能需要一两分钟）…"
  for _ in $(seq 1 40); do
    if curl -fsS -m 15 "$HEALTH" >/dev/null 2>&1; then say "后端已就绪"; return 0; fi
    systemctl is-active --quiet "$UNIT" || { echo -e "${RED}后端启动失败，最近的日志：${NC}"; journalctl -u "$UNIT" -n 30 --no-pager; return 1; }
    sleep 3
  done
  warn "等待超时，后端仍在启动中；用 $0 logs 查看进度"
}

status() {
  echo "后端  : $(systemctl is-active $UNIT)（开机自启：$(systemctl is-enabled $UNIT 2>/dev/null || echo no)）"
  echo "Nginx : $(systemctl is-active nginx)（开机自启：$(systemctl is-enabled nginx 2>/dev/null || echo no)）"
  echo "健康检查: $(curl -s -o /dev/null -m 5 -w '%{http_code}' "$HEALTH" || true)"
  echo "访问密码: $([ -s "$HTPASSWD" ] && echo 已开启 || echo "未开启，任何人打开网址即可访问（要加密码执行 $0 passwd）")"
  pid=$(systemctl show -p MainPID --value "$UNIT"); [ "${pid:-0}" != "0" ] && echo "后端内存: $(ps -o rss= -p "$pid" | awk '{printf "%.0f MB", $1/1024}')"
  free -h | sed -n 1,3p
}

case "${1:-start}" in
  start)
    install_configs
    systemctl enable "$UNIT" nginx >/dev/null 2>&1
    systemctl restart "$UNIT"
    nginx -t >/dev/null 2>&1 && systemctl restart nginx || { echo -e "${RED}Nginx 配置检查未通过：${NC}"; nginx -t; }
    wait_ready || true
    status
    say "访问地址：http://$(curl -s -m 3 http://100.100.100.200/latest/meta-data/eipv4 2>/dev/null || hostname -I | awk '{print $1}')/"
    ;;
  stop)     systemctl stop "$UNIT"; say "已停止" ;;
  restart)  install_configs; systemctl restart "$UNIT"; nginx -t >/dev/null 2>&1 && systemctl reload nginx; wait_ready || true; status ;;
  status)   status ;;
  logs)     journalctl -u "$UNIT" -f -n 100 ;;
  passwd)
    command -v htpasswd >/dev/null || dnf install -y -q httpd-tools
    if [ -f "$HTPASSWD" ]; then htpasswd "$HTPASSWD" "${2:-admin}"; else htpasswd -c "$HTPASSWD" "${2:-admin}"; fi
    chown root:nginx "$HTPASSWD"; chmod 640 "$HTPASSWD"
    install_configs; nginx -t >/dev/null 2>&1 && systemctl reload nginx
    say "已开启访问密码，用户名 ${2:-admin}"
    ;;
  open)
    rm -f "$HTPASSWD"
    install_configs; nginx -t >/dev/null 2>&1 && systemctl reload nginx
    say "已取消访问密码"
    ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac
