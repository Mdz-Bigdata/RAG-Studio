---
title: K8s 运维排障手册
version: "1.3"
department: 基础架构部
tags: [运维, Kubernetes, 数据库, 安全]
---

# K8s 运维排障手册

## Ingress 故障

### K8s Ingress 502 排查手册

Ingress 返回 502 Bad Gateway 时，按以下顺序排查：

1. 先看 upstream 健康检查：`kubectl get endpoints <svc>` 确认 Service 后端有就绪 Pod，没有 Endpoint 说明 readinessProbe 失败；
2. 再看 backend timeout 配置：Nginx Ingress 的 `proxy-read-timeout` 默认 60 秒，后端慢请求会被判为 502；
3. 检查 Pod 是否在滚动更新中被提前摘除，建议配置 `preStop` sleep 5 秒；
4. 查看 ingress-nginx controller 日志中的 `upstream prematurely closed connection`。

### K8s 常见 5xx 错误汇总

- 500 内部错误：应用自身异常，查 Pod 日志；
- 503 服务不可用：Service 没有可用 Endpoint，或 Ingress 配置了不存在的后端；
- 504 网关超时：后端处理超过 `proxy-read-timeout`，考虑异步化或调大超时。

## Nginx 反向代理调优

keepalive 连接数、`proxy_read_timeout`、buffer size 是三大关键参数。建议 `keepalive 32`，`proxy_read_timeout 120s`，`proxy_buffer_size 16k`。

## 数据库

### MySQL 1062 主键冲突处理

错误码 1062（Duplicate entry）表示插入的主键或唯一索引已存在。三种方案对比：

| 方案 | 行为 | 适用场景 |
| --- | --- | --- |
| INSERT IGNORE | 冲突时静默跳过 | 幂等写入、允许丢弃 |
| REPLACE INTO | 先删后插 | 需要覆盖旧数据，注意自增主键变化 |
| ON DUPLICATE KEY UPDATE | 冲突时更新指定列 | 最常用，保留主键 |

### 数据库唯一约束设计

主键、唯一索引、复合唯一键各有适用场景。复合唯一键在高并发下容易出现间隙锁导致的死锁，建议配合重试。

## 安全

### CVE-2024-3094 XZ 后门事件复盘

liblzma 5.6.0 / 5.6.1 被植入供应链后门，影响 sshd。紧急修复步骤：降级 xz 到 5.4.x，重启 sshd，检查 `ldd $(which sshd)` 是否链接到受影响版本。

### 开源软件供应链安全实践

依赖锁定（lockfile）、SBOM 生成、签名验证（sigstore / cosign）是工程落地的三件套。

## 监控告警

### Prometheus 告警规则

P95 延迟、错误率、饱和度是三大黄金指标。建议阈值：P95 延迟 > 500ms 持续 5 分钟告警，错误率 > 1% 告警，CPU 饱和度 > 80% 告警。
