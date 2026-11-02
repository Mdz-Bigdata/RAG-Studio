# OKF 权威知识包 + RAG

OKF（Open Knowledge Format，开放知识格式，Google Cloud 2026-06）：把经过审核的权威知识——指标公式、API 规范、数据表定义、运维手册——
写成 Git 管理的 Markdown 文件，每个文件一个概念，文件头是 YAML front matter。它解决的是 RAG 的"概率性"短板：
向量检索找到的是"相似"的段落，不一定是"官方"的那一份；三条 Slack 消息对流失率的定义略有出入时，检索到哪条全看运气。

本系统把两者接在一起：**问题点名了某个条目，就直接取那个条目（确定性，不经过向量）；没点名，照常走 RAG。**

## 条目长什么样

```markdown
---
type: metric
title: "Monthly Churn Rate"
description: "Official formula for calculating monthly customer churn."
owner: "data-engineering"
tags: [revenue, kpi, board-report]
timestamp: 2026-06-20T10:00:00Z
aliases: [月流失率, 客户流失率]        # 本系统扩展：命中别名也算命中（中文问英文条目靠它）
---
# Monthly Churn Rate
The official churn rate formula used in all board reports and investor decks:
Churn Rate = (Customers Lost During Month / Customers at Start of Month) × 100
### Rules
- **Do NOT** use trial accounts in the denominator.
### Related
- [Monthly Active Users](mau.md)
```

没有 front matter、或 front matter 里没有 `title` 的 Markdown 不算条目（仓库里的 README 之类会被跳过）。
文件名也算一个别名（`monthly_churn.md` → "monthly churn"）。

## 导入

「知识库」页 →「OKF 知识包」→ 填来源：

- Git 地址（`https://…/knowledge.git`、`git@…`）：克隆到 `data/okf/<kb_id>/repo/`，可指定分支 / 标签和子目录；
- 服务器上的目录：直接扫描。

每个条目成为一篇带「OKF」标签的文档，走普通的解析 / 切片 / 索引（所以它们也参与 RAG 召回）。
知识包改了就「重新同步」：内容哈希没变的条目不动，变了的重新入库，仓库里删掉的从知识库删除——改动全在 Git 提交历史里（GitOps）。
接口：`GET/POST /api/v1/okf/{kb_id}`（状态 / `import` / `sync` / `match?q=`）。

## 检索时发生什么

1. 问题先在 OKF 目录（本库全部条目的标题 + 别名）上做确定性匹配：中文名按子串，拉丁名按整词；多个命中取名字最长的，最多 `retrieval.okf.max_pinned` 个。
2. 命中的条目**整篇置顶**（最多 `max_parents` 段，score 1.0，标记 `authority: okf`），后面接 RAG 正常召回、融合、精排的结果；没命中则与原来完全一样。
3. 交给模型的资料里，条目的来源行标注"【权威条目（OKF）】"并附类型 / 负责人 / 更新日期；回答协议多一条：问的就是这个条目时以它为准，
   其他资料说法不同时按权威条目回答并指出不一致。
4. 时效 / 权限过滤同样作用于条目：过期或无权查看的条目不会置顶。

检索实验室和问答的来源卡片上会显示"权威 · OKF"标记；检索链路说明里写明命中了哪个条目、按哪个名字命中。

## 边界

- 匹配靠名字：问题没有出现条目的标题或任何别名就不会置顶（"我们怎么算流失" 不会命中《Monthly Churn Rate》，除非加了别名"流失"）。
  别名宁多勿少；太短、太泛的别名（单字、"数据"）会误命中，目录要求名字至少 2 个字符。
- 条目本身也在 RAG 索引里，所以没点名时仍可能作为普通资料被召回，只是不带权威标记。
- 导入是同步执行的（条目通常很少、很短）；几千个条目的大知识包建议用 Git 分支分批。
