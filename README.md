# RAG Agent Studio

> 检索质量 ≈ 切块质量 × 检索策略 × 排序质量 × Query 质量（乘法关系，短板拖全盘）。

把 8 篇 RAG 实战文章总结出的方法论落成一套**可运行、可观察、可评测**的全栈参考实现：

**MinerU 4.x 解析 → MinerU-Popo 文档树 → 清洗 → 语义感知切片 + Parent-Child → 向量 + BM25 + HyDE 多路召回 → RRF 融合 → Reranker 精排 → Small-to-Big → 回答协议 → 评测闭环**，
外加语料覆盖分析（意图 × 知识）、语料治理（质量体检 / 覆盖缺口 FAQ / 别名与关键词扩展 / 自动元数据）、MMR、UnWeaver 实体织网、上下文压缩、多跳 / Agentic RAG、跨库查询路由。

方法论全文：[docs/RAG_OPTIMIZATION_SUMMARY.md](docs/RAG_OPTIMIZATION_SUMMARY.md)。

元数据过滤：[从问题里识别年份 / 部门 / 标签 / 文件类型的限定，召回前先缩小文档范围](docs/METADATA_FILTERING.md)。

难负样本优化：[挖掘、审核、真实 embedding / reranker 微调与独立评测](docs/HARD_NEGATIVE_FINETUNING.md)。
`rag mine-negatives` 从生产索引生成训练集，用重排模型剔除伪正例、简单负例与疑似漏标正例，并导出教师分供 `--distill-weight` 蒸馏，`--synthetic` 可让文本模型改写正例生成待人工筛选的负例；`python -m prism.training.finetune --dry-run ...` 可无模型预检查，安装 `.[finetune]` 后训练并保存本地权重。新 embedding 激活前需要重建向量索引。

## 快速开始

```bash
./start.sh            # 后端 FastAPI + 前端 Vite；端口被占用时自动顺延，地址见终端输出
./start.sh --prod     # 构建前端，由后端单端口托管
```

不配置任何模型也能完整操作（离线体验模式：哈希向量 + BM25 + 词法精排 + 抽取式回答）。
要获得真实语义，在前端「模型配置」页添加并激活模型，或复制 `.env.example` 为 `.env` 填写密钥。

手动启动：

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
rag serve --port 8000 --reload               # 或 uvicorn app.main:app --port 8000 --reload
cd frontend && npm install && npm run dev    # http://localhost:5173（/api 代理到 8000）
pytest                                       # 39 个测试，无需模型与网络
```

## 页面（编号即调优顺序）

| # | 页面 | 做什么 |
| --- | --- | --- |
| — | 总览 | 六项核心检索能力的开关与当前值、离线建库 / 在线检索管线、运行时模型、漏斗 |
| 01 | 知识库 | 上传 → 解析 → 清洗 → 切片 → 索引；切片预览（Parent-Child）、Markdown 校对回灌、元数据（版本 / 生效区间 / 标签）、实体织网、解析能力矩阵、MinerU-Popo 流程、重新解析 |
| — | 飞书文档 | 飞书 / Lark 文档（docx、知识库 wiki 链接）同步进知识库（按版本号增量）；在页面上查看、新建、追加、整篇替换、查找替换、删除飞书文档；文档智能体用自然语言查询 / 分析 / 修改飞书文档（写操作默认需确认） |
| 02 | 语料覆盖 | 意图空间 × 知识空间：每条意图分到 高匹配 / 部分匹配 / 无匹配，给出缺失关键词；找出未被任何意图命中 / 已过期的文档 |
| 03 | 语料治理 | 质量体检（乱码 / 目录页 / 重复 / 近似重复 / 多版本同时生效…）与按内容指纹剔除；为覆盖缺口生成 FAQ 草稿、人工审核后发布；别名词典与从语料发现；自动元数据复核 |
| 04 | 检索实验室 | 查询层 → 各路召回 → 融合 → 精排 → MMR → 最终上下文 → 织网 / 多跳，逐层可见；右侧按请求覆盖任意参数做 A/B |
| 05 | 图文检索 | 文搜图 / 图搜图 / 图文混合 / 图搜文，各路召回逐路可见；多模态能力状态（图文向量 / VLM / OCR）与知识库图片库 |
| 06 | 智能问答 | SSE 流式、引用可点击定位、回答带图片（[图n]）、上传图片以图搜图 / 以图搜文 / 看图问答、检索链路、多轮指代补全、临时文档、语音输入、自动路由（路由选中的库答不上——没资料或精排最高分低于阈值——时自动换排名后面、分数达标的库，页脚显示「路由兜底」；手动选的库没检索到资料时提示可能有答案的库，一键换库重问） |
| 07 | 评测中心 | 黄金集 → 分阶段消融（仅向量 / 仅 BM25 / 融合 / 完整链路）→ Hit / MRR / NDCG → 错误归因 → 历史报告 |
| 08 | Agent 湖 | 问答链路沉淀的上下文资产（问题 / 检索日志 / 回答 / 反馈 / 工具调用 / 记忆 / 黑板 / 文档）：概览、知识缺口、问题主题、湖检索、只读 SQL、分析智能体；版本 · Tag · 分支 · 派生列 |
| — | LanceDB | LanceDB 控制台：全局库的表 / 版本 / 索引；知识库向量（后端或一键镜像）上的向量 / 精确 / 全文 / 混合检索与时间旅行、查询计划、ANN 索引、optimize；图文同表（CLIP 向量 + 缩略图） |
| — | 长期记忆 | 按用户的记忆：手动添加 / 从一段话抽取、召回测试（相似度 × 新近度 × 重要度）、编辑、遗忘清理；问答时自动召回与抽取 |
| — | 多智能体 | 规划 → 并行研究 → 分析 → 审校（证据不足改写重查）→ 撰稿，实时时间线、证据面板、运行回放（LanceDB 黑板） |
| — | 个性化推荐 | 候选条目同步、带理由的推荐、点赞 / 点踩 / 相似条目、兴趣画像、模拟提问 |
| — | 模型配置 | 文本 / 嵌入 / 重排 / 多模态 / 图文向量五类模型自由组合；Anthropic SDK、OpenAI 兼容端点、本地 sentence-transformers、Rerank API；拉取模型列表（本地模型扫描本机 HuggingFace 缓存）、连接测试；文本嵌入模型可在卡片里就地配置 |
| — | 策略配置 | 场景预设与"瓶颈 → 手段"对照；MinerU 档位动态加载；解析、清洗、切片、索引增强、查询层、召回与融合、精排、上下文工程、织网 / 多跳 / Agentic、生成的全部参数 |

## 六项核心检索能力

| 能力 | 解决什么 | 代码 | 关键参数（`config.yaml` → `retrieval.*`） |
| --- | --- | --- | --- |
| 向量 + 关键词混合检索 | 语义灵活性 + 字面精确性（编号 / 错误码 / 型号） | `index/vector_store.py` `index/bm25_store.py` `retrieval/retriever.py::_recall` | `top_n_per_route: 50` |
| 查询重写 / 扩展 | 口语、指代不清、提问角度与文档对不上 | `retrieval/query_transform.py` | `query_transform.rewrite` `multi_query: 3` `decompose` |
| 假设文档嵌入 HyDE | 问题与文档文体不同；短 query | `retrieval/query_transform.py` | `query_transform.hyde` `skip_hyde_on_precise_tokens` |
| 不同粒度检索 | 小块检索保精度，大块阅读保上下文 | `ingest/chunking/parent_child.py` `retrieval/granularity.py` | `chunking.child_tokens: 150` `parent_max_tokens: 600` `granularity.*` |
| RRF 融合 | 向量与 BM25 分数量纲不可比 | `retrieval/fusion.py` | `fusion.method: rrf` `rrf_k: 60` `weight_sparse` |
| ReRanker 精排 | 粗召"相关但不最相关" | `retrieval/rerank.py` | `rerank.candidates: 50` `keep_n: 10` `top_k: 5` `min_score` |

查询层的耗时：改写 / Multi-Query / 分解 / HyDE 互不依赖，并发调用；查询改写、HyDE、text-to-SQL、路由这些辅助调用一律关闭模型思考
（推理模型会把 `max_tokens` 全花在思考上，改写和 HyDE 直接返回空）。`generation.llm.aux_model`（策略配置 → 生成 → 辅助模型）
可以把辅助调用换成同一端点上的快速模型，端点的模型列表里没有它时自动沿用回答模型。示例库实测（DeepSeek）：
串行 + 思考 9~12 秒 → 并发 + 关思考 1~3 秒 → 辅助模型换 deepseek-flash 后 0.7~1.1 秒，黄金集检索质量不降。
回答生成同理：`generation.llm.thinking: auto`（策略配置 → 生成 → 回答前思考）让普通问答不思考、只有多跳 / Agentic 模式先思考。
deepseek-v4-pro 实测普通问答首字 3.4~5.5 秒 → 0.5~1.3 秒、生成 4.2~6.6 秒 → 1.5~2.1 秒，回答同样正确并带引用。

## 架构

```
文档 ─▶ 解析 ─▶ (Popo 文档树) ─▶ 清洗 ─▶ 结构 / 语义切片 ─▶ Parent-Child ─▶ 增强 ─▶ 索引
       MinerU 4.x / Markdown /    跨页表格合并     页眉页脚 / 断行    标题面包屑        child 150 tok    上下文摘要    向量 + BM25
       Excel 行级 / docx / pptx   段落续接          乱码 / 近重复      表格代码原子块    parent 600 tok   假设性问题    + 实体图

用户 Query
  │ 查询层   多轮归并 / 改写 / Multi-Query / 分解 / HyDE（精确 token 自动跳过）   retrieval/query_transform.py
  ├─ 向量检索（每条查询 top-50）                                                   index/vector_store.py
  ├─ BM25（jieba + 精确 token 保护，top-50）                                       index/bm25_store.py
  ├─ HyDE 假设文档 → 向量检索
  └─ SQL 结构化检索（Excel / CSV → SQLite；只读 text-to-SQL 或规则兜底）            index/table_store.py
     各路并发执行；多查询 × 多路一起做 RRF（RAG-Fusion）
  ▼ 融合层   RRF k=60（或凸组合 α=0.5）→ 候选 ≥ 50                                 retrieval/fusion.py
  ▼ 精排层   Cross-Encoder / Rerank API / LLM / 词法 → 10                          retrieval/rerank.py
  ▼ 多样化   MMR λ=0.7（可选）                                                     retrieval/diversify.py
  ▼ 粒度层   child → parent、相邻合并、Lost-in-the-Middle 排序 → 3~5               retrieval/granularity.py
  ▼ 织网层   UnWeaver 术语定义织入（可选）                                         graph/unweaver.py
  ▼ 压缩层   规则 / LLM 上下文压缩（可选）                                         retrieval/compress.py
  ▼ 生成层   回答协议：依据 / 拒答 / 冲突 / 安全 / 格式 + 引用                      generation/
```

## 多模态 RAG：文搜图 / 图搜图 / 图文混合 / 图搜文 / 带图回答

```
入库   PDF / Word / PPT / Markdown / 图片文件 ──▶ 抽取图片（记录页码、图注、邻近正文；过滤图标与每页重复的 logo）
       ──▶ OCR（图中精确文字）+ VLM（语义描述）──▶ 一张图三份索引：
           图文向量（CLIP 图像塔）· "图片文本摘要"的文本向量 · "图片文本摘要"的 BM25；图片挂到所在的 parent 块（图文关联）

文搜图   text ─▶ CLIP 文本塔→图片向量 ＋ 描述向量 ＋ 描述 BM25 ─▶ RRF
图搜图   image ─▶ 图像塔→图片向量（余弦相似度，可设阈值）
图文混合 text + image ─▶ score = α·z(text·I) + (1-α)·z(image·I)（标准化消除模态间隙），再与描述两路按 α 加权
图搜文   image ─▶ 图文向量直达正文 ＋ 相似图片所在段落 ＋ OCR / VLM 转文字再检索 ─▶ RRF ─▶ Small-to-Big
端到端   检索（文本上下文 + 图片）─▶ 提示词带 <图片资料>，前 N 张图以像素交给 VLM ─▶ 回答用 [n] 引文字、[图n] 引图片
```

| 能力 | 默认 | 增强 |
| --- | --- | --- |
| 图文向量 | `auto`：装了 torch + transformers 用 Chinese-CLIP（首次使用下载权重，国内可设 `HF_ENDPOINT=https://hf-mirror.com`）；否则离线感知特征——只支持图搜图，文搜图走描述文本两路 | 「模型配置 → 图文向量」激活本地 CLIP 或 Jina 兼容 API；不想下载权重就把 `multimodal.embedding.provider` 设为 `perceptual` |
| 图中文字 | 无 | `uv pip install rapidocr-onnxruntime`（CPU 可跑）；精确数值以 OCR 为准，不让 LLM 估读 |
| 图片描述 / 看图回答 | 只用图注与上下文 | 「模型配置 → 多模态」激活 VLM；描述按图片内容哈希缓存 |

```bash
rag image-search "各区域销售额柱状图"                    # 文搜图
rag image-search --image query.png                       # 图搜图
rag image-search "同样的图表，要 2024 年的" -i q.png     # 图文混合
rag image-search --image query.png --to-text             # 图搜文
```

Web：「图文检索」页四种方式逐路可见；「智能问答」直接提问会带出相关图片（不相关的问题不挂图），点回形针上传图片即以图搜图 / 以图搜文。
代码在 `prism/multimodal/`，参数见 `config.yaml` 的 `multimodal` 段。

**规模压测**（合成语料：柱状图 / 折线图 / 表格截图 / 工单卡片，Chinese-CLIP 跑 CPU + rapidocr + bge-m3，不含 VLM 描述）：

| 规模 | 入库 | 文搜图（标题 / 图中编号） | 以图搜图（压缩 / 模拟翻拍） | 以图搜文（翻拍） | 回答附图 | 延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| 300 张 | 72s（0.24s/张，OCR 占 59%） | 100% / 100% | 100% / 100% | 100% | 30/30 | 文搜图 ~55ms，以图搜图 ~0.25s，回答附图 ~0.5–0.8s |
| 1000 张 | 261s（0.26s/张） | 100% / 100% | 100% / 100% | 100% | 30/30 | 同上，基本不随规模增长 |
| 1000 张 · LanceDB 后端 | 231s（0.23s/张） | 100% / 100% | 100% / 100% | 100% | 30/30 | 文搜图 ~65ms（比 numpy 慢约 10ms），其余持平；索引 62MB |

入库耗时线性增长；峰值内存约 3.3GB（主要是模型，不随数据量明显增长）。开启 VLM 图片描述每张再加约 1 秒（API 调用）。OCR 并行没有收益：onnxruntime 单张图已用满所有核，多线程反而更慢（2 线程 0.96×、4 线程 0.72×），所以保持串行。
压测发现并修掉的问题：按图中编号文搜图 32%、手机翻拍以图搜图 50% / 以图搜文 73%、按完整标题 88%——见上一节"编号优先""图中文字辅助认图"。

**真实文档**（CLIP 论文 arXiv 2103.00020，48 页英文双栏，MinerU flash 抽出 43 张图表：图 1~22 全部、表格大多数）。
人工写的 26 条文搜图 + 18 条问答（中文跨语言提问为主，不照抄图注；8 条与图无关的负例）：

| 配置 | 文搜图 hit@1 / hit@3 / hit@5 | 回答附图 精确率 / 召回率 / 负例不附图 |
| --- | --- | --- |
| 修复前，无 VLM | 0.58 / 0.62 / 0.69 | 0.19 / 0.65 / 0.12 |
| 修复后，无 VLM | 0.65 / 0.81 / 0.85 | 0.05 / 0.10 / 1.00 |
| 修复后，无 VLM + 跨语言补译 | 0.73 / 0.85 / 0.85 | 0.67 / 0.75 / 1.00 |
| 修复后，deepseek-flash 图片描述 | **0.85 / 0.92 / 0.96** | **0.78 / 0.85 / 1.00** |

「从图片库生成」的自动样例在修复前就接近满分——自动样例测不出跨语言、近义改写这类问题，上线前务必用人工问答对评一次。
这份文档暴露并修掉的问题：

- **VLM 描述被丢弃**：MinerU flash 档给每张图附带图中文字的英文转写，原逻辑"解析器有描述就不用 VLM"把中文语义描述整段扔掉了。现在 VLM 描述在前、转写在后，两者都保留（已建好的知识库需重新入库才能受益）。
- **中文问英文图注时关键词路是噪声**：只能靠每张图都有的"CLIP"打分。现在按"查询信息量有多少落在图片文本词表里"给关键词路降权。
- **图文向量证据整体偏高**：英文学术图表遇到任何学术类提问都比中性基线高 1.3~1.6 个门槛，与图无关的问题照样挂图。现在再减去"各图超出量的中位数"，只认比别的图更贴近问题的那部分（图片少于 8 张时不做这一步）。
- **按图号 / 表号找不到**："Figure 20""表 2""图3"分词后只剩普通数字，三路都排不进前 5。现在图注以该编号开头的图排最前，问答里点名的图号也直接算作强证据。

- **以图搜文先给图所在的段落**：查询图被认定是库里的某张图、又没附带问题时，它所在的段落排第一。原来这一路在融合里只占一票，还会在上下文预算里被整段裁掉（该论文 hit@1 0.75 → 0.90）。相似图有好几张时只认第一张。
- **自动样例的关键词挑得太弱**：英文文档会选出 "and""to"。现在按出现过的正文块数从少到多挑，比对时不区分大小写。

- **跨语言补译**（`multimodal.cross_lingual`，默认开）：问题和图片文字不是同一种语言、关键词路又几乎匹配不上时，用辅助模型把问题译成图片文字的语言补充检索，附图的关键词证据也按译文再算一遍。同语言或词已对得上时不调用模型；每个问题首次约 0.6–0.9s，之后走缓存。不开 VLM 时，该论文的文搜图 hit@1 0.65 → 0.73，回答附图精确率 / 召回率 0.05 / 0.10 → 0.67 / 0.75，负例仍全部不附图；开着 VLM 时不触发，结果不变。反过来用英文问中文图表库也有效。
另外两点：Chinese-CLIP 在这类图表上的文搜图基本是噪声（单路 hit@1 0.12），图表为主的库可把 `multimodal.weight_clip` 调到 0.5 左右（实测文搜图 hit@3 0.92 → 0.96）；
rapidocr 的中文模型识别英文时，常常整行不输出空格（"Pepperthe"）。现在只对一个空格都没有、且至少 8 个拉丁字母的行，按逐字框的中心距补回空格；中心距先按字母宽窄归一化，否则会拆出 "M ale""Im ageNet"。数字之间永远不拆。标准答案测试（论文图注渲染成图，3 种字体 × 3 种字号，1260 行）：单词召回 0.850 → 0.882，精度 0.900 → 0.906；中文图的输出不变。

**MinerU 档位对比**（同一份论文，人工问题，不开 VLM 描述）：flash 文搜图 hit@1 0.65（补译 0.73），standard 0.54（补译 0.65）。standard 解析更全（表 1~20 全部抽到），但不给图片附文字转写；伪代码图被当成代码块，没当图片；复合图会被拆成几块。同页连续、只有一块带图注的碎片，现在会继承图注；图注前面混进图表标题时（"… Figure 5. …"），也能认出本图编号。开着 VLM 描述时，两档差别会小很多；不开时，英文学术 PDF 用 flash 档图片检索更好。

**多模态评测**（评测中心 →「多模态评测（图片）」，代码 `prism/eval/multimodal.py`）：文搜图 / 图搜图 Hit@K·MRR（文搜图按召回路消融）、
图搜文关键词命中、回答附图的精确率 / 召回率 / 负例不附图率。样例可手写 JSONL，也可「从图片库生成」——用图注 / 图中文字出题、
把库里的图缩小压缩后当查询图，再加几条与图无关的负例。自动样例是自检，真实效果要靠人工整理的问答对。

实测得到的几条运行规则（都已是默认行为）：

- **回答附图有门槛**：图片所在段落被检索命中且图片描述与问题相关、或关键词 / 图文向量证据达标、或问题里明确要图，才会附上。
  关键词证据按词的稀有程度加权，"出示""图表""怎么"这类请求用语不计；只要有图在关键词上达标，就不再用图文向量带进其他图——
  Chinese-CLIP 的文-图余弦挤在 0.32~0.49，对"都是图表"的图普遍偏高。
- **OCR 保留版面**：竖直对齐的文字归成一组（柱状图输出"300 华北区 | 450 华东区 | …"），否则 LLM 会把数值和类目配错。
- **扫描页回填**：一页没有文字层、却有一张识别出大段文字的大图时，OCR 结果作为这一页的正文进入文本检索。
- **CLIP 默认跑 CPU**：PyTorch 的 Apple GPU（MPS）后端在多线程服务里与 MinerU 并发使用会触发 Metal 断言、进程直接退出；
  CLIP base 在 CPU 上每张图几十毫秒。确需 GPU 时设 `multimodal.embedding.device: mps`。
- **推理模型的辅助调用关闭思考**：查询改写 / HyDE / 精排 / text-to-SQL / 图片描述走 OpenAI 兼容端点时带 `thinking: disabled`
  （端点不支持会自动去掉）。否则思考 token 会耗尽 `max_tokens`，返回空内容。生成回答的主模型保留思考。
- **编号优先**：查询里有编号（SO-8817、PUMP-07，或 3-11、2025-0931 这类数字码）时，图里确实写着它的图片排最前；
  描述的语义路与关键词路一致指向同一张图时，也优先于图文向量路（CLIP 对"13-5 和 2-5"这类细节不敏感）。
- **图中文字辅助认图**：以图搜图 / 以图搜文时，查询图里有文字就先 OCR，再与视觉相似度一起认定是库里哪张图——
  视觉分得出"同类图"，文字才分得清"是哪一张"。
- **启动预热**：服务启动后台加载向量 / 重排 / 图文向量模型，重启后的第一个提问不必等 20 秒。
- 开发模式（`--reload`）下如果工作进程因原生库崩溃退出，uvicorn 不会自动拉起，表现为请求一直转圈：重新运行 `./start.sh`。

## LanceDB：向量后端 · 长期记忆 · 个性化推荐 · 多智能体 · 多模态

LanceDB 是嵌入式（目录即数据库、无服务进程）、列式、带版本的多模态数据库。同一张表里放向量、正文、标量列和图片字节，
支持向量 / 全文（BM25）/ 混合检索、SQL 预过滤、ANN 索引与时间旅行。本项目把它用在五个地方（`uv pip install -e ".[lancedb]"`）：

| 能力 | 表 | 代码 | 要点 |
| --- | --- | --- | --- |
| RAG 向量后端 | 每个知识库 `data/indexes/<kb>/lancedb/children` | `prism/lance/vector_store.py` | `vector_store.backend: lancedb`；小表精确扫描（召回 100%），≥ `index_min_rows` 自动建 IVF_HNSW_SQ；权限过滤走 SQL 预过滤；`retrieval.sparse_backend: lancedb_fts` 用原生全文替换 BM25 |
| 长期记忆 | `memories` | `prism/memory/` | 去重合并写入；召回分 = 0.7·相似度 + 0.15·新近度 + 0.15·重要度；问答时按用户召回（只用于个性化，不作为事实依据），回答后后台抽取 |
| 个性化推荐 | `rec_items` / `rec_events` | `prism/recommend/` | 兴趣向量 = 交互条目向量的时间衰减加权和（+ 记忆偏好）；热度加分、MMR 去同质化；每条推荐带理由；冷启动按热度 |
| 多智能体协作 | `agent_board` | `prism/agents/` | 规划 → 并行研究 → 分析 → 审校（证据不足 → 改写重查）→ 撰稿；智能体通过共享黑板按语义取证，运行可回放、可检索相似任务 |
| 多模态 | 每个知识库 `data/indexes/<kb>/lancedb/mm` | `prism/lance/multimodal.py` | 图片（缩略图字节 + CLIP 向量 + 图注 / OCR）与正文（CLIP 文本向量）同表；`multimodal.lance_table: true` 时接入线上图文检索（见下） |

中文全文检索用 LanceDB 自带的 ICU 分词（`fts_tokenizer: icu`），不用下载模型，"住宿标准""SO-8817""502" 都能命中；`jieba/default` 需要另外下载模型文件。

**多模态表接入线上图文检索**（`multimodal.lance_table: true`，策略配置「LanceDB 应用」组里可开关）：
- 文搜图 / 问答附图多一路「LanceDB·全文」：表里图片行的图注 / 描述 / OCR / 上下文做全文检索（ICU 分词，与描述 BM25 的 jieba 是两套分词），
  和描述 BM25 一样按查询覆盖率降权；图搜文多一路正文行上"查询图的 CLIP 向量 + 图中文字 / 图片转文字"的原生混合检索。都参与多路 RRF。
- 文搜图这一路刻意不再叠 CLIP 向量：图文向量已经单独成一路，再算一次只会放大 CLIP 对"哪个区、哪个季度"这类细节不敏感的噪声。
  120 张同风格图表压测（Chinese-CLIP + BGE-M3 + OCR，hit@1，完整图注 / 口语改写 / 关键词）：不接入 1.000 / 0.825 / 0.800，
  接入后 1.000 / 0.833 / 0.825；图搜图、图搜文不变，回答附图精确率 0.279 → 0.282。混合检索版本（CLIP + 全文）实测反而下降，没有采用。
- 同步是增量的：每行带内容签名，保存索引时只重写变了的行、删掉已不存在的行（缩略图只为新图生成），查询前再用内容摘要兜底检查一次；
  这一路单次约 3 ms。表随知识库目录一起删除 / 整库重建。「LanceDB」页的多模态检索实验用的也是这张表，手动同步还会补齐缺的正文图文向量。
- 没有图片的知识库（如企业知识库）表里只有正文行，用于图搜文。开关打开时，保存索引（包括整库重建）会给新切片补图文向量
  （单次不超过 `chunk_sync_per_query`）并同步正文行；查询前的兜底检查从不现算向量。
不想切换线上后端时，可以在「LanceDB」页把知识库**镜像**进全局库（复制现有向量与正文，不重新向量化），先体验全部能力。

实测（示例库，BGE-M3，Apple Silicon）：61 个切片镜像 16 ms；向量检索 3.9 ms、全文 14 ms、混合（RRF）3.8 ms，向量与 numpy 精确检索结果一致；
多模态表 3 图 + 9 段正文，内容没变时的同步 2 ms（只比对签名）；
多智能体（deepseek-flash 规划 / 审校 + v4-pro 分析 / 撰稿，不开思考）一次 3 个子问题、审校一轮通过，共 7.3 秒。
用户身份在顶栏「用户」里切换（为空时为 default），问答请求会带上它，长期记忆与推荐按它区分。
API：`/api/v1/lancedb/*`、`/api/v1/memory/*`、`/api/v1/recommend/*`、`/api/v1/agents/*`；参数见 `config.yaml` 的 `vector_store.lancedb`、`memory`、`recommend`、`agents` 段。

## Agent 湖：检索增强 + 湖分析

思路来自"从多模态数据湖到 Agent 湖"（Lance 格式）：记忆只是上下文里提炼出的一小部分，真正值得管理的是上下文本身。
智能问答的每一轮都自动沉淀进 LanceDB 表 `agent_lake`（`data/lancedb`，与长期记忆 / 推荐 / 黑板同库）——一张带版本的宽表：

```
一行 = 一条上下文资产   kind: question / retrieval / answer / analysis / tool_call / memory / board / document / doc_event / retest / note
结构化列               session_id · user_id · agent · kb_id · status · score · latency_ms · feedback · ts · day
半结构化 + 正文 + 向量   meta（JSON：检索策略、改写、各阶段耗时、命中文档）· refs · content · vector（当前向量模型）
引用关系               retrieval / answer 的 parent_id → question；tool_call 的 parent_id → analysis
```

| 能力 | 做什么 | 代码 |
| --- | --- | --- |
| 检索增强 | 回答前按向量召回相似的历史问题（余弦 ≥ `lake.augment_min_score` 0.75，排除同一会话），只取状态正常、没被点踩的回答，作为 `<历史问答参考>` 交给模型——不编号、不可引用，事实以本轮资料为准；本轮没检索到资料时不带 | `lake/augment.py` |
| 自动沉淀 / 回填 | 每轮问答后台写入 question + retrieval + answer；「回填历史数据」把历史对话、知识库文档、长期记忆、黑板按 id 去重导入 | `app/services/lake_service.py` |
| 湖检索 | 向量 + 全文（ICU 中文分词）+ RRF，LanceDB 先做结构化预过滤再算相似度：类型 / 会话 / 用户 / 知识库 / 最近 N 天 / 任意列的比较（`where`，只认列的比较与 AND / OR / LIKE / IN，拒绝子查询和函数） | `lake/store.py::search`、`lake/catalog.py::check_filter` |
| 检索质量 / 知识缺口 | 检索耗时 p50 / p95、置信度分布、拒答率、反馈、被检索最多的文档、策略对比；没检索到 / 低置信 / 拒答 / 点踩 / 失败的问题列为缺口 | `lake/analytics.py` |
| 缺口复测 | 湖里记录的是历史：补了语料后，把未解决的缺口（可选：当时检索分 < 0.3 的问题）拿到现在的知识库重新检索并生成回答，按 已解决 / 有改善 / 仍是缺口 记成 `retest` 记录；之后的缺口统计以最新复测为准（检索到但回答仍拒答不算解决）。知识缺口页「复测缺口」，分析智能体工具 `retest_gaps` | `lake/retest.py` |
| 文档变更记录 | 上传解析 / 校对回灌 / 重新解析（保留或放弃校对）/ 恢复校对版 / 删除各写一条 `doc_event`（行级变化摘要），文档快照随之更新；整库重建等系统动作只在正文真的变了时记录。缺口会列出发生之后同一知识库里变更过的文档 | `lake/changes.py` |
| 问题主题 | 问题向量余弦 k-means + c-TF-IDF 主题词，按主题看缺口率；可把主题写成派生列 `topic` | `lake/analytics.py::topics` |
| 只读 SQL | DuckDB 直接查 Arrow（表名 `lake`），禁用外部文件访问，只允许单条 SELECT / WITH | `lake/analytics.py::run_sql` |
| 湖目录 | 让智能体先"理解数据"再查数据（思路来自火山引擎 LAS 的 Lance 数据检索 Agent）：列（类型 / 说明 / 真实取值）、meta JSON 里按 kind 出现的键与 `json_extract_string` 写法、指标口径（拒答率 / 低置信率 / 缺口率 / 延迟分位 / 点踩率 / 策略对比 / 热门文档 / 追问率，各带固定 SQL）、问题 → SQL 示例（内置 + 湖里以前分析成功的 SQL）。人工维护的指标 / 示例 / 列备注存在湖里（`agent = catalog` 的 note 记录，跟版本走；带 SQL 的保存前先在湖上试跑），分析时最优先采用。每条配向量，按意图检索；`GET/POST /api/v1/lake/catalog`，SQL 页右侧可点选填入、添加、删除 | `lake/catalog.py` |
| 分析智能体 | 自然语言提问 → 目录发现（只把相关的列 / 键 / 口径 / 示例喂给模型）→ 按"决策罗盘"规划工具（目录 / 概况 / 检索质量 / 缺口 / 复测 / 路由 / 主题 / SQL / 检索）→ 执行 → 模型看结果反思，不够就追加调用（ReAct，总步数 ≤ `lake.agent_max_steps`；结果为空不许只改引号重试）→ 汇总结论（写明口径与样本数，相关 ≠ 因果）。SQL 报错交给模型改一次；`search` 可叠加结构化过滤（`days` / `where`，LanceDB 先过滤再算相似度）；没有模型时按关键词路由 + 模板结论。每次分析和工具调用也写回湖里，成为下一次的 Few-Shot | `lake/agent.py` |
| 版本 / Tag / 分支 / 派生列 | 每次写入是一个版本，可在任意版本 / Tag / 分支上分析；派生列按主键追加（表达式用 DuckDB 求值），先在分支上验证、对比，再合并回主表 | `lake/store.py` |

```
主表持续写入 ─▶ 打 Tag 冻结快照 ─▶ 从 Tag 拉分支 ─▶ 分支上追加派生列（主题 / 置信度分级 / 意图……）─▶ 对比 ─▶ 合并回主表
```

实测（BGE-M3，示例库）：同义改写的历史问题相似度 0.87~0.93、无关问题 ≤ 0.48，阈值 0.75 区分干净；
分析智能体（deepseek-flash 规划 + 汇总）一次 3 个工具调用约 4 秒。社区版 LanceDB 没有分支 diff / cherry-pick，这里按 id 自己比较与合并。
API 在 `/api/v1/lake/*`，参数在 `config.yaml` 的 `lake` 段；依赖 `uv pip install -e ".[lake]"`（lancedb + duckdb + pandas）。

## 飞书文档：同步 · 增删改查 · 文档智能体

飞书企业自建应用（tenant_access_token）读写新版文档（docx）、云空间文件夹和知识库（wiki）节点；凭据在「飞书文档 → 应用配置」里填，
或用环境变量 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_DOMAIN`（`feishu` / `lark`）。应用需开通 `docx:document`、`drive:drive`、
`wiki:wiki:readonly` 并发布版本；文档要在「…→ 添加文档应用」里加上这个应用（或放进应用有权限的文件夹）。

| 能力 | 做什么 | 代码 |
| --- | --- | --- |
| 同步进知识库 | 链接 / document_id / 文件夹里勾选 → 块转 Markdown（标题层级、嵌套列表、待办、代码、引用 / 高亮块、表格；图片 / 画板留占位）→ 带 front matter（标题、标签「飞书」、来源链接）存成 .md → 普通的解析 / 切片 / 索引。按 `revision_id` 判断飞书上改没改，没改跳过；之后检索、问答、评测、Agent 湖（`doc_event`「飞书同步」）都把它当普通文档 | `prism/feishu/convert.py`、`app/services/feishu_service.py` |
| 自动同步 | 定时：「应用配置 → 自动同步」选间隔（`auto_sync_minutes`，0 = 关闭），后台线程按间隔对所有接入了飞书文档的知识库增量同步，同一时间只跑一轮。事件（可选，更及时）：飞书开发者后台「事件与回调」请求地址填 `https://<域名>/api/v1/feishu/events`、订阅「文档编辑」`drive.file.edit_v1`，在页面填 Verification Token（设了 Encrypt Key 也一并填）；收到事件只同步那一篇。飞书只推送订阅过的文档：配好 Token 后新接入的会自动订阅，之前接入的在页面上一键补订阅。`deploy/nginx-rag-studio.conf` 已让这个地址免登录（它自己校验 Token） | `app/services/feishu_service.py` |
| 增删改查 | 读最新全文；新建（可直接入库）；追加；整篇替换（先删正文再写）；查找替换（只改命中的文字段，保留加粗 / 链接等样式）；删除（进飞书回收站并移出知识库）。改过的文档如果已入库，写完自动重新同步。带表格的 Markdown 优先用官方转换接口写入，不可用时按行写成文本 | `prism/feishu/client.py` |
| 文档智能体 | 规划 → 调工具（列出 / RAG 检索 / 读取 / 新建 / 追加 / 整篇替换 / 查找替换 / 删除 / 同步）→ 看结果反思追加 → 汇总；总结、对比、找矛盾等分析由模型基于读到的原文完成。写操作默认只列出「待确认」，页面上确认后才执行（`agent_allow_write` 或页面开关可改为直接写）；没有文本模型时按规则只做查询 | `prism/feishu/agent.py` |

API 在 `/api/v1/feishu/*`（`config` / `docs` / `import` / `sync` / `sync/auto` / `events` / `events/subscribe` / `remote` / `folder` / `agent/run`（SSE）/ `agent/execute`）；
经代理或私有化部署访问开放平台时用 `FEISHU_API_BASE` 覆盖接口地址。测试用内存里的假开放平台（`tests/test_feishu.py`），不碰网络。

## 语料治理

在调检索参数之前先把语料做干净、做完整（`prism/corpus/`，页面「语料治理」）：

| 能力 | 做什么 |
| --- | --- |
| 质量体检 | 段落级：乱码 / 目录页与纯符号 / 不成句的短段 / 超长 / 完全重复 / 近似重复（MinHash + LSH，保留现行较新版本）；文档级：已过期 / 同名多版本同时生效 / 无切片 / 解析差 / 缺元数据；健康分。剔除按内容指纹记录，重新解析与重建后仍生效，可恢复；多版本冲突可一键让旧版本失效 |
| 覆盖增强 | 从「语料覆盖」的部分匹配 / 无匹配意图出发，文本模型只根据检索到的原文写 FAQ（资料不足标为知识缺口，不编造）；人工审核通过后发布为「增强语料-FAQ.md」，像普通文档一样检索与引用，可撤回 |
| 别名词典 | 缩写 ↔ 全称、口语 ↔ 书面语：查询扩展立即生效，文档扩展（别名写进向量与 BM25 文本）自动重建；可从语料里的"全称（简称）""以下简称""又称"写法发现候选 |
| 关键词扩展 | `corpus.keywords`：每段补 TF-IDF 主题词与编号 / 错误码，不需要 LLM（默认关闭，打开会重建） |
| 自动元数据 | 上传时按规则抽取标题 / 版本 / 生效与失效日期 / 部门 / 标签 / 密级，只填空字段；可选文本模型补全（结果须在原文中有依据）；页面上批量复核 |

覆盖分析报告记录所用的嵌入与实际生效的精排模型；与上次报告的模型、打分公式或阈值不同时提示"不可直接比较"。

## MinerU 4.x 与 MinerU-Popo

策略配置页的「MinerU 档位」下拉由 `GET /api/v1/system/mineru/tiers` 动态加载：四档全部可选，逐档显示状态
（可用 / 需下载模型包 / 缺少运行依赖 / 未安装 / 由自托管服务 `/v1/tiers` 提供）、本机硬件推荐与下载命令；MinerU 解析失败时自动降级到内置解析器。

```bash
uv pip install -U "mineru[torch]>=4.0,<5"     # macOS Apple Silicon 自动包含 torch
mineru-kit models download --tier basic       # basic / standard 需要先下载模型包；flash 首次解析时自动拉取小模型
mineru-kit models verify                      # 检查本机模型是否齐全
```

**用真实 PDF 验证过的行为（mineru 4.0.10，Apple Silicon）**

- `flash` 档解析 PDF 也会跑版面检测与 OCR 小模型（PP-DocLayoutV2 + PP-OCRv6，约 240MB，首次解析时自动下载到 `~/.mineru/models`）；
  `basic` 还需要公式 / 表格模型（再约 660MB），`standard` 另需 VLM。
- `ocr_mode: auto` 遇到文本层里有图标字体 / 私用区字符的 PDF 会改走 OCR（4 页约 6 秒）；`txt` 直接读文本层（0.1 秒），保留 emoji，但段落更粗。
- Content List 的 `index` 块里常常是正文（课程大纲、章节清单），按列表保留，不当目录丢弃；
  页眉 / 页脚类型名是 `header` / `footer` / `aside_text`；首页标题可能被标成页眉，只出现一次时当正文保留。
- MinerU 的 PDF 渲染用多进程（spawn）：在脚本里直接调用时要放在 `if __name__ == "__main__":` 之下；Web 服务与 `rag` 命令不受影响。
- 回归测试：`RAG_TEST_MINERU=1 pytest tests/test_pdf_real_world.py -k live`（现场生成 PDF → MinerU → 切片 → 检索）。

| 需求 | 方式 |
| --- | --- |
| 数字 PDF / Office 快速建索引 | `parsing.mineru.tier: flash`（默认；只用版面 + OCR 小模型，CPU 即可） |
| 扫描件 / 表格 / 公式 | `basic`（ONNX / Torch 小模型，CPU 可跑）或 `standard`（+ VLM） |
| 集群部署 | `mineru-kit api-server --tier standard`，然后 `parsing.mineru.mode: server` 并填 `parsing.mineru.api_url` |
| 机器内存小 / 没有 GPU，又要 standard 质量 | 云端 API：`parsing.mineru.mode: cloud`（策略配置 → 文档解析 →「解析方式」）。文档上传到 mineru.net 解析，本机不需要模型包；密钥填 `parsing.mineru.cloud.api_key` 或环境变量 `MINERU_API_KEY`，不填则匿名调用（有页数 / 频率限制）。云端目前只提供 `standard` 档，选了其他档位会自动改用它；云端连不上时有内置解析器的格式自动降级 |
| 跨页表格 / 段落续接 / 全文标题层级 / 图文关联 | MinerU-Popo：知识库页对 PDF 点「Popo」→ 导出输入 → `scripts/run_popo.sh` → 打开 `parsing.popo.enabled` → 重新解析 |

未安装 MinerU 时自动使用内置解析器（pdfplumber / python-docx / python-pptx）；Markdown 永远直解析，Excel / CSV 永远走行级结构化。
`prism/ingest/popo_bridge.py` 同时兼容 MinerU 三种输出（`*_model.json` 千分比坐标、`*_middle.json` 物理坐标、`*_content_list.json`），统一归一化为 Popo 要求的 0~1 xyxy 块并做推理前校验。

## 文本嵌入（Embedding）模型

默认 **BAAI/bge-m3**（1024 维，中英双语，8192 上下文），通过 sentence-transformers 加载（`uv pip install -e ".[models]"`）；
没装时回退哈希向量（只反映词面重叠）。Apple Silicon 上单条查询编码约 30 毫秒，200 个切片约 1.2 秒。

换向量模型（改 `config.yaml`、在模型配置里激活 / 停用向量模型、或装上 sentence-transformers 后重启服务）时，
服务会发现索引记录的向量模型与当前不一致，**自动在后台重建受影响的知识库**，进度见 `GET /api/v1/settings/reindex-status`。
人工校对过的文档用保存的 Markdown 重建，不会被原文件覆盖。改了入库参数（解析 / 清洗 / 切片 / 索引增强 / BM25 / 向量库）同样触发。

整库重建不影响线上检索：
- 新索引建在 `data/indexes/.<kb>.rebuild-staging`，只有重建线程写它，检索 / 问答照常读完整的旧索引；
- 建好后锁住知识库，把重建期间的上传 / 校对 / 改元数据 / 删除补进新索引，等在途检索结束后用两次目录改名原子替换（LanceDB 按路径懒读文件，替换必须等在途检索结束）；
- 进程被打断（`uvicorn --reload`、重启）时正式索引从未被动过：待重建记录存在 SQLite，启动自检删掉临时目录并继续重建；索引文档数少于「已完成」文档数也会判为不完整并重建；
- 全部文档重建失败时放弃新索引、保留旧索引；Qdrant 后端（集合名取自目录名）退回"先备份、原地重建"。

示例黄金集上（关闭重排，只看向量路）：哈希向量 Hit@1 0.786 / MRR 0.875 → BGE-M3 Hit@1 0.929 / MRR 0.964；
口语问法（"出差能不能住贵一点的酒店"）BM25 排第一的是无关文档，BGE-M3 直接命中「住宿标准」。

## 重排模型

默认是本地交叉编码器 **BAAI/bge-reranker-v2-m3**（中英双语），直接用 transformers 加载，装了 `mineru[torch]` 或 `rag-agent-studio[models]` 即可用，
不需要 sentence-transformers。Apple Silicon（MPS）上 50 个候选约 0.2 秒，首次加载约 1 秒；同一台机器上 LLM 打分精排一次要十秒量级。

| provider | 实现 | 配置 |
| --- | --- | --- |
| `auto`（默认） | 显式配置的 API → 本地 bge-reranker-v2-m3 → LLM 打分 → 词法 | 什么都不用配 |
| `cross_encoder` | 本地模型，`retrieval.rerank.model` | 权重不在本机时 `auto_download: true` 会后台下载（约 2.3GB），期间用 LLM / 词法顶上；也可先 `rag download-reranker` |
| `cohere` | Cohere Rerank v2（`POST https://api.cohere.com/v2/rerank`） | `cohere_model`：rerank-v4.0-pro / rerank-v4.0-fast / rerank-v3.5 …；密钥：模型配置页添加「Cohere Rerank」或环境变量 `COHERE_API_KEY` |
| `api` | Jina / SiliconFlow 等 `{base}/rerank` 接口 | `api_url` + `api_key` + `model` |
| `llm` / `lexical` / `none` | LLM 批量打分 / 词法兜底 / 不重排 | — |

模型配置页里激活的重排配置优先于 `config.yaml`：选「本地 sentence-transformers → BAAI/bge-reranker-v2-m3」或「Cohere Rerank」，「测试」会用一相关一无关两条文档打分，
顺带检查分数方向。拒答阈值 `min_score` 要按模型标定：bge-reranker-v2-m3 对无关文档的分数 < 0.01，
同义改写的相关文档只有 0.1~0.2，取 0.02~0.05 比较稳（`RAG_TEST_RERANKER=1 pytest tests/test_rerankers.py -k live` 可复现）。
示例库实测（BGE-M3 + bge-reranker-v2-m3）：黄金集可答题的精排最高分 0.79~0.999，口语改写最低（"出差能不能住贵一点的酒店" 0.07~0.09），
库外问题（"公司食堂几点开饭""明天北京天气怎么样"）≤ 0.002；取 0.03 时黄金集命中率不变、无答案题全部拒答。
阈值拦不住"沾边"的问题（"宿舍怎么申请"会拿到 0.04 的「办公室租金报销」），这类靠回答提示词里的"资料不足就说明"兜底。

## 本地模型与 Apple GPU

BGE-M3、bge-reranker、MinerU 本地解析（以及显式设为 `mps` 的 CLIP）在 Apple Silicon 上共用 `prism/gpu.py` 的一把锁串行执行。
PyTorch 的 MPS 后端不是线程安全的：两个线程同时在 MPS 上计算，要么 Metal 断言失败、整个进程 abort（uvicorn 的 worker 退出，之后请求全部挂起，
直到下一次热重载），要么内部缓存被并发改坏、线程满 CPU 卡死——两种都在本机复现过。
代价：MinerU 解析大 PDF 期间，同一进程里的检索要等它跑完；向量化按 64 条一段持锁，入库期间查询可以插在两段之间。CUDA / CPU 不加锁。

## 命令行

```bash
rag ingest data/raw --lite                    # 解析 + 切片 + 建索引（--lite：哈希向量，无需模型）
rag query "K8s Ingress 502 怎么排查" --lite --routes
rag ask "出差住宿标准能不能超额"               # 需要 LLM
rag coverage intents.txt --lite -o coverage.md   # 意图 × 知识覆盖分析
rag eval --lite -o report.md                  # 分阶段消融 + 错误归因
rag popo-export report.pdf                    # 生成 MinerU-Popo 输入
rag download-reranker                         # 把默认重排模型 bge-reranker-v2-m3 下载到本地缓存
rag serve --port 8000                         # 启动 Web 后端
```

## 目录

```
prism/                    核心库（不依赖 Web）
  ingest/parsers/         mineru / markdown / excel / docx / pptx / pdf 兜底 / html_table
  ingest/cleaning.py      页眉页脚 / PDF 断行 / 乱码 / 近重复
  ingest/chunking/        structural（标题面包屑）/ semantic / recursive / parent_child
  ingest/popo_bridge.py   MinerU ⇄ MinerU-Popo（三种输入适配 + 校验 + 文档树导入）
  ingest/enrich.py        Contextual Retrieval / 假设性问题
  index/                  embeddings（hash / sentence-transformers / OpenAI 兼容）/ 向量（numpy / Qdrant / Chroma）/ BM25 / 块存储
  graph/unweaver.py       术语定义抽取与织入（GraphRAG-lite）
  index/table_store.py    Excel / CSV → SQLite，SQL 结构化检索路（只读 text-to-SQL + 规则兜底）
  multimodal/             图文向量 / 图片抽取 / OCR·VLM 描述 / 图片索引 / 文搜图·图搜图·图文混合·图搜文
  retrieval/              查询变换 / RRF / 重排 / MMR / 粒度 / 压缩 / 多跳 / Agentic / 路由 / HybridRetriever
  generation/             Anthropic SDK / OpenAI 兼容 / 回答协议
  eval/                   黄金集 / 指标 / 消融 / 错误归因 / 意图覆盖分析
  corpus/                 语料治理：质量体检 / 别名词典 / 关键词扩展 / 自动元数据 / 覆盖缺口 FAQ
  feishu/                 飞书开放平台客户端 / 文档块 ⇄ Markdown / 文档智能体（增删改查与分析）
  cli.py                  rag ingest | query | ask | coverage | eval | popo-export | serve | stats
app/                      FastAPI 后端：api/ 路由、services/ 业务、db/ SQLite 元数据
frontend/                 React 19 + Vite + TypeScript
data/raw/                 示例语料（含一份已过期的 2024 版制度，演示时效过滤）
data/eval/golden_qa.jsonl 示例黄金评估集
config.yaml               策略基线（前端"策略配置"页的覆盖保存在 SQLite）
scripts/                  MinerU 批量解析、Popo 推理
tests/                    单元 + API 端到端（离线）
```

## 调优顺序

1. **语料**：在「语料覆盖」里确认知识库能回答用户会问的问题；在「语料治理」里剔除噪音与重复、处理多版本冲突、为缺口补 FAQ、补别名与元数据；下线无关与过期文档。
2. **切片**（约一半时间）：在「知识库 → 切片」里看 parent / child 是否语义完整、表格是否带表头、面包屑是否正确。
3. **混合检索**：在「检索实验室 → 各路召回」里看编号 / 错误码类查询是否由 BM25 命中；必要时加领域词典 `sparse.user_dict`。
4. **重排**：确认候选数 ≥ 50，配置真实的重排模型。
5. **Query 改写**：口语化 / 指代不清的问题再开 rewrite / multi-query / HyDE。

每改一个变量在「评测中心」跑一次，指标不动就不合入。

「策略配置」页顶部有四个**场景预设**（企业综合知识库 / 医疗·法律 / 客服 FAQ / 长文档问答）和一张"瓶颈 → 手段"对照表：
查不全 → 混合检索 / 多路召回；查不准 → RRF + 重排；块又长又吵 → 上下文压缩；答不对 → 父块与切片。手段不是全上最好。
