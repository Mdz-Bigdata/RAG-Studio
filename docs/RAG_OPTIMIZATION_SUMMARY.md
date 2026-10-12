# RAG 优化总结：从语料治理到混合检索的完整方法论

> 综合 8 篇实战文章（2026-05 ～ 2026-09）、语料优化四维度笔记（数据质量 / 切片策略 / 索引优化 / 知识增强）、MinerU 4.x 与 MinerU-Popo 资料整理。
> 每一节末尾标注了本项目中的对应实现（核心库 `prism/...`、后端 `app/...`、前端页面），便于对照代码；第 12 节是完整的代码映射。

---

## 0. 一句话结论

**RAG 的效果不是由某一个模型决定的，而是由数据、解析、切分、索引、召回、排序和生成共同构成的系统结果。**

检索质量可以抽象成四个因子做乘法：

```
检索质量 ≈ 切块质量 × 检索策略 × 排序质量 × Query 质量
```

乘法意味着短板拖全盘：把一个维度从 0.6 调到 0.9，不如把四个维度都从 0.5 调到 0.7（0.6⁴ = 0.13，0.7⁴ = 0.24，后者接近两倍）。实践中把 embedding 从 ada-002 换成 bge-large-zh 命中率只涨 3 个点，把固定 500 字切块改成语义切块直接涨 15 个点。

产品原则：**不要一遇到效果问题就换大模型，先判断错误发生在哪一层。**

### 全景图：12 个调优入口（Leonie Monigatti 框架）

| 阶段 | 可调优策略 |
| --- | --- |
| 数据录入（离线） | 数据清洗、文档分块、嵌入模型、元数据、多重索引、索引算法 |
| 在线推理 | 查询转换、检索参数、高级检索、重排序模型、大语言模型、提示工程 |

这 12 项不是 12 个独立开关，而是一条质量链路：源数据错了，后面模型越强、错误越像真的；切分破坏了语义，再好的检索模型也找不到完整答案。

### 六项核心检索能力速览

| 能力 | 一句话 | 解决的问题 | 关键参数 | 什么时候不要用 | 代码 / 页面 |
| --- | --- | --- | --- | --- | --- |
| 向量 + 关键词检索 | 最常见组合：语义灵活性 + 字面精确性 | 同义表达靠向量；编号 / 错误码 / 型号 / 新词靠 BM25 | 每路 top-N = 50-100；BM25 k1=1.5、b=0.75；jieba + 精确 token 保护 | 库 < 100 条且只有语义查询时纯向量即可 | `index/`、`retrieval/retriever.py::_recall`；检索实验室「各路召回」 |
| 查询重写 / 扩展 | LLM 把原始提问改写成多个角度、不同措辞的子问题检索同一个库 | 口语、指代不清、提问角度与文档描述角度对不上、多跳 | rewrite、multi_query = 3-5、decompose；**原始问题必须保留** | 延迟敏感；改写可能理解错原问题 | `retrieval/query_transform.py`；检索实验室「查询层」 |
| 假设文档嵌入 HyDE | LLM 先凭空写一段假设答案，用它的向量去真实知识库检索 | 问题与文档文体不同；短 query 语义锚点不足 | 仅 ≤ 30 字的短 query；temperature 0；约 100-200 字 | **含精确数字 / 编号 / 日期的查询**；需要 metadata 精确匹配的查询 | `query_transform.py`（`has_precise_tokens` 自动跳过） |
| 不同粒度检索 | 同时拿到聚焦的小块（高精度）与带完整上下文的大块（高可读） | 小块检索准但内容碎，大块完整但语义稀释 | child 100-150 tok、parent 500-800 tok、overlap 10%-15%；相邻命中自动合并；句子窗口 | — | `ingest/chunking/parent_child.py`、`retrieval/granularity.py`；知识库「切片」 |
| RRF 融合 | 不看分数只看排名：`Σ w / (k + rank)` | 余弦相似度与 BM25 分数量纲不可比，归一化不稳定 | k = 60（可试 20 / 40）；加权 RRF 起手 1:1 | 两路分数分布稳定且有评估集时可 A/B 凸组合（α = 0.5 起手） | `retrieval/fusion.py`；检索实验室「融合与精排」 |
| ReRanker | 交叉编码器对 (query, chunk) 逐对打分，把真正相关的浮上来 | 粗召"相关但不最相关"；Lost in the Middle | **候选 ≥ 50** → 保留 10 → 上下文 3-5；`min_score` 拒答阈值 | 延迟 < 50ms 的场景先降候选到 30，而不是砍掉 | `retrieval/rerank.py`（Cross-Encoder / Rerank API / LLM / 词法四种实现） |

六项都可以在前端「总览」看到开关与当前值，在「检索实验室」逐层观察效果，在「策略配置」或单次请求里覆盖参数，在「评测中心」用分阶段消融量化收益。

### 生产级完整链路

```
意图 × 知识覆盖分析 → 补知识 / 下线无关与过期材料
文档 → 解析(MinerU) → Popo 文档树 → 清洗 → 切分 → 向量化 → 入库
用户 Query
   │  查询层：多轮归并 / 改写 / Multi-Query / 分解 / HyDE（精确查询跳过）
   ▼
   ├── 路 1：向量检索（Dense，top-50）
   ├── 路 2：BM25 关键词（Sparse，top-50）
   └── 路 3：HyDE 假设文档 → 向量检索
   ▼  融合层：RRF k=60（或等权凸组合 α=0.5）→ 候选集 ≥ 50
   ▼  精排层：Cross-Encoder（bge-reranker-v2-m3）→ 保留 10
   ▼  多样化：MMR λ=0.7（可选）
   ▼  粒度层：child → parent（Small-to-Big）、相邻合并、Lost-in-the-Middle 排序 → 3~5
   ▼  织网 / 压缩：术语定义织入、上下文压缩（可选）
   ▼  生成层：回答协议（依据 / 拒答 / 冲突 / 安全 / 格式）+ 引用
   ▼  评估闭环：Recall@K / MRR / NDCG + 错误归因 → 反哺上游
```

---

## 1. 阶段一：文档准备 —— 意图空间 vs 知识空间

在解析任何文件之前，先回答两个问题：**用户会问什么（意图空间）**，**知识库能答什么（知识空间）**。

### 1.1 构建意图空间

- 收集真实提问：客服工单、搜索日志、群聊问题、销售 FAQ；
- 按类型分桶：常见标准问题、口语化表达、多轮追问、多意图问题、无答案问题、权限敏感问题、时效性问题、冲突问题；
- 每桶给出代表问题与期望答案 —— 这也是后续黄金评估集的雏形。

### 1.2 构建知识空间

对每份候选文档打上"身份证"：来源、版本、生效时间、部门 / 产品线、权限级别、语言、内容类型。

### 1.3 匹配分区间与处置

| 匹配区间 | 含义 | 处置 |
| --- | --- | --- |
| 高匹配 | 意图有明确对应文档，且文档完整、最新 | 直接进入解析 |
| 部分匹配 | 文档存在但缺章节 / 版本旧 / 表格散架 | **继续准备知识**：补充章节、更新版本、导出结构化表格 |
| 无匹配 | 高频意图完全没有对应文档 | 列入知识补充计划，同时配置拒答话术 |
| 无关 | 文档与任何意图不相关（会议纪要、营销稿、重复旧稿） | **移出语料**，避免噪音干扰与权重虚高 |

### 1.4 知识治理五问（生产级数据清洗 ≠ 格式整理）

- **版本**：同一制度是否存在多个版本？哪份已失效？
- **事实**：新旧文档冲突时，以哪个信息源为准？
- **权限**：不同岗位是否拥有不同的查看范围？
- **时效**：文档更新后索引能否同步？
- **解析**：表格、图片、扫描件、页眉页脚是否被正确处理？

典型翻车：知识库同时存在"2024 年差旅制度"和"2025 年差旅制度"，系统在两份之间反复横跳 —— 这不是幻觉，是知识源冲突。解法是**生效时间 + 版本元数据 + 检索时过滤过期内容**。

### 1.5 本项目的落地：覆盖分析

把上面的表格变成一个可反复运行的检查（前端「语料覆盖」页 / `rag coverage intents.txt`）：

1. **输入意图空间**：每行一个真实问题（工单、搜索日志、群聊、销售 FAQ），也可直接贴黄金集 JSONL；
2. **每条意图跑一次真实检索**（关闭查询改写，看的是语料本身），得到三类信号：
   - `term_coverage`：意图关键词在最优上下文中的覆盖率，按语料 IDF 加权 —— 语料里从未出现的词权重最高，精确 token（编号 / 错误码）权重加倍，"能不能 / 多少 / 怎么"这类只表达提问方式的词不计；
   - `dense_sim`：向量路 top-1 余弦相似度；
   - `rerank_score`：交叉编码器 / Rerank API / LLM 精排的 top-1 相关度（只有这三类是绝对分数）；
3. **匹配分**：有绝对精排分时 `0.6 × rerank + 0.4 × term_coverage`，否则 `0.7 × term_coverage + 0.3 × min(1, dense_sim / 0.5)`；
4. **分区间**：≥ 0.6 高匹配、≥ 0.35 部分匹配、其余无匹配（阈值可调），并列出**缺失关键词** —— 它们就是知识缺口；
5. **反查知识空间**：每份文档被多少条意图命中 → 核心语料（≥ 3）/ 辅助语料 / **未被命中**（与 RAG 目的可能不相关，确认后移出）/ **已过期**；
6. **"只被过期版本覆盖"单独提示**：这不是没有知识，而是知识没更新。

离线模式（哈希向量）下没有真实语义，分数偏保守；配置真实向量与重排模型后更准。意图数建议 ≥ 100 条（常见 50% / 长尾 30% / 多跳 20%），高匹配的意图标注答案后直接沉淀为黄金评估集。

> 代码：`schemas.DocMetadata`（version / valid_from / valid_to / department / tags）、`retrieval.filter_expired`、Markdown front matter 解析 `ingest/parsers/markdown_parser.py`；覆盖分析 `eval/coverage.py`（`CoverageAnalyzer`）、`POST /api/v1/eval/{kb_id}/coverage`、前端 `pages/coverage.tsx`。示例语料 `data/raw/员工手册_2024版.md` 演示过期过滤。

---

## 2. 阶段二：文档解析 —— MinerU 4.x 与 MinerU-Popo

### 2.1 不同格式的解析策略

| 格式 | 推荐做法 | 原因 |
| --- | --- | --- |
| Markdown | **必做**，永远走 Markdown 解析器 | 标题层级天然给出章节结构，代码块 / 表格是显式原子块；MinerU 输出的 markdown 也能回灌 |
| PDF（数字） | MinerU `flash` tier | 不加载推理模型，最快，适合建索引 |
| PDF（扫描 / 复杂版面） | MinerU `basic`（CPU 可跑）或 `standard`（+ VLM） | OCR / 公式 / 表格识别 |
| Word / PPT / EPUB / OFD / HTML | MinerU 4.x 原生解析（DocVortex，无需转 PDF） | 保留结构 |
| Excel / CSV | **行级结构化块**（每行一个块，附表名与列名） | "查表格"本质是查数据库；整表压扁后检索到的是一坨乱码碎片 |
| 图片 / 图表 | MinerU `image_analysis=True` 输出描述文字 | 图片需要 OCR 或视觉模型生成文字描述再切 |
| 代码文档 | 按 AST / 函数切 | 函数定义和调用不能分在两个块 |

坑 3（表格和 PDF 直接当文本灌）的解法：表格单独走结构化解析（哪怕人工导出 CSV），然后决定走 RAG 还是干脆做进 Function Calling。

### 2.2 MinerU 4.x（2026-09-16 发布 4.0，当前 4.0.10）

MinerU（opendatalab，AGPL-3.0）是**页面级**文档解析引擎 —— "读懂每一页"。实现策略经历了三代，读历史教程时要先分清版本：

| 版本 | 形态 | 说明 |
| --- | --- | --- |
| 1.x – 2.x | magic-pdf 线性流水线 | 多个专用模型串成固定 Pipeline：预处理分类（PyMuPDF 判断文字型 / 扫描型）→ 版面检测（DocLayout-YOLO）→ 内容识别（PaddleOCR 文字、UniMERNet 公式→LaTeX、StructEqTable 表格→HTML）→ 阅读顺序（LayoutReader）→ Markdown / JSON |
| 3.x | 多后端并行 | `pipeline`（传统多模型）/ `hybrid`（小模型 + VLM）/ `vlm`（纯 VLM 端到端），另有 `*-http-client` 远程变体 |
| 4.0 | 档位制（当前主线） | 不再按"后端"选型，改为档位（Tier）+ 独立模型配置：小模型后端（ONNX / Torch）与 VLM 引擎（llama.cpp / vLLM / LMDeploy）完全解耦 |

MinerU 4.0 用**四个质量档位**取代了 3.x 的 pipeline / vlm / hybrid 后端选择，核心差异在"用不用 VLM"：

| Tier | 做什么 | 适用 |
| --- | --- | --- |
| `flash` | 原生 PDF 文本、Office 等数字文档，不跑 VLM；PDF 仍会用版面检测 + OCR 小模型（实测首次自动下载约 240MB）；无需 GPU | 发现、预览、**索引构建** |
| `basic` | 仅 ONNX / Torch 小模型做 OCR、公式、表格，无 VLM；CPU 可跑 | 扫描件、普通表格 |
| `standard`（默认） | 小模型 + VLM，复杂版面高质量解析；建议 16GB 内存 + GPU | 复杂报告、混排 |
| `advanced` | 与 standard 共用模型，投入更多推理计算；最慢 | 疑难文档 |

安装（Python 3.10 ～ 3.14）：

```bash
uv pip install -U "mineru[torch]>=4.0,<5"      # [torch]：Torch 小模型 + 更高吞吐 VLM；Apple Silicon 自动包含 torch
uv pip install -U "mineru[full]>=4.0,<5"       # + vLLM（Linux）/ LMDeploy（Windows），需 16GB 内存 + 8GB 显存
mineru-kit models download --tier standard     # MinerU-4_models_onnx / _torch 模型包
mineru-kit models verify
```

> 3.x 的 `mineru[core]` / `mineru[pipeline]` 已被基础包、`mineru[torch]`、`mineru[full]` 取代；配置改为 `$MINERU_HOME/config.yaml`，`model.small_backend`（auto/onnx/torch）与 `model.vlm.engine`（auto/llama-cpp/vllm/lmdeploy）独立配置。

Python SDK（本项目使用的接口）：

```python
from mineru.parser import parse, MinerUApiParser, ParseResult
from mineru.render import render_content_list          # Content List V1：扁平列表，带 text_level / page_idx / bbox

result = parse("report.pdf", tier="standard", ocr_mode="auto", image_analysis=True, page_range="1-5,r3-r1")
md = result.markdown()                                  # Markdown
content_list = render_content_list(result.middle_json)  # [{"type": "text", "text": "...", "text_level": 1, "page_idx": 0, "bbox": [...]}, ...]
result.to_json()                                        # middle_json（docvortex.middle 2.0 协议，可 ParseResult.from_json 还原）

# 服务化：mineru-kit api-server --host 0.0.0.0 --port 8000 --tier standard
parser = MinerUApiParser(api_url="http://127.0.0.1:8000", tier="standard", include_images=False)
result = parser.parse("report.pdf", page_range="1-3")
```

Content List 的块类型：`text`（带 `text_level` 即标题）、`list`、`table`（`table_body` 为 HTML，`table_caption` / `table_footnote`）、`image` / `chart`（`image_analysis` 时含描述）、`code`、`equation`、`page_header` / `page_footer` / `page_number` / `page_footnote` / `index`。页眉页脚页码在 MinerU 层已打标，清洗时直接丢弃。

**用真实 PDF 实测（mineru 4.0.10，4 页浏览器导出的中文 PDF，图标字体项目符号）**：

| 路径 | 耗时 | 结果 |
| --- | --- | --- |
| flash + `ocr_mode: auto` | 约 6 秒 | 文本层含私用区图标字形 → MinerU 改走 OCR：字形正确，但空心圆点被认成"。"、emoji 丢失；课程大纲整块被标成 `index` |
| flash + `ocr_mode: txt` | 约 0.1 秒 | 直接读文本层：保留 emoji，项目符号是私用区字符，段落粒度更粗；首页标题被标成 `header` |
| pdfplumber 兜底 | 约 0.2 秒 | 文本层的"音 手 一"是康熙部首码位（⾳ ⼿ ⼀，137 处），不归一化则"语音"检索不到"语⾳" |

据此在接入层做了四件事：`index` 按列表保留、页眉只在首页出现一次时当正文、康熙部首 / 部首补充区归一化为统一汉字、图标字体与 OCR 的"。"项目符号转成列表项；
切片器把连续的"纯标题"并入后一节，避免主标题、副标题从索引里消失。修复后三条路径都能检索到大纲、标题、价格等全部内容。

其他 4.0 能力：多格式原生输入（PDF、图片、DOC/DOCX、PPT/PPTX、XLS/XLSX、RTF、ODT/ODS/ODP、EPUB、OFD、HTML、CSV/TSV）；本地文档库与 Agent 阅读（`mineru parse / read / search`，定位符 `doc:{id}/tier:{tier}/page:{page}/block:{block}` 可做可验证引用）；一个文档模型渲染九种目标（Markdown / HTML / LaTeX / DOCX / EPUB / PDF / Structured Content / Content List V1 / V2）；V1 HTTP API（`/v1/parse/jobs` 等）；`llm_aided` 后处理（标题分级、跨页表格单元格合并）。

> 代码：`ingest/parsers/mineru_parser.py`（`parse` / `MinerUApiParser` → `content_list_to_blocks`，HTML 表格转 Markdown），`scripts/mineru_batch_parse.py`。

### 2.3 MinerU-Popo：从"读懂一页"到"读懂一篇"

页面级解析和文档级理解之间有一条鸿沟：OCR 工具能读懂每一页，但页边界会切断段落、劈开表格、丢失标题层级、把图片和图注拆散。下游 RAG 需要的是整篇文档 —— 标题属于哪一级、这段接哪段、这张表跨了几页、这张图归哪个章节。

MinerU-Popo（opendatalab，MIT，交大 / UC Berkeley 合作论文 arXiv:2605.24973）的思路很直接：**不重新解析，只做后处理** —— 复用现有 OCR 输出，用一个 4B 轻量模型（基于 Qwen3-VL-4B 微调）重建文档级逻辑结构。论文对主流 OCR 的摸底显示，没有任何一个模型能完整处理所有文档级关系：

| 跨页关系 | MinerU | PaddleOCR | GLM-OCR | Dolphin | MonkeyOCR |
| --- | --- | --- | --- | --- | --- |
| 表格截断 | 支持 | 部分 | 不支持 | 不支持 | 支持 |
| 文本截断 | 部分 | 部分 | 部分 | 不支持 | 部分 |
| 标题层级 | 不支持 | 支持 | 部分 | 部分 | 支持 |
| 图文关联 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |

**四个子任务**（同一个套路：过滤 → prompt → 模型 → 解析 → 写回）：

| 子任务 | 过滤策略（只喂相关元素，缩短上下文） | 模型输出 | 写回字段 |
| --- | --- | --- | --- |
| 文本截断恢复 | 只留 text，过启发式规则；每块只取首句 + 末句 | `<|src_id|>A<|tgt_id|>B`（A 续接到 B） | `contd` |
| 表格截断恢复 | 相邻页表对 + 预筛，提取行级单元格 | 是否续表 + 单元格续接坐标 | `table_merge` + `cell_list` |
| 标题层级重建 | 只留 title，内容截 50 字符 | `<|id|>X<|level|>N`（开放整数层级，适配任意深度） | `level` |
| 图文关联 | 只留视觉块，先做嵌套块重叠链接 | caption ↔ 图表、图表 ↔ 所属章节标题 | `image` |

**三大支撑机制**：

1. 任务导向数据引擎：文档级过滤（按来源 / 领域 / 长度平衡采样 30K 训练数据）+ 任务级过滤（每个子任务只喂相关元素）；
2. 动态分块与同步：重叠分块（`chunk_size=50, overlap=1`，块边界选在目标页 ±5 页窗口内任务元素密度最高的页）→ 每块把涉及页面渲染成带红色页码的拼接大图，随结构化文本一起输入多模态模型 → 以首块为锚，用重叠区共同标题的层级差 `avg_bias` 逐块校准，避免后续块把二级标题误升为一级；
3. 文档富化：按标题层级组装文档树；长节点按段落边界切分；LLM 为节点生成摘要 —— 兼顾检索粒度与信息完整性。

**性能（官方）**：

| 上游解析器 | 层级 TEDS 处理前 | 处理后 |
| --- | --- | --- |
| MinerU | 53.7 | **90.6** |
| MonkeyOCR | 48.9 | 87.4 |
| Dolphin | 60.4 | 83.5 |
| PaddleOCR | 59.3 | 82.6 |
| GLM-OCR | 53.5 | 81.8 |

比通用大模型更强更快：Popo 90.6 TEDS @ 0.37 doc/s，Qwen3-VL-32B 为 78.0 @ 0.04 doc/s。下游 RAG（ViDoRe V3）五个子集中四个准确率领先（计算机科学 84.4 vs Raw RAG 82.3，HR 66.8 vs 63.2，医药 71.6 vs 64.4），查询延迟最多降约 70%。

**四阶段数据流**：页面级 OCR 输出 → 标签归一化（`label_normalization.py`）→ 四子任务推理（`inference.py`）→ 文档树构建（`get_json_tree.py`）。

归一化把不同 OCR 的标签空间对齐到统一的块格式（规范类型只有 title / text / image / table / caption 五类，`popo_type` 细分出 list_item / equation / image_caption / table_footnote / header / footer / page_number 等），按页打包成 `{"pages": {"1": [block...]}}`：

```json
{ "type": "title", "content": "1. Introduction", "bbox": [0.12, 0.34, 0.65, 0.38], "source_id": "doc:5" }
```

推理前严格校验：**bbox 必须是 0~1 的 xyxy 顺序**，否则拒绝。

**MinerU 各种输出能不能喂给 Popo**（Popo 的 `MineruReader` 按优先级读取）：

| 优先级 | 文件 | 对应 MinerU 模式 | 坐标与内容 |
| --- | --- | --- | --- |
| 1 | `{doc}_model.json` | VLM 模式 | 页面块数组，bbox 为千分比（0~1000），内容在 `lines[].spans[].content` |
| 2 | `{doc}_middle.json` | 经典 pipeline / hybrid | `pdf_info[].para_blocks[]`，bbox 为页面物理坐标，按 `page_size` 归一化 |
| 3 | `{doc}_content_list.json` | Content List 协议 | 扁平块列表，带 `text_level` 的视为标题 |

- MinerU < 4.0：pipeline 模式给 `middle.json`，vlm 模式给 `model.json`，直接适用；
- MinerU 4.0：九种渲染目标中 Content List V1 / V2 与 Popo 消费的 `content_list.json` 同源（type / bbox / text_level 结构一致），理论上可直接喂，但官方仓库没有 4.0 专门适配器，需要实测对齐字段；
- **不适用**：Markdown / Structured Content 等纯渲染输出 —— 丢失了逐块 bbox 和元素类型；
- 类型映射里 title / image / table / equation / 各类 caption / footnote / list 全部支持，`discarded` / `image_body` / `table_body` 会被跳过；
- 推理时要用 `input_label` 指向**源 PDF 路径**来渲染拼接页图。

**树构建**：跨页表格合并 → 按 title 切分文本组件（续接标记转 `<|txt_contd|>` / `<|txt_split|>`）→ 按 level 用栈建树 → 视觉元素挂到所属标题节点（caption → title、footnote → metadata）→ 页眉页脚页码挂根节点。节点结构：`{type, title, metadata, content, level, location[{bbox, page}], block_ids, children[]}`。

**与 MinerU 4.0 内置后处理的选型**：MinerU 4.0 自身也内置了轻量 LLM 辅助后处理（`title_leveling` 标题分级、`cross_page_table_cell_merge` 跨页表格单元格续接，默认关闭）。

- 需求轻（只要标题分级 / 跨页表格续接）→ 直接开 MinerU 内置功能，零额外部署；
- 需求重（完整文档树、图文关联、节点摘要、多 OCR 通用、面向 RAG / QA）→ 接 MinerU-Popo。

**本项目的工作流**：

```bash
# Popo 仓库（Python 3.10 + GPU，或官方 vLLM Docker）
hf download DreamEternal/MinerU-Popo --local-dir models/MinerU-Popo

# 本项目：前端「知识库」页对 PDF 点「Popo」→「导出 Popo 输入」，或命令行
rag popo-export report.pdf         # MinerU 解析 → post-process/mineru/<doc_id>/vlm/<doc_id>_content_list.json
                                   #             → eval_pdf_dir/<doc_id>.pdf
                                   #             → outputs/label_normalization/mineru/<doc_id>.json（已归一化并通过校验，可跳过官方第 2 步）
POPO_REPO=/path/MinerU-Popo bash scripts/run_popo.sh   # run_inference.py → get_json_tree.py
# 策略配置打开 parsing.popo.enabled → 对文档点「重新解析」，用文档树替换页面级结构
```

对 RAG 的直接收益：跨页表格被合并成一张完整的表；被分页打断的段落续接后再切片，不再拦腰斩断；标题层级来自全文而不是单页，**面包屑更准**（第 4.4 节）；图片 / 表格与说明文字关联。

> 代码：`ingest/popo_bridge.py` —— `export_for_popo`、`content_list_to_popo_input`（MinerU 4.x）、`model_json_to_popo_input` / `middle_json_to_popo_input`（MinerU < 4.0）、`mineru_output_to_popo_input`（自动识别三种输入）、`validate_popo_input`（0~1 xyxy 校验）、`import_popo_tree`（`<|txt_contd|>` 拼接、合并表格、层级还原）；`scripts/run_popo.sh`；`ingest/pipeline.py` 自动检测树文件；API `GET/POST /api/v1/documents/{id}/popo[-export]`、`POST /documents/{id}/reprocess`、`GET /system/parsing`。

---

## 3. 阶段三：数据清洗与预处理（提升数据质量）

| 问题 | 表现 | 解法 |
| --- | --- | --- |
| 页眉 / 页脚 / 页码 / 水印 | 每页重复的公司名、"第 x 页" | MinerU 已打标的直接丢；未打标的按"同一行出现在 ≥40% 页面"识别 |
| PDF 断行 | 一行一个换行，切块得到大量半截句 | 上一行不以终止标点结尾且下一行不是列表项 → 拼接；中文直接拼，英文补空格，连字符断词去连字符 |
| 乱码 / 不可打印字符 | 扫描件 OCR 噪音 | 非法字符占比 > 30% 的块丢弃；NFC 归一化（不要 NFKC，会改掉中文标点） |
| 重复内容 | 同一段落多处出现，权重虚高 | 精确哈希去重 + 字符 5-gram Jaccard ≥ 0.9 近似去重 |
| 内容精简 | 长文档冗余 | 提炼核心知识点，去掉营销话术、免责声明 |

> 代码：`ingest/cleaning.py`（`clean_document` / `fix_linebreaks` / `drop_repeated_page_lines` / `dedup_blocks`）。

---

## 4. 阶段四：文本切片（提升检索命中率）

### 4.1 四种切法与实测数据

| 切法 | 做法 | 适用 | 问题 |
| --- | --- | --- | --- |
| 固定长度 | 按字符 / token 一刀切 | 纯叙述文本（公告、新闻稿） | 句子被拦腰砍断、表格被切散、函数定义和调用分家 |
| 递归 | 段落 → 句子 → 字符 | 一般文档 | 没有自然段落的流水账退化为固定长度 |
| **语义切块** | 相邻句 embedding 相似度骤降处下刀 | 主题切换明显的长文 | 建库时多一次 embedding 计算（离线可忽略） |
| 结构切块 | 按 Markdown 标题 / 章节 / 段落 | 制度、手册、技术文档 | 需要解析器给出层级（MinerU / Popo / Markdown） |
| 多模态 | 代码按 AST、表格按行 + 标明来源、图片先转描述 | 混合语料 | 已经是文档解析的范畴 |

实测（2000 页操作手册 + 500 个技术 wiki）：

| 切块方式 | Top-3 命中率 | Top-5 命中率 | 平均块大小 |
| --- | --- | --- | --- |
| 固定 500 字 | 61% | 71% | 492 字 |
| 固定 300 字 | 68% | 77% | 294 字 |
| 递归 500 字 | 74% | 83% | 463 字 |
| **语义切块** | **82%** | **91%** | 385 字 |

坑 1 的例子：退款政策 300 字按 200 切，"不符合退款条件的情形"和具体情形列表被切进两个 chunk，模型只能瞎编。

### 4.2 参数速查

- 中文块大小 **300-500 字 / 600-1000 token**；overlap **10%-15%**（太大会重复检索、浪费 token）；
- 产品经理真正要问的不是"chunk size 设多少"，而是"**用户的一个问题通常需要多大的知识单元才能被完整回答**"；
- 制度类文档：把规则和它的例外放在同一块（"住宿标准 600 元 … 部门负责人可审批上浮 20%"）。

### 4.3 Parent-Child（Small-to-Big）：检索用小块、阅读用大块

核心矛盾：检索需要小粒度（向量语义聚焦），LLM 需要大粒度（上下文完整）。

```
文档
├── 父 chunk（~500 token）  ← LLM 阅读
│   ├── 子 chunk A（~150 token）  ← 向量 / BM25 检索
│   ├── 子 chunk B
│   └── 子 chunk C
```

三种实现：Parent-Child（parent_id 关联）、摘要索引（LLM 为每段生成摘要建索引，命中后塞原文）、多粒度分层索引（章节 / 段落 / 句子三层，按问题类型选粒度）。

### 4.4 保留标题层级（坑 2：孤儿 chunk）

"该费用不予退还" —— 哪个费用？答案在两页之前。解法：给每个 chunk 附加**元数据面包屑**，拼进向量化文本：

```
[来源：员工手册 > 假期管理 > 年假]
员工每年享有 5 天带薪年假，需提前在 OA 系统提交申请……
```

这是性价比最高的一次性投入：即使块本身没有"年假"这个词，标题也能补上语义信号。

### 4.5 进阶

- **Late Chunking**：先对整篇 / 长段落 embed 得到 token 级表示，再在语义边界切分，保留长距离上下文；jina-embeddings-v3 支持，长篇技术文档再涨 3-5 个点。
- **Contextual Retrieval**（Anthropic 2024）：索引阶段为每个 chunk 加"这段在整篇文章里讲什么"的背景说明再一起 embed；arXiv 2604.01733：Contextual Dense 比普通 Dense 的 Recall@5 高 2.8pp（0.615 vs 0.587）。代价是索引阶段每块一次 LLM 调用。

> 代码：`ingest/chunking/structural.py`（标题栈 → 面包屑 → section → parent；表格 / 代码原子块；超长表格按行拆并重复表头）、`semantic.py`（相邻句向量距离 P80 断点）、`recursive.py`、`parent_child.py`（child 句子窗口 + 重叠）、`enrich.py`（Contextual Retrieval / 假设性问题）。`Chunk.embed_text()` = 面包屑 + 上下文摘要 + 假设性问题 + 正文。

---

## 5. 阶段五：索引优化（提升检索效率与准确率）

| 手段 | 做法 | 收益 |
| --- | --- | --- |
| 索引降噪 | 提取核心关键词，去掉索引中的无效成分 | 减少噪音命中 |
| 假设性问题 / QA 对 | LLM 为语料生成用户可能问的问题，与正文一起索引（"反向 HyDE"） | 拉近"问题 ↔ 文档"两种文体的向量距离 |
| 多级索引 | 长文档先建摘要索引过滤文档，再在文档内做细粒度检索 | 精度 |
| 混合索引 | 向量索引（语义）+ 关键词索引（BM25，精确） | 召回率（见第 6.2 节） |
| 元数据 | 文档类型、部门、产品线、生效时间、版本、权限、语言、来源 | 精准过滤 + 权限 + 可追溯；**元数据是 RAG 的结构化控制面** |
| 多重索引 | 按业务域 / 文档类型 / 租户 / 安全等级拆库 + 路由 | 不同类型内容不互相干扰；代价是路由错误与维护成本 |
| 索引算法（ANN） | HNSW 等 | 只解决规模 / 速度 / 成本；几万条数据不准的问题不在这里 |
| 知识图谱 | 抽取实体关系 | 多跳复杂查询 |
| 持续迭代 | 命中率 + 用户反馈 → 补盲点、剔过时 | 时效性 |

> 代码：`index/`（`NumpyVectorStore` / `QdrantVectorStore` / `ChromaVectorStore`、`BM25Store`、`ChunkStore` + 元数据过滤 `filter_child_ids`），`enrich.py` 假设性问题，`graph/unweaver.py` 实体关系，`retrieval/router.py` 多库路由，`retrieval.filter_expired`。Web 应用里每个知识库一个索引目录（`data/indexes/<kb_id>`）。

---

## 6. 阶段六：检索四层框架

索引层决定知识怎么**存**，查询层决定问题怎么**转**，召回层决定从哪些路径去**找**，重排序层决定最终谁**最相关**。比喻：仓库里放了什么 → 用什么钥匙开门 → 从哪几扇门进去 → 把最好的几件带出来。

### 6.1 查询层：用户怎么问 ≠ 系统应该怎么搜

| 方法 | 解决什么 | 做法 | 注意 |
| --- | --- | --- | --- |
| Query 改写 / 压缩 | 口语、指代不清、噪音词稀释向量 | "那啥我想问一下…租办公室的钱报销政策" → "办公室租金 报销政策" | 成本极低 |
| Multi-Query 扩展 | 提问角度和文档描述角度对不上 | LLM 改写成 3~5 个角度分别检索，合并去重（撒网捕鱼） | **原始问题必须保留**；覆盖率 +10~15% |
| 子问题分解 | 多跳问题 | "通讯补贴和交通补贴有什么不同" → 两个子问题分别检索再合并 | 多跳效果碾压式提升 |
| **HyDE** | 问题与文档文体不同 | LLM 先写一段假设答案，用其向量检索 | 短 query 零样本 +15~25%，只需 1 次 LLM 调用，temperature=0 |
| Step-back | 问题太具体、库里只有通用背景 | 先抽象成背景问题检索 | 两步走 |

**HyDE 两个反模式（必须避开）**：

1. 数值 / 精确查询：「2024 年 Q3 财报净利润那页」—— LLM 编的"净利润同比增长 23.5%"会把检索拉向"讨论增长率"的文档；arXiv 2604.01733 明确指出 HyDE 对数值查询有害。
2. 需要精确 metadata 匹配：「作者是张三的合同」—— 用 metadata filter，别绕远。

决策规则：**query 含精确数字、人名、编码、型号、日期 → 跳过 HyDE，直接双路 RRF 或 metadata filter。** 查询转换也可能理解错原问题，重要场景保留原始查询与改写查询一起检索。

> 代码：`retrieval/query_transform.py`（`QueryTransformer.plan`：rewrite / multi_query / decompose / hyde / step_back；`has_precise_tokens` 自动跳过 HyDE；原始查询始终保留）。

### 6.2 召回层：向量与 BM25 互相补盲

**纯向量的三个盲区**（精确 token 在 embedding 空间里没有独立位置，被泛化进周边语义簇）：

1. 专有名词和错误码：K8s Ingress 502、CVE-2024-3094、MySQL 1062 —— 搜 "502" 召回一堆 "5xx 通用文章"；
2. 新词汇和产品名：Claude Skills、MCP Server —— 通用 embedding 训练时没见过；
3. 长尾精确匹配：合同"第 3.2.1 条"、函数名 `handle_stream_response`、订单号 ORD20260831001。

**BM25 的盲区**：只认字面。"退货"和"申请售后"词不重叠，完全召不回。

| 查询类型 | 向量检索 | BM25 |
| --- | --- | --- |
| 语义问答（"怎么提升系统鲁棒性"） | ✅ | ⚠️ 依赖关键词 |
| 同义词泛化（"退货" vs "申请售后"） | ✅ | ❌ |
| 精确专有名词（"GPT-4o"、"ORD-9527"） | ⚠️ | ✅ |
| 型号 / 版本号 / 方法名 | ❌ | ✅ |
| 财务数字（"Q3 营收"、"净利率 18.3%"） | ❌ | ✅ |

**反直觉数据（arXiv 2604.01733，T2-RAGBench 23,088 条金融查询，p<0.001）**：

| 方法 | R@1 | R@5 | MRR@3 |
| --- | --- | --- | --- |
| 纯 Dense（text-embedding-3-large） | 0.248 | 0.587 | 0.351 |
| 纯 BM25（2009 年算法） | **0.293** | **0.644** | **0.411** |

术语密集领域（金融 / 法律 / IT 运维 / 医疗）BM25 可能直接比 SOTA 向量模型更准；一份内部评测里纯向量 Recall@10 0.72，加 BM25 双路后 0.84；双路 RRF 比单路 Dense Recall@5 高 10.8pp。

**BM25 三个工程要点**：

- TF 饱和：k1 控制，默认 1.5，长文档可调到 2.0；
- 长度归一化：b 默认 0.75，文档长度均匀可调到 0.5；短查询被长文档稀释时调小 b；
- IDF 天然过滤停用词，但加停用词表和分词器能再提 2-3 个点。

**中文分词是 BM25 的生死线**：索引和查询必须用同一套分词（jieba / pkuseg / ES 中文分词器）；专有名词（"A2024-001"）建议加"不过词"保护，或对查询里的编号做正则提取、子串匹配兜底。库：`rank_bm25`（简单）或 `bm25s`（快、可持久化）。

暴力调大 top-K 不能解决单路结构性漏召 —— 候选越大无关 chunk 越多，LLM 注意力越稀释。

> 代码：`index/bm25_store.py`（jieba + `PRECISE_TOKEN` 正则整词补充 + 可选精确 token 加分）、`index/embeddings.py`（BGE-M3 / OpenAI 兼容 / 哈希）、`retrieval/retriever.py::_recall`。

### 6.3 融合层：RRF vs 加权

分数不能直接比：余弦相似度理论 [-1,1]、实际集中 [0.3, 0.9]；BM25 无上界（12.7、8.3、5.1…）；直接相加向量贡献被淹没；min-max 归一化被异常值拉偏，α 从 0.5 调到 0.7 涨 3 个点，换一批数据又跌回去。

**RRF（Reciprocal Rank Fusion，Cormack et al. 2009）**只看排名：

```
score(doc) = Σ_i  w_i / (k + rank_i(doc))      k = 60（业界标准，几乎不用调）
```

| 文档 | 向量排名 | BM25 排名 | RRF | 最终 |
| --- | --- | --- | --- | --- |
| iPhone 14 参数页 | 2 | 1 | 1/62 + 1/61 = 0.0321 | 🥇 |
| 苹果产品历史 | 1 | 3 | 1/61 + 1/63 = 0.0321 | 🥇（同分） |
| 手机价格比较 | 4 | 2 | 1/64 + 1/62 = 0.0317 | 3 |

优点：不需要归一化、对分数分布不敏感、五行代码、k 是"保底分"避免偶尔失误直接淘汰；Elasticsearch 8.9+ hybrid、Weaviate、Qdrant 默认 RRF。

加权 / 凸组合何时用：确定两路分数分布稳定、要精调某路权重。经验值：中文技术文档 α = 0.6~0.7（向量为主）；FAQ / 合同 / 法规 α = 0.4~0.5（BM25 更高）。arXiv 2604.01733 消融：金融数据上等权凸组合 Recall@5 0.726 略优于 RRF k=60 的 0.695，有评估集时值得 A/B。加权 RRF：`w1/(60+rank_dense) + w2/(60+rank_bm25)`，起手 1:1；k 常见 20 / 40 / 60。

top_n 每路 50-100（比最终 top_k 大 5-10 倍）：太小两路取不全，太大融合层拿到一堆低质量长尾稀释高分文档。

> 代码：`retrieval/fusion.py`（`rrf_fusion` 带权重、`convex_fusion` min-max）。

### 6.4 精排层：Cross-Encoder Reranker

Bi-encoder（向量检索）query 和 chunk 各自编码再比距离 —— 只看了两个人的简历；Cross-encoder 把 (query, chunk) 拼成一对整体打分 —— 把两个人放一个房间观察怎么配合，精度高一个档次，但每个候选都要跑一次，只能对小规模候选精排。

流程：多路召回 → RRF → 候选 20~50 → Cross-encoder 逐对打分 → top 3~5 → prompt。不重排的坑 7："相关但不最相关"，第 3 条才是答案；加重排通常抬 10-15 个点。

**🔴 候选集不能低于 50**（arXiv 2604.01733）：

| 传给 Reranker 的候选数 | Recall@5 |
| --- | --- |
| 20 → top-10 | 0.458 |
| 50 → top-10 | **0.826** |
| 100 → top-10 | 0.888 |

为省延迟把候选压到 10~20，Reranker 形同虚设。延迟超限时先 50→30 看效果，而不是直接砍掉 Reranker。

模型：BGE-Reranker-v2-m3（BAAI，中英双语，开源首选）、BCE-Reranker；API：Cohere Rerank v3（跨语言强）、Jina Reranker。实战：2000 块中文库，混合检索 Top-3 78% → BGE-Reranker 89% → Cohere 93%（每 4 次翻车一次 → 每 9 次一次）。Pointwise（逐对打分，BGE）vs Listwise（RankGPT，考虑文档间关系，延迟高）。

**本项目的默认与实测**：默认本地交叉编码器 BAAI/bge-reranker-v2-m3，直接用 transformers 加载（不依赖 sentence-transformers）。
同一台 Apple Silicon 上：LLM 打分精排一次查询约 11.7 秒，bge-reranker-v2-m3 首次约 2.8 秒（加载 + 预热），之后约 0.17 秒（19 个候选，MPS）。
分数分布：无关文档 < 0.01，直接相关 0.6~0.9，但同义改写的相关文档只有 0.1~0.2 —— `min_score` 拒答阈值取 0.02~0.05，不能照搬 0.2。

**Cohere Rerank v2**：`POST https://api.cohere.com/v2/rerank`，Bearer 鉴权；请求 `{model, query, documents, top_n, max_tokens_per_doc}`（单次建议 ≤ 1000 篇，超长文档按 `max_tokens_per_doc` 默认 4096 截断），
响应 `results[{index, relevance_score}]`；模型 rerank-v4.0-pro（默认推荐）/ rerank-v4.0-fast（低延迟）/ rerank-v3.5 / rerank-multilingual-v3.0 / rerank-english-v3.0；模型列表接口是 `GET /v1/models?endpoint=rerank`。

> 代码：`retrieval/rerank.py`（`CrossEncoderReranker`：懒加载、权重不在本机时后台下载并由 LLM / 词法兜底、`pop_note()` 把"这次用了谁"写进检索链路；`CohereReranker`；`APIReranker`：Jina / SiliconFlow 等 `/rerank` 端点，429 / 5xx 重试、超 1000 篇分批；`LLMReranker`；`LexicalReranker`；`provider: auto` 按 API → 本地 → LLM → 词法选择），`config.retrieval.rerank.candidates = 50`，候选 < 50 时 trace 记录告警；`rag download-reranker`；`GET / POST /api/v1/models/reranker/{status,download}`。

### 6.5 粒度层：用于搜索的块 ≠ 用于回答的上下文

- **句子窗口检索**：命中一个句子，同时返回前后内容；
- **父子分块**：小块检索，命中后返回所属大块；
- **自动合并**：多个相邻小块同时命中时合并为完整上下文；
- **Lost in the Middle**：LLM 对上下文中部注意力显著下降，把最相关的放两头；
- **坑 8**：8 个 chunk 6000 token 全塞给模型，关键信息被淹没 —— 重排后只取 top 3-5，指令里明确"仅根据以下资料回答"；
- **Top-K 不是越大越好**：真正的问题是在保证正确答案被召回的前提下，系统最少需要多少上下文；
- **坑 5**：默认 topK=4、阈值 0.7 从不调参 —— 答不上（漏检）调高 topK / 降低阈值，答错（误检）反向操作，每次调完跑评测集。

> 代码：`retrieval/granularity.py`（`expand_contexts`：child→parent、相邻合并、句子窗口、token 预算、Lost-in-the-middle 排序）。

### 6.6 扩展层：在标准四层之上按需叠加

| 手段 | 解决什么 | 做法 | 代码 |
| --- | --- | --- | --- |
| 相对阈值（动态 top-K） | top-K 被弱相关内容凑满 | 分数低于 top-1 × ratio 的候选不进入上下文（`rerank.min_score_ratio`） | `retrieval/retriever.py` |
| MMR 多样化 | top-K 都是同一段的不同切片 | 在相关性与冗余之间取舍，λ = 0.7 | `retrieval/diversify.py` |
| 上下文压缩 | 大块里只有几句和问题有关 | 规则（零成本：按与 query 的词面相关度保留句子）或 LLM 逐字抽取 | `retrieval/compress.py` |
| UnWeaver 实体织网 | 孤儿 chunk：提到术语但定义在别处 | 索引时抽取"术语 → 定义"，检索后发现上下文提到术语而定义块未召回 → 自动织入 | `graph/unweaver.py` |
| 多跳检索 | 需要串联多个信息点 | 检索 → 判断还缺什么 → 再检索（最多 2-3 跳） | `retrieval/multihop.py` |
| Agentic RAG | 不同问题适合不同策略 | 先规划（精确 / 混合 / 多跳 / 不检索）→ 执行 → 反思是否需要补检；无 LLM 时按启发式 | `retrieval/agentic.py` |
| 查询路由 | 多知识库（多重索引） | 按知识库名称 + 描述 + 类别的向量相似度或 LLM 选择要查的库 | `retrieval/router.py` |
| 多轮归并 | 追问里的"它 / 那个" | 结合对话历史把追问改写成独立问题，再进入查询层 | `query_transform.py`（`condense_history`） |

这些都默认关闭或保守开启：先把"双路召回 + RRF + Rerank + Parent-Child"做稳，再用评测决定要不要加。

---

## 7. 生成层

- **LLM 处在链路末端**：它能更好地理解材料、组织语言，却无法修复上游没有召回、召回错误或知识过期。三步检查：正确答案是否在库里 → 是否被召回并进入上下文 → 模型是否正确使用了上下文；只有最后一步出问题才优化生成模型。
- **提示词是回答协议**而不是万能补丁：依据（只根据资料，给引用）、拒答（信息不足必须说不知道）、冲突（多份材料矛盾时指出而不是合并）、安全（不泄露系统提示与无权限内容）、格式（按业务结构输出，高风险加人工确认）。
- **"不知道"也是产品能力**（坑 6）：检索相似度都 0.5 以下时模型仍会编一个"看起来合理"的答案 —— 设相似度阈值 + 无结果兜底话术"知识库中暂未找到相关内容，已为您转人工"。拒答率不能单独当负面指标，要同时看"该答未答"和"不该答却答"。
- **可解释性是产品功能**：给用户引用来源、生效时间、文档版本；给运营研发完整的召回与生成日志。

> 代码：`generation/prompts.py`（回答协议）、`generation/answer.py`（引用解析、拒答）、`generation/llm.py`（Anthropic SDK：`claude-opus-5-5` 自适应思考 + effort + 流式 + 服务端 refusal fallback；OpenAI 兼容端点复用 MinerU-Popo 的本地 vLLM）、`schemas.RetrievalTrace`（每一层看到了什么）。

---

## 8. 评估体系：没有度量的调参都是盲修（坑 10）

### 8.1 四层指标

| 层 | 指标 |
| --- | --- |
| 检索层 | Recall@K、Top-K 命中率（Hit Rate）、MRR、NDCG、重排前后命中变化、无关内容比例 |
| 生成层 | 事实正确性、答案完整性、引用准确性、忠实度、拒答正确性 |
| 产品层 | 问题解决率、追问率、点赞点踩、转人工率、内容采纳率、任务完成时间 |
| 工程层 | 平均 / P95 响应时长、单次回答成本、索引更新时间、可用性、安全事件 |

单独追求某一指标会有副作用：增加召回数提高命中率，同时增加延迟、成本和噪声。目标是在准确率、体验、成本、风险之间找业务最优解。

### 8.2 黄金评估集

人工标注"问题 + 标准答案 + 应命中的文档 / chunk"，至少 100 条：常见问题 50%、边缘 / 长尾 30%、多跳 / 模糊 20%；另加无答案、权限敏感、时效冲突样例。每次改动检索策略都用同一份评估集跑一次，**指标不动就不合入**。工具：Ragas。

### 8.3 错误归因比指标更重要

| 错误类型 | 表现 | 优先攻的维度 |
| --- | --- | --- |
| 知识缺失 / 切块失败 | 任何一路都没召回；答案被切到两个块 | 语料、切片 |
| 语义漂移 | BM25 找到了、向量没找到 | embedding、面包屑、上下文摘要 |
| 关键词缺失 | 向量找到了、BM25 没找到 | 分词词典、精确 token 保护 |
| 融合失败 | 某路找到了，融合后掉出候选集 | RRF 权重、候选数 |
| 排序失败 | 候选里有，精排后被挤下去 | reranker、候选 ≥ 50 |
| Query 噪音 | 用户输入啰嗦，向量被无关词稀释 | Query 压缩 |
| 数据过期 / 冲突 | 新旧版本同时召回 | 元数据、时效过滤 |
| 模型未正确使用上下文 | 资料对但答错 | 提示词、模型 |

> 代码：`eval/`（`metrics.py`、`dataset.py` 黄金集 JSONL、`runner.py` 分阶段消融 dense / sparse / fused / final + 分类指标 + 错误归因），`rag eval -o report.md`。

---

## 9. 调优顺序、场景决策树与完整案例

### 9.1 顺序与时间分配

1. **先优化切块**（50% 时间）：最前置，改完不动下游代码；先把命中率从 60 拉到 80；
2. **再开混合检索**（20%）：补专有名词、编码、错误码；
3. **然后上重排**（20%）：Top-3 再拉 10 个点；确认候选 ≥ 50；
4. **最后 Query 改写**（10%）：唯一从前端入手的手段，检索合格后再做。

一个维度稳定之前调另一个会误判效果，每次只改一个关键变量。另一种视角（按实现成本 / 收益比）：召回层（双路 RRF）→ 精排层（Reranker）→ 查询层（HyDE / 扩展）→ 索引层（Parent-Child）。

产品化六步：定义核心场景与答案边界 → 建评估集与基线 → 优先治理数据与分块 → 建可解释检索链路 → 逐项验证高级策略 → 优化模型、提示词、成本与速度。**先让系统找到正确知识，再让模型基于正确知识回答，最后才是让答案更快、更便宜、更好看。**

### 9.2 场景决策树

| 场景特征 | 推荐配置 |
| --- | --- |
| 知识库有大量专有名词 / 型号 / 编码 | 向量 + BM25 双路（必须） |
| 用户提问多变、口语化 | 双路 + Multi-Query 扩展 + Query 改写 |
| query 普遍偏短（< 10 字）、非精确 | 双路 + HyDE |
| 召回质量要求极高（Recall@10 > 85%） | 三路全上 + Reranker（候选 ≥ 50） |
| 延迟敏感（< 50ms） | 双路 RRF，跳过 Reranker（p99 ≈ 15ms） |
| 知识库极小（< 100 条）、仅语义查询 | 纯向量 |
| 金融 / 法律 / IT 系统文档 | 适当提高 BM25 权重 |
| 数据量小、问题简单、纯同义查询 | 纯向量就够 |
| 问答质量直接决定业务结果 | 混合 + 重排 |

典型生产搭配：**Parent-Child 索引 + 向量 / BM25 多路召回 + Rerank 精排**，用户提问质量差再加 Query 改写。大多数场景混合检索（不加重排）已是"性价比拐点"，解决 80% 的"答不对"；重排是最后 20% 的锦上添花。

### 9.3 完整案例（企业制度知识库，模型全程不变：bge-large-zh）

| 步骤 | Top-3 命中率 |
| --- | --- |
| 初始：固定 500 字切块 + 纯向量 + 无重排 | 61% |
| 语义切块 + Parent-Child | 79% |
| + BM25 混合检索 | 84% |
| + BGE-Reranker | 91% |
| + Query 压缩 | 94% |

另一组（arXiv 2604.01733，23,088 条查询）：单路 58.7% → 双路 RRF +10.8pp → + Reranker 相对 +17.4% → 81.6%。

---

## 10. RAG 落地十坑速查

| # | 环节 | 坑 | 解法 |
| --- | --- | --- | --- |
| 1 | 切分 | 固定长度硬切，语义拦腰斩断 | 语义感知 / Markdown 标题层级切分 |
| 2 | 切分 | chunk 不带上下文，成为孤儿 | 元数据面包屑（文档标题 + 章节标题）拼进向量化文本 |
| 3 | 解析 | 表格和 PDF 直接当文本灌 | 结构化解析（MinerU / 导出 CSV）；查表走 Function Calling |
| 4 | 召回 | 只做向量检索 | 混合检索 BM25 + 向量（PGVector / ES 原生支持） |
| 5 | 召回 | topK / 阈值从不调参 | 按漏检 / 误检方向调，每次跑评测集 |
| 6 | 召回 | 库里没答案模型硬编 | 相似度阈值 + 兜底话术，敢说"不知道" |
| 7 | 重排 | 召回即采用 | 加 rerank，准确率 +10-15 点 |
| 8 | 上下文 | 一股脑全塞 | 重排后 top 3-5，"仅根据以下资料回答" |
| 9 | 运营 | 知识库只进不出 | 生效时间 / 版本元数据，检索时过滤，定期除草 |
| 10 | 运营 | 没有度量 | 评测集 + recall@5，指标不动不合入 |

心法：**RAG 不是"接个向量库"的功能，是一条需要度量、持续运营的数据管线。**

---

## 11. 参数速查表

| 参数 | 推荐值 | 出处 / 说明 |
| --- | --- | --- |
| child chunk | 100-150 token | Parent-Child |
| parent chunk | 500-800 token（中文 300-500 字） | 检索 / 阅读粒度 |
| overlap | 10%-15% | 边界截断 vs 重复 |
| 语义切块断点 | 相邻句距离 P80+ | LangChain SemanticChunker 同款 |
| 每路 top-N | 50-100 | 比 top-K 大 5-10 倍 |
| RRF k | 60（可试 20 / 40） | Cormack 2009 |
| 凸组合 α | 0.5 起手；技术文档 0.6-0.7；FAQ / 合同 0.4-0.5 | 有评估集再调 |
| BM25 k1 / b | 1.5 / 0.75（长文档 k1=2.0；均匀文档 b=0.5） | |
| Reranker 候选 | ≥ 50（延迟紧张 → 30，不要砍掉） | arXiv 2604.01733 |
| 最终 top-K | 3-5 | Lost in the Middle |
| 相对阈值 min_score_ratio | 0.5 左右（按评测调） | 低于 top-1 × ratio 的候选不凑数 |
| MMR λ | 0.7 | 1 = 只看相关性，0 = 只看多样性 |
| 覆盖分析阈值 | 高匹配 ≥ 0.6，部分匹配 ≥ 0.35 | 第 1.5 节；配置真实模型后按业务校准 |
| MinerU 档位 | 建索引 flash；扫描件 basic；复杂版面 standard | 第 2.2 节 |
| Multi-Query | 3-5 条，保留原始 | |
| HyDE | 短 query（< 30 字）、非精确；temperature 0；约 100-200 字 | 数值 / 编号查询跳过 |
| 黄金集 | ≥ 100 条：50% 常见 / 30% 长尾 / 20% 多跳 | |
| 双路 RRF p99 | ≈ 15ms | 工程经验 |

---

## 12. 本项目代码映射

### 12.1 核心库 `prism/`

| 技术点 | 模块 |
| --- | --- |
| 意图 × 知识覆盖分析（高 / 部分 / 无匹配，未命中与过期文档） | `eval/coverage.py` |
| 语料身份证 / 时效过滤 | `schemas.DocMetadata`、`retrieval.filter_expired`、Markdown front matter |
| MinerU 4.x 解析（本地 SDK / API server，四档位） | `ingest/parsers/mineru_parser.py` |
| Markdown 必做 / Excel 行级 / Docx / PPTX / PDF 兜底 | `ingest/parsers/` |
| HTML 表格 → Markdown（rowspan / colspan） | `ingest/parsers/html_table.py` |
| MinerU-Popo 桥接（三种输入适配 / 校验 / 文档树导入） | `ingest/popo_bridge.py`、`scripts/run_popo.sh` |
| 清洗：页眉页脚 / 断行 / 乱码 / 去重 | `ingest/cleaning.py` |
| 结构切片 + 面包屑 + 原子块 | `ingest/chunking/structural.py` |
| 语义切块 / 递归切块 | `ingest/chunking/semantic.py`、`recursive.py` |
| Parent-Child、表格重复表头 | `ingest/chunking/parent_child.py` |
| Contextual Retrieval / 假设性问题 | `ingest/enrich.py` |
| 向量索引（BGE-M3 / OpenAI 兼容 / 哈希；numpy / Qdrant / Chroma） | `index/embeddings.py`、`index/vector_store.py` |
| BM25 + jieba + 精确 token 保护 | `index/bm25_store.py` |
| 元数据过滤 / 相邻块 | `index/doc_store.py` |
| 多轮归并 / 改写 / Multi-Query / 分解 / HyDE（反模式跳过） / Step-back | `retrieval/query_transform.py` |
| RRF / 凸组合 | `retrieval/fusion.py` |
| 重排（Cross-Encoder / API / LLM / 词法；候选 ≥ 50） | `retrieval/rerank.py` |
| MMR 多样化 | `retrieval/diversify.py` |
| Small-to-Big / 自动合并 / 句子窗口 / Lost-in-the-middle | `retrieval/granularity.py` |
| 上下文压缩（规则 / LLM） | `retrieval/compress.py` |
| UnWeaver 实体织网 | `graph/unweaver.py` |
| 多跳 / Agentic RAG / 查询路由 | `retrieval/multihop.py`、`agentic.py`、`router.py` |
| 漏斗式主链路 + 可解释 Trace | `retrieval/retriever.py`、`schemas.RetrievalTrace` |
| 回答协议 / 拒答 / 引用 / 流式 / 多模态 | `generation/`（Anthropic SDK、OpenAI 兼容） |
| 黄金集 / 指标 / 消融 / 错误归因 | `eval/dataset.py`、`metrics.py`、`runner.py` |
| 命令行 | `cli.py`：`rag ingest / query / ask / coverage / eval / popo-export / serve / stats` |

### 12.2 后端 `app/`（FastAPI + SQLite）

| 能力 | 接口（前缀 `/api/v1`） | 实现 |
| --- | --- | --- |
| 运行状态 / 解析能力矩阵 | `GET /system/status`、`GET /system/parsing` | `api/system.py`、`services/model_service.py` |
| 知识库 CRUD / 统计 / 实体 / 重建索引 | `/knowledge-bases/...` | `api/knowledge_bases.py`、`services/index_service.py`（每库一个 Index，带缓存与写锁） |
| 文档上传（后台解析索引）/ 切片预览 / Markdown 校对回灌 / 元数据 / 重新解析 / Popo / 临时解析 | `/documents/...` | `api/documents.py`、`services/document_service.py` |
| 检索实验室（完整链路 + 按请求覆盖参数）/ 查询路由 | `POST /retrieval/{kb}/playground`、`POST /retrieval/route` | `services/retrieval_service.py` |
| 对话（SSE：meta → plan → sources → relations → message* → done）/ 会话 | `POST /chat/stream`、`/chat/sessions...` | `services/chat_service.py` |
| 模型配置中心（预设、CRUD、激活、拉取列表、连接测试） | `/models/...` | `api/models.py`；解析优先级：激活配置 → `.env` → `config.yaml` → 离线兜底 |
| 策略配置（基线 + 数据库覆盖） | `GET/PUT /settings/`、`POST /settings/reset` | `services/settings_service.py` |
| 评测 / 覆盖分析 | `POST /eval/{kb}/run`、`GET /eval/{kb}/reports`、`POST/GET /eval/{kb}/coverage` | `services/eval_service.py` |

### 12.3 前端 `frontend/`（React + Vite）

| 页面 | 对应本文章节 |
| --- | --- |
| 总览：六项核心能力、管线、运行时、漏斗 | 第 0 节 |
| 01 知识库：上传、切片预览、校对、元数据、实体、解析能力、Popo | 第 2-5 节 |
| 02 语料覆盖 | 第 1 节 |
| 03 检索实验室 | 第 6 节 |
| 04 智能问答 | 第 7 节 |
| 05 评测中心 | 第 8 节 |
| 模型配置 / 策略配置 | 第 11 节参数速查 |

### 12.1 "一张图讲清 RAG 的完整架构"对照与 LanceDB 落地

| 图中环节 | 本项目实现 | LanceDB 的角色 |
| --- | --- | --- |
| ① 多源数据接入（PDF / Word / PPT / 网页 / 表格） | MinerU 4.x + 内置解析器，Excel / CSV 进 SQL 路 | — |
| ② 解析与智能分块（视觉解析、语义边界、父子块） | MinerU-Popo 文档树、结构 / 语义切片、Parent-Child | 多模态表 `mm`（每库一张）：图片字节 + CLIP 向量 + 图注 / OCR 与正文同表，接入线上图文检索（文搜图的全文路、图搜文的混合路） |
| ③ 向量化 | BGE-M3（1024 维） | 向量列 `fixed_size_list<float32>`，换模型自动重算 |
| ④ 三路索引 + 元数据（向量 / BM25 / 知识图谱；权限、时效、版本） | 向量 + BM25 + UnWeaver 实体织网；权限 / 时效元数据过滤 | `vector_store.backend: lancedb`：同表向量 + ICU 全文索引，权限走 SQL 预过滤，每次写入一个版本（可时间旅行） |
| 在线 ① Query 改写（纠错、HyDE、拆子问题） | 查询层：改写 / Multi-Query / 分解 / HyDE（辅助模型 deepseek-flash、关思考、并发） | — |
| 在线 ② 混合检索（双路 + RRF Top-50） | 多路召回 + RRF | LanceDB 原生混合检索（向量 + 全文 + RRF），控制台可直接对比 |
| 在线 ③ 重排（Cross-Encoder，留 Top 3~5） | bge-reranker-v2-m3 / Cohere，`min_score` 拒答 | — |
| 在线 ④ 组装生成（只基于资料作答、查不到就说） | 回答协议 + 拒答；长期记忆只用于个性化 | `memories` 表：按用户召回身份 / 偏好 |
| 在线 ⑤ 引用溯源（文档、页码、权限过滤） | [n] 引用可点击、页码、权限过滤 | — |
| Agentic RAG：证据不够 → 改写重查 | 多跳 / Agentic；**多智能体协作**：审校判定证据不足时给出改写问句，研究员重查 | `agent_board` 共享黑板：证据按子任务语义取回，运行可回放 |
| RAG 3.0：图谱 + Agentic | UnWeaver 实体织网 + 多智能体 | — |

LanceDB 官方示例集（vectordb-recipes，《基于 LanceDB 的 80+ 开箱即用的 GenAI 示例教程》）的几类场景在本项目的对应：
**RAG 与向量搜索**（混合检索、重排、HyDE、上下文压缩、Agentic RAG）→ 检索链路 + LanceDB 向量后端 / 控制台；
**多模态**（CLIP 跨图文检索）→ 多模态表（已接入线上文搜图 / 图搜文 / 回答附图）；**AI 智能体**（多智能体协作、减少幻觉）→ 多智能体 + 审校回路；
**推荐系统**（商品 / 文章 / 论文推荐）→ 个性化推荐；**聊天机器人的记忆** → 长期记忆。详见 README「LanceDB」一节。

---

## 13. 参考来源

1. 宇翊《混合检索：BM25 与向量》（走在大数据架构路上的笔，2026-09-04）— https://mp.weixin.qq.com/s/BHzWrIxyH36gH01Mc06s9g
2. 路灯下没有行人《RAG 检索精度上不去？混合检索 + 重排，效果立竿见影》（2026-08-20）— https://mp.weixin.qq.com/s/c_L4Y3Clj2BTiLtdX-TV9Q
3. Seven《RAG 多路召回：让检索命中率从 60% 干到 85% 的 4 层优化》（LLM大模型Seven，2026-07-05）— https://mp.weixin.qq.com/s/eSAspfrmq18nE7Q43J-Ffg
4. AI小宇宙的漫游《RAG 生产级调优》（2026-08-23，基于 Leonie Monigatti 12 种调优策略）— https://mp.weixin.qq.com/s/CrnFMMn7NZUyzIqSANAppQ
5. 小林coding《阿里 Agent 岗二面："RAG 检索效果不好，你怎么优化？"》（2026-05-05）— https://mp.weixin.qq.com/s/ipMjYyYFl3pxbaX9Hl1yaw
6. merlinfeng《RAG 检索质量调优：从能用到好用》（码农札记，2026-07-29）— https://mp.weixin.qq.com/s/e9iMg5nLdOgZyqPDpxmnzQ
7. JAVAHOME《RAG 落地的 10 个坑：切分、召回、重排，每一步都有讲究》（2026-09-26）— https://mp.weixin.qq.com/s/CVYwAFBga6K-LuJ2HpecbA
8. 糖醋小黄鱼《从"读懂一页"到"读懂一篇"：MinerU 与 MinerU-Popo 文档解析全解析》（小黄学AI，2026-09-20）— https://mp.weixin.qq.com/s/aZ0aZX-s7q7c8QSjhS31AA
9. Akarsu et al., *From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents*, arXiv:2604.01733（T2-RAGBench，23,088 条查询）
10. Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009
11. MinerU 4.x 文档：https://opendatalab.github.io/MinerU/ （Quick Start / Python SDK / Tiers / Output Files / Migration to 4.0）；PyPI `mineru` 4.0.10
12. MinerU-Popo：https://github.com/opendatalab/MinerU-Popo ；模型 https://huggingface.co/DreamEternal/MinerU-Popo ；arXiv:2605.24973
13. Anthropic, *Contextual Retrieval*（2024）
14. Lance & LanceDB《建议收藏！基于 LanceDB 的 80+ 开箱即用的 GenAI 示例教程！》（2026-01-06，译自 lancedb/vectordb-recipes）— https://mp.weixin.qq.com/s/i4jSDxE5tCuWoRg70NKEoQ
15. LanceDB 文档（OSS 0.39：Hybrid Search / FTS Index（ICU、jieba 分词）/ Vector Index / Scalar Index / Reranking / Versioning / Multimodal Blobs）— https://docs.lancedb.com/
