# 千万级知识库：存储形态、批量导入与压测

单个知识库到千万级段落（child）时，原来"全部常驻内存、保存时整份重写"的设计会先在这几处垮掉：

| 环节 | 原来的做法 | 千万级时的问题 | 现在 |
|---|---|---|---|
| 段落存储 | 全部 `Chunk` 常驻内存，每传一篇文档整份重写 `chunks.jsonl` | 十几 GB 内存；每次上传写几 GB | `storage.chunk_store: sqlite`：按 id 读盘（LRU 缓存），每篇文档一个事务增量提交，zlib 压缩 |
| 关键词路 | bm25s：全部分词结果在内存，增删任何文档整库重建 | 内存 + 每次删改全库重建 | LanceDB 全文索引建在 **jieba 预分词列** 上（与 BM25 同一套分词，含编号精确 token），增量维护 |
| 时效 / 权限 / 元数据过滤 | 每次查询遍历全部段落，拼出"可见段落 id 集合"再传给向量库 | 每次查询几秒；IN 列表放不下 | `ChunkScope`：按**文档**记允许 / 排除（取小的那份），按条件缓存；只排除少数文档时多取再过滤（O(1)），允许列表下推成 `doc_id IN (…)` 预过滤 |
| 向量 ANN | IVF 分区数用 Lance 默认、`nprobes` 固定 20 | 分区随数据增长，固定探查数召回悄悄变差 | 分区数 √n/4（IVF_HNSW_SQ），探查 ≥ `nprobes_ratio`（默认 0.3）比例的分区；数据涨到 4 倍重训 |
| 写向量 | 每篇文档 `merge_insert`（按 id 与整表连接） | 千万行时每篇几秒 | 同一文档重新入库前已整篇删除，直接追加；删除按 `doc_id` 一条语句 |
| 导入 | 只能逐篇上传、每篇保存一次索引 | 光"每篇保存"就要几天 | `scripts/bulk_import.py`：跨文档成批向量化 / 多进程分词 / 成批写入，索引最后建一次，断点续传 |
| 整库重建 | 改入库参数自动逐篇重建 | 千万级要好几天，且在后台悄悄开始 | 超过 `storage.auto_rebuild_max_chunks`（默认 200 万）只记"待重建"，用 `bulk_import.py --rebuild` |
| 每次检索的文档名 | 每次把知识库全部文档行整行读出（含正文） | 几十万篇文档时每次搬 GB 级数据 | 只读 id / 文件名两列，只查本次结果涉及的文档 |
| 自查询（元数据条件） | 每次查询遍历全部文档统计取值 | O(文档数)/次 | 按范围缓存 |
| 文档元数据的过滤与统计 | 逐篇拿 `DocMetadata` 判断（Python 循环） | 100 万篇：过滤一遍 0.2 s、统计取值 1.5 s，对象占 1.8 GB；亿级约 500 万篇 | `DocIndex` 列式索引（numpy）：100 万篇过滤 1 ms、按条件匹配 22 ms、统计取值 48 ms，索引约 200 MB；语义与逐篇判断逐项对拍（tests/test_doc_index.py） |
| 「知识库」页文档列表 | 一次返回全部文档行（含正文） | 几十万篇时页面卡死 | `/documents/page` 分页 + 按文件名 / 标题搜索；轮询按整库"处理中"计数，不再扫全列表 |
| 实体织网（`unweaver_enabled` 默认开） | 每条上下文对全部实体关系做子串匹配 | 关系几万 ~ 几十万条时每次查询多出几十毫秒：20 万条 × 5 条上下文 36 ms | 关系表按实体名建索引，枚举正文子串查表：0.2 ms，不随关系数增长；语义与逐条匹配逐项对拍（tests/test_entity_index.py） |
| 向量后端 | numpy / LanceDB / Qdrant / Chroma，都在一台机器上 | 亿级索引放不进一台机器 | 新增 **Milvus** 后端：嵌入式 Milvus Lite（无服务）或 Standalone / 集群 / Zilliz Cloud，文档级过滤下推成 `doc_id in / not in` 表达式 |

批量导入还踩到过一个坑，已修：批次够大时导入结束不合并碎片，但随后保存索引时"写入满 `optimize_every` 次自动合并"仍会触发，
整表重写一遍、旧版本按天保留，千万级实测 LanceDB 从约 30 GB 涨到 69 GB、把压测机器的磁盘写满。现在批量导入结束时不会再触发这次合并；
需要合并时（每批很小）合并完立即清掉旧版本。

另外修了一个已有的问题：LanceDB 后端用 IVF_HNSW_SQ 索引时，`refine_factor × 每路取回数` 超过 `ef_search`（默认 5 × 50 = 250 > 128）
Lance 会直接报错——任何超过 2 万段落（`index_min_rows`）的 LanceDB 知识库，向量检索都会失败。现在 `ef` 自动不小于实际取回数。

## 怎么用

### 配置（四项一起改）

```yaml
storage:
  chunk_store: sqlite
vector_store:
  backend: lancedb
  lancedb:
    fts_tokenizer: jieba
retrieval:
  sparse_backend: lancedb_fts
```

也可以在「策略配置」页改（新增「存储形态（千万级）」一组）。`storage` 段**不触发重建**：加载索引时自动迁移段落存储（jsonl ↔ SQLite）；
后三项是建索引期参数，改了会重建——大知识库按上面说的不会自动开始，用 `--rebuild`。

### 批量导入

```bash
# JSONL：每行 {"id", "title", "text", 可选 "source" / "version" / "valid_from" / "valid_to" / "tags" / "department" / "access_level" / "allowed_departments"}
.venv/bin/python scripts/bulk_import.py --kb <kb_id> docs.jsonl --embed-workers 8 --tokenize-workers 8
# 目录：按文件类型解析（与网页上传同一套解析器；不处理文档里的图片）
.venv/bin/python scripts/bulk_import.py --kb <kb_id> /path/to/docs/
# 改了入库参数 / 换了向量模型后，批量重建大知识库（新索引建在 <索引目录>.rebuild，建好后替换，旧索引改名留底）
.venv/bin/python scripts/bulk_import.py --kb <kb_id> --rebuild
```

- 切片与网页上传完全一致；导入的文档出现在「知识库」页（JSONL 的正文存进文档行，目录导入把文件硬链接进上传目录——整库重建要靠它们）。
- 中断后重跑同一条命令：已导入的跳过；写到一半的那批（journal 里没有"已提交"）先从 LanceDB / SQLite 里清掉再重导，不会有半批数据或重复向量。
- `--embed-workers`：API 向量模型时并发请求；本地模型保持 1（GPU 上一个大批次比多线程快）。
- 导入期间不要在网页上改这个知识库（两个进程同时写同一个索引）；导入完成后运行中的服务会自动加载新索引。

### 压测

```bash
.venv/bin/python scripts/scale_bench.py build --dir /data/bench --children 10000000 --dim 1024
.venv/bin/python scripts/scale_bench.py query --dir /data/bench        # 新进程里测冷启动、常驻内存、延迟、召回
```

## 压测结果

机器：MacBook（Apple Silicon，16 核，48 GB 内存，NVMe）。向量是现成给的（不跑向量模型），所以导入吞吐**不含向量化**。

合成数据：按 jieba 词典词频抽词拼成中文段落（每个 child 约 60 词，入库时照常用 jieba 重新分词）；向量 = 512 个簇中心 + 48 维低秩语义方向 + 小噪声。
没有用纯随机向量：高维各向同性随机向量"人人等距"，近邻没有意义，原型实测 ANN 召回只有 0.24，会严重低估。
每篇文档 5 个 parent × 4 个 child；5% 的文档已过期、10% 是机密（access_level 2），所以"带过滤"的检索确实在排除文档。
查询：从库里随机挑一个 child，取它的向量加噪声当查询向量、正文里的几个词当查询文本（已知答案）。

| | 100 万段落 × 1024 维 | 1000 万段落 × 256 维 |
|---|---|---|
| 文档 / 段落（parent + child） | 5 万 / 125 万 | 50 万 / 1250 万 |
| 导入吞吐（不含向量化） | 7,365 child/s | 9,212 child/s |
| 导入总耗时（含建 ANN / 全文 / BTREE 索引） | 3.3 分钟 | 24 分钟（建索引 6 分钟） |
| 导入峰值内存 | 5.0 GB | 9.2 GB（训练 IVF 时） |
| 磁盘：LanceDB / SQLite | 6.9 GB / 2.0 GB | 21.7 GB / 19.5 GB |
| 冷启动：加载索引 / 首次查询 | 0.8 s / 0.45 s | 3.0 s / 1.3 s |
| 常驻内存（100 次查询后） | 2.2 GB | 8.6 GB |
| IVF 分区 / 每次探查 | 250 / 75 | 790 / 237 |
| 向量 p50 / p95 | 10 / 27 ms | 17 / 108 ms |
| 向量 + 过滤 p50 / p95 | 21 / 28 ms | 39 / 41 ms |
| 全文 p50 / p95 | 2.8 / 5.4 ms | 8.5 / 16 ms |
| 全文 + 过滤 p50 / p95 | 3.1 / 3.9 ms | 9.5 / 15 ms |
| **完整检索（带时效 + 权限）p50 / p95** | **33 / 53 ms** | **72 / 97 ms** |
| 向量 recall@10（ANN vs 精确全扫描） | 0.965 | 0.99 |
| 全文命中原文 @10 | 0.97 | 0.94 |

千万级那次用 256 维，是因为压测机的磁盘只剩约 90 GB：1024 维的千万级索引要 80 GB 以上，再加建索引的临时空间会把磁盘写满
（第一次跑 512 维就因为上面那个合并的 bug 写满过）。段落数决定的开销（过滤、全文、SQLite、IVF 分区数、文档元数据）都是按 1000 万实测的；
向量字节数与维度成正比，1024 维的数字按 100 万 × 1024 维那组外推（见下）。

修改前（同一份 100 万数据，代码改动前的形态）没有可比的数字：内存版块存储 + bm25s 要把 125 万段落和全部分词结果装进内存，
每次查询还要遍历全部段落拼过滤集合——这正是这轮改造要去掉的。

指标说明：
- **向量 / 全文**：单路召回（取 50 条）；**+过滤**：带时效 + 权限范围；**完整检索**：`HybridRetriever.retrieve`（多路召回 + RRF 融合 + 粒度扩展 + 织网，
  带时效 + 权限），不含精排模型（cross-encoder 精排 50 条候选的耗时与库大小无关，按所用模型另计）。
- **recall@10**：ANN 结果与精确全扫描结果的重合度；**全文命中原文**：用原段落里的词检索，原段落在前 10 的比例。
- **常驻内存**：新进程加载索引、跑完全部延迟测试后的峰值（不含召回测量里的精确全扫描）。

### 向量索引：IVF_HNSW_SQ 还是 IVF_PQ

同一份 100 万段落 × 1024 维，两种 ANN 索引（`vector_store.lancedb.index_type`）：

| | IVF_HNSW_SQ（默认） | IVF_PQ |
|---|---|---|
| 索引文件 | 2.6 GB | 0.3 GB |
| 常驻内存（100 次查询后） | 2.2 GB | 0.9 GB |
| 向量 p50（探查 30% 分区） | 10 ms | 21–23 ms |
| recall@10 | 0.965 | refine 5 / 20 / 50 → 0.72 / 0.90 / 0.94 |
| 建索引 | 63 s | 43 s |

IVF_PQ 把 1024 维压成 64 字节，索引小约 9 倍、内存省一半多，但粗排误差大，召回全靠 `refine_factor`（多取 k × refine 条原始向量重排）补：
默认的 5 只有 0.72，30 以上才到 0.9+，而重排只是多读几百条原始向量，耗时几乎不变。所以 IVF_PQ 下 `refine_factor` 自动不低于 30；
再加大探查比例（60%、100%）召回不再提高、耗时翻倍——瓶颈在量化误差，不在分区覆盖。

选择：千万级单机内存够（上表）用默认的 IVF_HNSW_SQ；往亿级走、索引缓存放不进内存时换 IVF_PQ，索引与内存降到约 1/9 和 1/2，
召回约 0.94，向量路慢 10 ms 左右（完整检索里向量只是多路之一，p50 影响不到 15 ms）。

### 推到 1 亿

按实测外推（1024 维；磁盘与内存按 100 万 × 1024 维的实测线性放大，与 1000 万 × 256 维换算的结果一致）：

| | 1000 万段落 | 1 亿段落 |
|---|---|---|
| 磁盘（LanceDB + SQLite） | 约 85 GB | 约 850 GB |
| 常驻内存：IVF_HNSW_SQ | 约 20 GB | 约 200 GB |
| 常驻内存：IVF_PQ（召回约 0.94） | 约 9 GB | 约 90 GB |
| 导入（不含向量化） | 约 30–45 分钟 | 约 8–12 小时 |
| 向量化 | 段落数 ÷ 每秒向量化条数 | 例：每秒 2,000 条约 14 小时，每秒 500 条约 2.3 天 |

结论：**千万级单机可用**——一台 32 GB 内存、200 GB 以上 SSD 的机器，带过滤的完整检索 p50 在 100 ms 以内（上表实测，不含精排模型）。
**亿级没有实测**，按外推单机要约 1 TB 磁盘、IVF_HNSW_SQ 约 200 GB 内存（IVF_PQ 约 90 GB），已超出一台机器的舒适区：
建议按业务拆成多个知识库（跨库路由已有），或把向量 / 全文换成外部服务（见下）。文档元数据这一侧不再是瓶颈：500 万篇文档的列式索引约 1 GB，
过滤与统计都在百毫秒内。

## 接外部服务：Milvus（向量）

```yaml
vector_store:
  backend: milvus
  milvus_uri: http://milvus:19530     # 留空 = 每个知识库一个嵌入式 Milvus Lite 文件 <index_dir>/milvus.db；https://… 连 Zilliz Cloud
  milvus_token: user:password          # 集群鉴权，可空
```

安装：`pip install 'pymilvus[milvus_lite]'`（嵌入式）或 `pip install pymilvus`（只连集群）；项目 extras：`rag-agent-studio[milvus]`。

- 每个知识库一个 collection（`<collection_prefix>_<知识库目录名>`），列 id / doc_id / vector，索引 AUTOINDEX（集群上由服务端选 HNSW / DiskANN），余弦度量；
- 时效 / 权限 / 元数据过滤（`ChunkScope`）下推成 `doc_id in [...]` / `doc_id not in [...]`，先过滤再找最近邻；整篇删除 `doc_id == "..."` 一条语句；
- 远程 Milvus 的 collection 跟着知识库走、不跟目录走：整库重建改为原地重建（与 Qdrant 同一套：旧索引挪到备份，建好替换）；
- Milvus 不存正文：关键词路仍是内存 BM25 或 LanceDB 全文（`vector_store.backend: milvus` 时只能用 BM25）。**亿级的全文检索建议接 Elasticsearch / OpenSearch**，
  本轮没有实现——接法与 Milvus 相同：`search(query, top_n, allowed_ids)` 接口 + 把 `ChunkScope` 翻译成 `terms` / `must_not` 过滤。

实测（嵌入式 Milvus Lite，30 万段落 × 256 维，同一台 Mac）：导入 3,474 child/s，向量 p50 76 ms（LanceDB 同机 100 万条才 10 ms），
带过滤的完整检索 p50 181 ms，recall@10 1.0，常驻内存 3.6 GB。**Milvus Lite 只适合开发和测试**——它把数据全装进内存、查询也慢；
后端的意义在于连 Milvus 集群，把亿级索引分散到多台机器上。集群上的延迟取决于部署，这里没有实测。

两个注意事项：
- `milvus_lite` 的原生库一旦载入进程，同一进程里再跑 onnxruntime（rapidocr OCR）会段错误（本项目测试实测复现）。
  所以 Milvus 的测试默认跳过，`RAG_TEST_MILVUS=1 pytest tests/test_milvus.py` 单独跑；生产上用服务端 Milvus（pymilvus 客户端本身没有这个问题）。
- 新进程打开已有 collection 时它处于 released 状态，后端会自动 `load_collection`；集群上首次 load 要把索引读进内存，大 collection 需要几分钟。

其他后端同样可插拔：`NumpyVectorStore` / `LanceDBVectorStore` / `QdrantVectorStore` / `ChromaVectorStore` / `MilvusVectorStore` 同一接口
（add / search / remove / remove_doc / get_vectors / save）。

## 已知限制

- `GET /documents/?kbId=` 仍一次返回全部文档（脚本 / 旧调用方用）；界面已改用分页接口。
- **全库分析类功能仍然遍历全库**：语料体检、别名发现、个性化推荐、LanceDB 镜像、覆盖率评测（检索链路之外，跑得动但慢、占内存）。
- **辅助索引（摘要 / 分层）与图片索引的 BM25 仍在内存**；批量导入不处理文档里的图片。
- **实体图（EntityStore）在内存**（已按实体名建索引，查询不再扫全表），只是没有下盘；LLM 抽取的三元组很密时 entities.json 会大。
- `--rebuild` 期间网页上对这个知识库的写入不会进新索引。
- 当前阿里云服务器（2 核 4 GB）放不下千万级：按上表至少需要 32 GB 内存、200 GB 以上 SSD 的机器。
