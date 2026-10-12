# RAG 难负样本挖掘与微调

目标是改善业务问题的召回与排序。当前实现提供生产索引上的候选挖掘、审核与训练数据导出，以及可选的真实 embedding / reranker 微调。是否提升业务效果，需要独立黄金集验证；程序能完成训练、loss 下降都不能替代检索评测。

## 0. 两篇文章讲了什么，本实现对应到哪里

- [《RAG最难的地方，不在检索，也不在生成（下篇）》](https://mp.weixin.qq.com/s/zBDNhtnA2aRM17GRXPCU6w)：embedding 用（锚点、正例、负例）三元组训练；完全不相关的简单负例价值不大，有价值的是“看起来相似、其实不同”的难负例；难负例靠“模型生成 + 人工筛选”，并引用 Conan-Embedding 的分阶段做法（先用高相似负例，训练中不断替换）。
- [《万字长文详解优图RAG技术》](https://mp.weixin.qq.com/s/tGL6N4r94z05oor7eSso-w)：先构造（问题，相关文档）对，再从语料挖难负例；用 Reranker 给训练语料打分做三件事——剔除分数极低的伪正例、按相关性分布滤掉简单负例、识别难负例里的潜在正样本并替换；Reranker 分数再用于 embedding 的 label 层知识蒸馏。Reranker 自身用对比损失加蒸馏训练。

| 文章里的做法 | 本仓库 | 说明 |
| --- | --- | --- |
| 从语料挖“相似但不同”的负例 | 已实现 | dense + BM25 融合召回，排除已知正例、同 parent、近重复 |
| 剔除伪正例 | 已实现 | `min_positive_score` |
| 滤掉简单负例 | 已实现 | `min_negative_score`，默认 0（不过滤），见第 2 节 |
| 识别负例里的潜在正样本并替换 | 已实现 | 教师分贴近正例的候选不作负例，由下一名候选顶上；并列入 `suspected_positives` 供补标 |
| Reranker 分数做 label 层蒸馏 | 已实现 | 训练集带 `pos_scores` / `neg_scores`，`--distill-weight` 开启，见第 4 节 |
| 训练中动态替换难负例（Conan） | 只有多轮流程 | 没有训练内替换；可按第 6 节用上一轮模型重挖 |
| 大模型改写正例生成难负例 | 已实现 | `synthetic_negatives`，默认 0（不生成）；结果单独存放，人工筛选后才导出，见第 2 节 |
| 大模型生成问题构造（问题，文档）对 | 部分 | `scripts/synth_eval.py` 生成的是评测题；用作训练标注前要和测试集分开 |
| 大模型识别假负例 | 未实现 | 目前只有 Reranker 教师过滤加人工复核 |
| 跨设备负样本、指令感知多任务、模型融合、Reranker 分层蒸馏 | 未实现 | 面向千万级语料和多卡训练，超出单知识库微调的范围 |

## 1. 先准备可审核的标注

使用现有黄金集 JSONL 格式，优先标注**实际能回答问题的 child chunk ID**：

```json
{"question":"年假可以留到明年吗","relevant_chunk_ids":["替换为当前索引中的真实child ID"],"category":"年假"}
```

`relevant_doc_ids` 或 `relevant_sources` 可以限定关键词标注的来源，`answer_keywords` 用于定位包含答案的 child。只有文档 / 来源标签的记录会跳过：一个员工手册里还有大量不相关内容，不能把整篇文档的所有切片都当作正样本。parent ID 的范围也较宽；仅 parent 标注筛选出来的 child 即使教师分数达标，也仍需要人工审核。parent 加答案关键词可缩小 child 范围，相关 parent 下的兄弟切片不会被挖成负例。

把独立测试问题保存为 `test.jsonl`。先识别同义问法、同一原文派生的问题，将测试查询族与训练标注分开。程序只保护 Unicode、大小写、标点和空白归一化后的同一问题，不能自动识别所有同义改写和来源泄漏。

## 2. 挖掘并检查数据

```bash
rag mine-negatives annotations.jsonl \
  --held-out test.jsonl \
  --config config.yaml \
  --output data/training/run1
```

输出目录必须尚不存在。默认使用当前生产索引的 dense 与 BM25 child 召回，融合后收集候选；不执行生成、查询改写或 Small-to-Big。挖掘排除已知正样本、同 parent / 重叠片段，默认也排除正样本所在文档，再限制同一文档与 parent 的负例数量。语义相似的候选仍可能是漏标的正确答案，因此检查 `mining-report.json` 中的原文、分数、来源、排名和过滤统计。

教师必须对 query-positive 与 query-negative 使用同一语义分数尺度。RRF 排名分、词法重排分与 embedding 余弦分不可混作同一尺度。默认教师过滤采用：

```text
negative_score <= min(
  max_negative_score,
  positive_score - absolute_margin,
  positive_score - abs(positive_score) * relative_margin
)
```

多个正样本以导出正样本的最低教师分为参考。默认 `absolute_margin=0.1`、`relative_margin=0.05`、`min_positive_score=0.5`、`max_negative_score=0.8`；两个 margin 同时开启时执行较严格的门槛。它们是实验起点，需根据教师模型的分数分布与业务标注调节，不能解释成“95% 确定不相关”。改变 scorer 或分数归一化方式后要重新检查门槛。

可用 `--mining-config mining.json` 覆盖配置，例如：

```json
{"candidate_pool":200,"rank_min":1,"rank_max":200,"num_negatives":4,"relative_margin":0.05,"absolute_margin":0.0,"dev_fraction":0.2,"seed":42}
```

负例不足会保留已有有效负例并记录 `shortage`；不会从被排除的片段中补齐。没有安全负例的查询跳过。

`min_negative_score` 是简单负例的下限：教师分低于它的候选不进训练集，每组滤掉的数量记在 `easy_filtered`。默认 0 表示不过滤，此时同一问题的难候选不够时，排在前 `rank_max` 名里的低分候选仍会被选作负例。合适的下限取决于教师模型的分数分布，先看一轮报告里的 `neg_scores` 再定；提高它会让更多问题出现 `shortage` 或被跳过。

教师分高于上面门槛的候选不会当负例，分数最高的 5 条写进该问题的 `suspected_positives`（块 ID、分数、排名、原文），全部数量记在 `suspected_positive_count`。这些候选要么是漏标的正确答案，要么只是和正例很像。逐条看原文：确实能回答问题的，把块 ID 补进标注的 `relevant_chunk_ids` 后重新挖掘。被整组跳过的问题（`no_safe_negatives`）同样保留这份名单。

教师缺席、未就绪或降级为词法 / 排名评分时，候选标记为待审，默认不会写进训练集。离线可以用 `--lite --no-teacher` 检查流程；哈希向量的排名不证明语义难度。只有逐条人工确认后才使用 `--allow-unverified` 导出未通过教师过滤的候选，该选项本身不会审核数据。Web 评测中心的难负例区域支持检查来源与原文，并逐条选择批准的行导出。

### 大模型改写正例生成负例

语料里找不到够难的负例时（教师分普遍很低），可以让文本模型把正例改写成“似是而非”的段落。`--synthetic 2` 或 `{"synthetic_negatives":2}` 开启，每个问题调用一次辅助文本模型，最多 4 段：

```bash
rag mine-negatives annotations.jsonl --held-out test.jsonl --synthetic 2 --output data/training/run1
```

- 改写对象是教师分最高的那条正例。提示词要求写“相邻的另一件事”（换对象、适用人群、业务环节），不要只改数字或日期：同一事实的错误版本对检索模型来说和正例一样相关，当负例只会教它乱分。标注里有 `answer_keywords` 时，提示词会要求模型避开这些词。
- 生成结果依次过滤：包含标注里的答案关键词、与已有正负例近似重复、教师分高于负例门槛（仍像在回答问题）、教师分低于 `min_negative_score`。各原因的数量记在 `synthetic_rejected`。教师与挖掘用的是同一个、同一尺度。
- 通过过滤的写进该问题的 `synthetic_neg`（训练文本、正文、教师分、改写自哪个块），**不进入** `neg`。训练文本沿用正例的来源面包屑但去掉最末一级，不带正例的关键词和假设性问题。最末一级标题说的是正例自己的主题（FAQ 块里就是问题本身），套在改写文本上会让教师把它判成相关。
- 它们不是知识库原文，教师过滤也不能保证不回答问题。逐条看过之后，命令行用 `--include-synthetic` 把报告里全部改写负例并入训练文件；这个选项本身不做审核，要剔除个别条目请在 Web 评测中心逐条勾选后导出。只有改写负例、没有挖到语料负例的问题，在此之前不会出现在训练文件里。
- 开启后，问题和正例原文会发送给配置的文本模型。没有文本模型、调用失败或输出不是 JSON 数组时，这个问题不产生改写负例，其余流程照常。

在“企业知识库（示例）”上用 `deepseek-flash` 和 `bge-reranker-v2-m3` 实跑过一次（12 个问题，每个问题 3 段，约 41 秒）：36 段里 28 段通过过滤，8 段被拒（6 段教师分贴近正例，2 段近似重复）。通过的 28 段教师分中位数 0.04，6 段在 0.1 以上；同一批问题挖到的 48 条语料负例中位数 0.002。12 个问题里有 8 个，最难的改写负例比最难的语料负例分更高。通过过滤的里面仍有该剔除的，例如把同一张运费表改了几格的版本、讲同一个漏洞但没给修复办法的段落，所以人工筛选不能省。这只是一个 6 篇文档的示例库上的一次结果，不代表其他语料和模型。

文件如下：

| 文件 | 用途 |
| --- | --- |
| `train.jsonl` / `dev.jsonl` | embedding 数据：`query`、`pos: list[str]`、`neg: list[str]`；通过教师过滤的行另有与之对齐的 `pos_scores`、`neg_scores` |
| `reranker.train.jsonl` / `reranker.dev.jsonl` | reranker 数据：`query`、`passage`、`label: 0 或 1` |
| `mining-report.json` | 参数、模型 / 索引指纹、过滤原因、待审记录、疑似漏标正例、改写负例、样本来源与不足数量 |

train/dev 按归一化 query 与 seed 的稳定哈希划分，追加其他 query 不会把已有 query 换组。小数据集或过滤后可能出现空 train/dev 文件，训练脚本会拒绝；此时补充独立标注的查询组，或在保持查询族 / 原文分组的前提下重新划分，不能复制训练行填充 dev。`--held-out` 的测试问题不会用于挖掘训练监督。

自动划分不会绑定同一原文的不同问题或同义查询族。对有此类关联的数据，审核后需调整导出的 train/dev 分组，或使用独立标注的 dev；同步调整 embedding 与 reranker 文件。若要测未见原文的泛化能力，还要单独按原文划分测试语料，当前固定生产 corpus 的挖掘不会自动完成这一实验。

## 3. 无模型、无网络预检查

```bash
python -m prism.training.finetune \
  --task embedding \
  --train data/training/run1/train.jsonl \
  --dev data/training/run1/dev.jsonl \
  --model BAAI/bge-m3 \
  --output data/models/embedding-run1 \
  --dry-run
```

`--dry-run` 只使用标准库，打印 JSON 计划、样本数与数据 SHA256，不导入 torch / sentence-transformers，不下载模型，也不创建输出目录。它检查非空文本、正负冲突、数值参数、train/dev 的归一化 query 重叠与输出目录占用。reranker 的 train 与 dev 各自必须包含 0 和 1；同 query/passage 不能有冲突标签。

## 4. 真实微调

安装训练依赖后，选择能由 Sentence Transformers 加载的模型 ID，或**已有本地模型目录**：

```bash
uv pip install -e '.[finetune]'

python -m prism.training.finetune \
  --task embedding \
  --train data/training/run1/train.jsonl \
  --dev data/training/run1/dev.jsonl \
  --model /absolute/path/to/local/embedding-model \
  --output data/models/embedding-run1 \
  --device cpu --local-files-only \
  --epochs 1 --batch-size 8 --learning-rate 2e-5 --max-length 512

python -m prism.training.finetune \
  --task reranker \
  --train data/training/run1/reranker.train.jsonl \
  --dev data/training/run1/reranker.dev.jsonl \
  --model /absolute/path/to/local/reranker-model \
  --output data/models/reranker-run1 \
  --device cpu --local-files-only \
  --epochs 1 --batch-size 8 --learning-rate 2e-5 --max-length 512
```

去掉 `--device cpu` 时由训练库选择设备；去掉 `--local-files-only` 并使用模型 ID 时可能下载权重。默认不启用远程代码、混合精度或上传模型。每次运行使用新的或完全空的输出目录；不覆盖、不自动恢复旧训练。`--max-steps 1` 可限制一次权重更新，用于本地运行检查。

embedding 使用 `SentenceTransformerTrainer`、`MultipleNegativesRankingLoss` 和官方 `NO_DUPLICATES` 的无重复批次规则。每条记录的所有 positive × negative 组合转成三列 triplet，负例数量不同无需补齐或丢弃。重复 triplet 去重，元数据不会误入模型输入。runner 缓存完整的无重复 batch 计划并提供准确的 batch 数量，避免官方 sampler 的估算长度使重复 query 的后续 triplet 未被访问；每个 epoch 打乱 batch 及 batch 内顺序，batch 组成保持稳定。显式 `--max-steps` 会按指定步数提前停止。默认不使用 warmup，避免小数据集仅有的更新步学习率为零。

相同 query 在同 batch 不能重复，但不同 query 的正样本仍可能回答彼此；批次内语义假负例需要数据审核，`NO_DUPLICATES` 只能防重复文本。可选教师 `pos_scores` / `neg_scores` 会检查长度与有限值，默认只做对比学习，**不使用这些分数**。

加 `--distill-weight 0.5` 开启 label 层蒸馏（仅 embedding）。损失变为：

```text
loss = MultipleNegativesRankingLoss + distill_weight × KL(教师分布 ‖ 学生分布)
教师分布 = softmax([logit(pos_score), logit(neg_score)])
学生分布 = softmax(20 × [cos(query, pos), cos(query, neg)])
```

对比项把每个难负例当成完全不相关；蒸馏项让模型对“正例只比负例高一点”的样本对不要拉得太开，减轻漏网假负例的影响。开启后 train 与 dev 的每条记录都必须带 0–1 的 `pos_scores` / `neg_scores`，否则训练前就报错；分数来自 `rag mine-negatives` 的教师，即当前重排模型，本地 Cross-Encoder 的分数是 sigmoid 输出，取 logit 即还原为原始打分。Rerank API 的分数未必是 sigmoid，这时教师分布的软硬程度会偏。无教师、经 `--allow-unverified` 导出的行没有分数，不能用于蒸馏。0.5 是起点，不是调好的值；蒸馏会把教师的错误一并学进来，是否有收益仍以第 5 节的独立评测为准。reranker 微调不做蒸馏：教师就是它自己。

模型需要 query instruction 时，显式传 `--query-instruction '所需前缀'`；生产的 `embedding.query_instruction` 必须使用相同前缀。训练输入应与部署的正文构造、标题路径、截断长度与 tokenizer 对齐，尤其注意长文在 `--max-length` 处截断。

reranker 使用 `CrossEncoderTrainer`、单输出 `num_labels=1` 与 `BinaryCrossEntropyLoss`。它只消费两列文本和二元 label，不会把 embedding 数据的 list 列当文本编码。训练保存：

| 路径 | 内容 |
| --- | --- |
| `<output>/model/` | 按 dev loss 选择的可加载模型与 tokenizer |
| `<output>/checkpoints/` | 训练检查点与 trainer 状态 |
| `<output>/training_manifest.json` | 输入指纹、参数、状态、依赖版本与结果；失败时保留失败状态 |
| `<output>/metrics.json` | 训练与 dev loss / 性能诊断 |

这是普通稠密 embedding / CrossEncoder 微调，不会训练 BGE-M3 的稀疏或多向量检索头。大模型的真实训练仍需要对应内存与算力。

## 5. 在独立检索评测中验收并激活

dev loss 在生成的候选集上衡量训练，不代表全库的检索收益。保留独立业务 `test.jsonl`，固定语料、切片、检索策略与 qrels，对基座与新模型分别运行现有 `rag eval test.jsonl --config 配置.yaml --output 报告.md`；比较 Hit / MRR / NDCG 及错误归因。需要 Recall@K 时，可单独使用 Sentence Transformers `InformationRetrievalEvaluator`，输入全部固定 corpus 和每个 query 的全部 relevant IDs，不能仅在人工挑好的负例列表上评测召回。

本地 `model/` 目录可在模型配置里使用。激活新的 embedding 后，**重建所有相关知识库的向量索引**，保证 query 与文档使用同一 checkpoint；维度相同也不能沿用旧向量。reranker 更换通常无需重建向量索引，但须重新验证分数分布、拒答阈值 `retrieval.rerank.min_score` 与真实首阶段候选覆盖。sigmoid 输出在 0–1 之间仍不等于校准过的相关概率。

权重更新与离线测试能证明训练接口可执行（包括蒸馏损失）；本次没有业务数据上的真实微调收益报告。先审核领域数据，再用独立测试集决定是否激活。

实现检查：`pytest tests/test_finetune.py` 不加载模型；`RAG_TEST_FINETUNE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest tests/test_finetune.py -k offline_cpu_training` 在本机随机创建极小 BERT，分别检查 embedding、带蒸馏的 embedding、reranker 的一次完整 epoch 与显式一步 CPU 训练，验证权重变化、保存与重新加载，不下载模型。该测试衡量训练链路可执行性，不衡量领域检索效果。

## 6. 多轮挖掘：用上一轮模型重新找难负例

Conan-Embedding 在训练过程中定期重算负例分数，把已经变“简单”的负例换成新的难负例。本仓库的训练脚本不做训练内替换，可以按轮次做同样的事：

1. 第一轮按第 2–5 节挖掘、训练、验收，得到 `data/models/embedding-run1/model`。
2. 复制一份配置，把 `embedding.model` 指向这个目录，`paths.index_dir` 指向一个新目录，用 `rag ingest <语料> --config 新配置.yaml` 重建索引。
3. 用同一份训练标注执行 `rag mine-negatives annotations.jsonl --held-out test.jsonl --config 新配置.yaml --output data/training/run2`。此时 dense 召回来自第一轮模型，排在前面的是它仍然分不开的片段。
4. 从**原始基座**出发，用 run2 的数据（或 run1 与 run2 合并去重后的数据）重新训练，再回到第 5 节验收。

每一轮都保持 `--held-out` 不变，测试问题始终不参与挖掘。教师仍是同一个重排模型，分数门槛不用改。这个流程是按现有命令拼出来的，本次没有在业务数据上跑过完整两轮，轮数与是否合并数据都要靠独立评测决定。

## 方法与接口来源

- [Conan-Embedding 原论文](https://arxiv.org/abs/2408.15710)：训练中动态替换难负例与跨 GPU 负样本，第 6 节的多轮流程是它的离线近似。

- [NV-Retriever 原论文](https://arxiv.org/abs/2407.15831)：以正样本分数为参考过滤潜在假负例；其 95% 方案是实验配置。
- [Sentence Transformers 难负例 API](https://sbert.net/docs/package_reference/util/hard_negatives.html)：`relative_margin`、候选范围与训练输出格式。
- [Sentence Transformers embedding 训练文档](https://sbert.net/docs/sentence_transformer/training_overview.html)：按 loss 选择输入格式，输入列顺序重要，剔除元数据。
- [Sentence Transformers losses API](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)：MNRL、NO_DUPLICATES 与可扩展的 GIST 假负例过滤。
- [CrossEncoder 训练文档](https://sbert.net/docs/cross_encoder/training_overview.html)：单输出 reranker 与二元训练。
- [FlagEmbedding embedding 官方示例](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/embedder/README.md)：`query` / `pos` / `neg` JSONL，以及蒸馏分数格式。
- [InformationRetrievalEvaluator](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html)：固定 corpus 上的 Recall / MRR / NDCG。
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)：原文 / 查询族分组划分的可用接口。
