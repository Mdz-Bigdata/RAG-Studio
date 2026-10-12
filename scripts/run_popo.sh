#!/usr/bin/env bash
# 在 MinerU-Popo 仓库中跑推理 + 建树，输入来自 `rag popo-export`。
#
# 用法：
#   POPO_REPO=/path/to/MinerU-Popo POPO_DIR=data/popo bash scripts/run_popo.sh
#
# 前置（在 Popo 仓库内，需 GPU）：
#   conda create -n popo python=3.10 && conda activate popo && pip install -r requirements.txt
#   hf download DreamEternal/MinerU-Popo --local-dir models/MinerU-Popo
set -euo pipefail

POPO_REPO="${POPO_REPO:?请设置 POPO_REPO=MinerU-Popo 仓库路径}"
POPO_DIR="$(cd "${POPO_DIR:-data/popo}" && pwd)"
MODEL_PATH="${POPO_MODEL_PATH:-${POPO_REPO}/models/MinerU-Popo}"
BACKEND="${POPO_INFERENCE_BACKEND:-transformers}"   # transformers | vllm（vllm 需在 post_processing/model_utils.py 配置 url/key）

NORM_DIR="${POPO_DIR}/outputs/label_normalization/mineru"
INFER_DIR="${POPO_DIR}/outputs/inference/mineru"
TREE_DIR="${POPO_DIR}/outputs/build_tree/mineru"
TREE_TXT_DIR="${POPO_DIR}/outputs/build_tree_txt/mineru"
mkdir -p "${INFER_DIR}" "${TREE_DIR}" "${TREE_TXT_DIR}"

echo "[popo] inference: ${NORM_DIR} -> ${INFER_DIR}"
cd "${POPO_REPO}/post_processing"
POPO_INFERENCE_BACKEND="${BACKEND}" POPO_MAX_NEW_TOKENS="${POPO_MAX_NEW_TOKENS:-8192}" \
python run_inference.py \
  --input-dir "${NORM_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${INFER_DIR}" \
  --raw-output-root "" \
  --limit 0 --resume

echo "[popo] build tree: ${INFER_DIR} -> ${TREE_DIR}"
python get_json_tree.py --input-dir "${INFER_DIR}" --output-dir "${TREE_DIR}" --txt-dir "${TREE_TXT_DIR}"

echo "[popo] done. 现在把 config.yaml 的 parsing.popo.enabled 设为 true，重新执行 rag ingest 即使用文档树。"
