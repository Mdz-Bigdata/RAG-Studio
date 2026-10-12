#!/usr/bin/env python3
"""批量用 MinerU 4.x 解析目录下的 PDF / Office / 图片，落盘 markdown + content_list + middle_json。

用法：
    python scripts/mineru_batch_parse.py data/raw --tier flash --out data/mineru_out
    python scripts/mineru_batch_parse.py report.pdf --tier standard --api-url http://127.0.0.1:8000

tier 选择：数字文档（Word 导出的 PDF）用 flash 最快；扫描件 / 复杂表格用 basic（CPU）或 standard（VLM）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from prism.config import MinerUConfig
from prism.ingest.parsers.mineru_parser import MINERU_EXTENSIONS, MinerUParser, mineru_available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--tier", default="flash", choices=["flash", "basic", "standard", "advanced"])
    ap.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    ap.add_argument("--out", default="data/mineru_out", type=Path)
    ap.add_argument("--api-url", default=None, help="连接 mineru-kit api-server")
    ap.add_argument("--page-range", default="")
    args = ap.parse_args()

    if not mineru_available() and not args.api_url:
        print("未安装 MinerU：uv pip install -U 'mineru[torch]>=4.0,<5' && mineru-kit models download --tier standard", file=sys.stderr)
        return 1

    cfg = MinerUConfig(tier=args.tier, ocr_mode=args.ocr_mode, api_url=args.api_url, page_range=args.page_range, save_outputs=True)
    parser = MinerUParser(cfg, out_dir=args.out)
    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*") if x.suffix.lower().lstrip(".") in MINERU_EXTENSIONS))
        else:
            files.append(p)
    for f in files:
        t0 = time.perf_counter()
        try:
            doc = parser.parse(f)
            print(f"OK   {f}  blocks={len(doc.blocks)}  {time.perf_counter() - t0:.1f}s  -> {args.out / doc.doc_id}")
        except Exception as exc:
            print(f"FAIL {f}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
