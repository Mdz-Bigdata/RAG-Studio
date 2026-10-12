"""进程内 torch 在 Apple GPU（MPS）上的计算串行化。

PyTorch 的 MPS 后端不是线程安全的：进程里所有 torch 计算共用一条 MPS stream 和一批内部缓存，两个线程同时在 MPS 上算，
要么 Metal 断言失败、整个进程 abort（"A command encoder is already encoding to this command buffer"），要么内部的
kernel 缓存哈希表被并发改坏、线程在满 CPU 上空转卡死——两种都在本机复现过，前者会让 uvicorn 的 worker 直接退出。
Web 服务里请求线程池、后台重建、模型预热、MinerU 解析都会跑本地模型，所以凡是在 MPS 上搬权重、做前向的地方
（向量模型、重排模型、CLIP、MinerU 本地解析）都持有同一把锁。CUDA / CPU 可以多线程并发，不加锁。
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any, ContextManager

MPS_LOCK = threading.RLock()   # 可重入：持锁的代码里再调用别的本地模型不会自锁


def on_mps(device: Any) -> bool:
    """device 可以是 "mps" / "mps:0" / torch.device。"""
    return str(device or "").split(":")[0] == "mps"


def mps_guard(device: Any) -> ContextManager[Any]:
    """在 MPS 上时返回全进程共享的锁，否则返回空上下文。"""
    return MPS_LOCK if on_mps(device) else nullcontext()


def auto_device() -> str:
    """与 sentence-transformers / MinerU 的默认选择一致：cuda > mps > cpu。"""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
