"""多模态 RAG：图文（视频）统一嵌入 + 图片描述索引 → 文搜图 / 图搜图 / 图文混合 / 图搜文 / 文搜视频 / 图搜视频 / 查重 → 带图片引用的回答。"""

from .describe import ImageDescriber, OCREngine, available_ocr
from .embedders import APIImageEmbedder, BaseImageEmbedder, CLIPEmbedder, PerceptualImageEmbedder, build_image_embedder, build_wemm, clip_available, image_to_base64, is_wemm
from .extract import IMAGE_EXTENSIONS, ExtractedImage, decode_image_b64, extract_images, extract_markdown_images
from .image_index import ImageIndex
from .retriever import MultimodalRetriever, wants_images
from .schemas import ImageHit, ImageRecord, MultimodalResult, RouteHit
from .wemm import VideoClip, WeMMAPIEmbedder, WeMMEmbedder, wemm_available

__all__ = [
    "APIImageEmbedder", "BaseImageEmbedder", "CLIPEmbedder", "ExtractedImage", "IMAGE_EXTENSIONS", "ImageDescriber", "ImageHit", "ImageIndex",
    "ImageRecord", "MultimodalResult", "MultimodalRetriever", "OCREngine", "PerceptualImageEmbedder", "RouteHit", "VideoClip", "WeMMAPIEmbedder", "WeMMEmbedder",
    "available_ocr", "build_image_embedder", "build_wemm", "clip_available", "is_wemm", "wemm_available", "decode_image_b64", "extract_images", "extract_markdown_images", "image_to_base64", "wants_images",
]
