"""多模态 RAG：图文统一嵌入 + 图片描述索引 → 文搜图 / 图搜图 / 图文混合 / 图搜文 → 带图片引用的回答。"""

from .describe import ImageDescriber, OCREngine, available_ocr
from .embedders import APIImageEmbedder, BaseImageEmbedder, CLIPEmbedder, PerceptualImageEmbedder, build_image_embedder, clip_available, image_to_base64
from .extract import IMAGE_EXTENSIONS, ExtractedImage, decode_image_b64, extract_images, extract_markdown_images
from .image_index import ImageIndex
from .retriever import MultimodalRetriever, wants_images
from .schemas import ImageHit, ImageRecord, MultimodalResult, RouteHit

__all__ = [
    "APIImageEmbedder", "BaseImageEmbedder", "CLIPEmbedder", "ExtractedImage", "IMAGE_EXTENSIONS", "ImageDescriber", "ImageHit", "ImageIndex",
    "ImageRecord", "MultimodalResult", "MultimodalRetriever", "OCREngine", "PerceptualImageEmbedder", "RouteHit", "available_ocr",
    "build_image_embedder", "clip_available", "decode_image_b64", "extract_images", "extract_markdown_images", "image_to_base64", "wants_images",
]
