"""个性化推荐（LanceDB）：知识条目 + 用户交互 → 兴趣向量召回 → 热度加权 → MMR 多样化，带推荐理由。"""

from .engine import EVENT_WEIGHTS, AlreadyVoted, LanceRecommender, RecItem

__all__ = ["EVENT_WEIGHTS", "AlreadyVoted", "LanceRecommender", "RecItem"]
