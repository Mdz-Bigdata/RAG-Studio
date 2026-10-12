"""多智能体协作（LanceDB 黑板）：规划 → 并行研究 → 分析 → 审校（证据不足改写重查）→ 撰稿。"""

from .board import Blackboard
from .team import ROLES, AgentTeam

__all__ = ["ROLES", "AgentTeam", "Blackboard"]
