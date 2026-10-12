import json
import re
import requests
from sqlalchemy.orm import Session
from app.config import settings
import importlib
_llm_mod = importlib.import_module("app.service.llm-service")
llm_service = _llm_mod.LLMService()

class RerankService:
    """
    重排序服务。支持大模型批量评分精排、外部 API (Jina/Cohere) 精排以及本地确定性词频 Mock 精排。
    """
    
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5, db: Session = None) -> list[dict]:
        """
        对候选文档块进行重排序，返回最相关的 top_k 个块。
        """
        if not candidates:
            return []
            
        # 如果候选项只有一个，直接返回
        if len(candidates) == 1:
            return candidates
            
        # 探测当前是否有活跃的 LLM 客户端可以用来做重排打分
        client, _ = llm_service._get_client_and_model(db)
        
        if settings.USE_LLM_RERANK and client:
            try:
                return self._rerank_with_llm(query, candidates, top_k, db)
            except Exception as e:
                print(f"LLM Rerank failed: {e}. Falling back to default scoring.")
                return self._rerank_with_mock(query, candidates, top_k)
        elif settings.RERANK_API_KEY:
            try:
                return self._rerank_with_api(query, candidates, top_k)
            except Exception as e:
                print(f"API Rerank failed: {e}. Falling back to default scoring.")
                return self._rerank_with_mock(query, candidates, top_k)
        else:
            return self._rerank_with_mock(query, candidates, top_k)

    def _rerank_with_llm(self, query: str, candidates: list[dict], top_k: int, db: Session = None) -> list[dict]:
        """
        使用大模型进行一脉相承的批量 Rerank 打分
        """
        # 构建大模型精排提示词
        doc_entries = []
        for idx, c in enumerate(candidates):
            doc_entries.append(f"【ID: {c['id']}】\n{c['content']}\n---")
            
        docs_text = "\n".join(doc_entries)
        
        prompt = (
            "作为专业的搜索精排助手，请根据给出的用户问题，评估以下各文档片段的相关度评分（分值范围 0-100）。\n"
            "分值意义：90-100（精准解答问题），70-89（高度相关，提供核心背景），40-69（部分相关，有边角关联），0-39（不相关）。\n"
            "【输出格式】：你必须输出一个纯 JSON 数组，不包含任何 markdown 代码块或解释性文字。格式如下：\n"
            '[{"id": "xxxxx", "score": 95}, {"id": "yyyyy", "score": 75}]\n\n'
            f"【用户问题】: {query}\n\n"
            f"【待评估文档】:\n{docs_text}\n"
            "相关度打分结果 JSON:"
        )
        
        try:
            client, model_name = llm_service._get_client_and_model(db)
            if not client:
                raise ValueError("No active LLM client available for reranking")
                
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional search ranker. Only output valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"} if "gpt-4" in model_name else None
            )
            
            content = response.choices[0].message.content.strip()
            # 兼容 Markdown 围栏
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
                
            scores_list = json.loads(content)
            if isinstance(scores_list, dict) and "results" in scores_list:
                scores_list = scores_list["results"]
            
            # 将打分回填给 candidates
            score_map = {item["id"]: float(item["score"]) for item in scores_list if "id" in item and "score" in item}
            
            for c in candidates:
                # 获取 LLM 分数，若不存在则用原 RRF 得分降级折算
                c["rerankScore"] = score_map.get(c["id"], c["score"] * 100)
                
            # 排序
            sorted_candidates = sorted(candidates, key=lambda x: x["rerankScore"], reverse=True)
            return sorted_candidates[:top_k]
        except Exception as e:
            print(f"Error parsing LLM Rerank JSON: {e}")
            raise e

    def _rerank_with_api(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        调用外部 Rerank API (如 Jina / Cohere)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.RERANK_API_KEY}"
        }
        
        payload = {
            "model": settings.RERANK_MODEL,
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k
        }
        
        response = requests.post(settings.RERANK_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 解析返回得分并对齐
        results = data.get("results", [])
        reranked = []
        
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            candidate = dict(candidates[idx])
            candidate["rerankScore"] = score * 100
            reranked.append(candidate)
            
        return reranked

    def _rerank_with_mock(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        本地确定性精排算法。
        结合 TF-IDF 简化思想：对 Query 中的关键词在候选 Chunk 中进行词频和位置加权，微调原本的 RRF 排序。
        确保在 Mock 状态下同样具备合理的重排序效果。
        """
        # 提取 Query 中的关键词
        words = re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", query.lower())
        
        for c in candidates:
            content_lower = c["content"].lower()
            term_score = 0.0
            
            for word in words:
                if word in content_lower:
                    # 匹配次数加分
                    freq = content_lower.count(word)
                    term_score += 10.0 + min(freq * 2.0, 15.0)
                    
                    # 实体词出现在头部/前部加分
                    first_idx = content_lower.find(word)
                    if first_idx < 100:
                        term_score += 5.0
            
            # 融合 RRF 得分做最终精排分
            c["rerankScore"] = (c["score"] * 30.0) + term_score
            
        sorted_candidates = sorted(candidates, key=lambda x: x["rerankScore"], reverse=True)
        return sorted_candidates[:top_k]
