import time
from typing import Generator
from sqlalchemy.orm import Session
from openai import OpenAI
from app.config import settings

class LLMService:
    """
    大语言模型交互服务。支持从数据库动态读取活跃的文本模型配置，并缓存客户端实例。
    """
    def __init__(self):
        self.clients_cache = {}  # 缓存 config_id -> client

    def _get_client_and_model(self, db: Session = None, category: str = "text") -> tuple[OpenAI | None, str]:
        """
        动态获取大模型客户端和模型名称。
        优先从 SQLite 读取 active 状态的模型配置。
        """
        if db:
            try:
                from app.repository import SQLiteRepository
                repo = SQLiteRepository(db)
                active_config = repo.get_active_model_config(category)
                
                # 如果是多模态，但未激活该分类配置，自动回退到 text 分类活跃配置
                if not active_config and category == "multimodal":
                    active_config = repo.get_active_model_config("text")
                    
                if active_config:
                    cid = active_config.id
                    if cid not in self.clients_cache:
                        if active_config.apiKey:
                            self.clients_cache[cid] = OpenAI(
                                api_key=active_config.apiKey,
                                base_url=active_config.baseUrl
                            )
                        else:
                            self.clients_cache[cid] = None
                    return self.clients_cache[cid], active_config.modelName
            except Exception as e:
                print(f"Error querying active {category} config: {e}")

        # 降级到环境变量配置
        if settings.OPENAI_API_KEY:
            if "default" not in self.clients_cache:
                self.clients_cache["default"] = OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_API_BASE
                )
            return self.clients_cache["default"], settings.LLM_MODEL

        return None, settings.LLM_MODEL

    def generate_hyde_document(self, query: str, db: Session = None) -> str:
        """
        生成假设文档 (HyDE)。
        """
        prompt = (
            "请针对用户提出的问题，写一个学术性、专业且简明扼要的回答。该回答将作为搜索系统的语义匹配媒介。\n"
            "即使你无法给出完全精准的回答，也请根据常识写一段推测性描述，保证段落涵盖关键术语。\n"
            f"问题: {query}\n"
            "假想回答 (要求 150-300 字):"
        )
        
        client, model_name = self._get_client_and_model(db)
        
        if client:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful RAG retrieval helper."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM HyDE generation failed: {e}. Falling back to default search query.")
                return query
        else:
            return f"关于“{query}”，这通常涉及到企业级系统的核心架构设计。实现上，我们需要引入数据层解析（如 MinerU 处理 PDF）、向量数据库（如 Qdrant 存储密向量）以及基于 RRF 融合的多路召回策略。此外，为了丰富背景信息，可以借助 UnWeaver 织布机实现实体间的上下级与词义定义编织召回。"

    def generate_answer_stream(
        self, 
        query: str, 
        context: str, 
        db: Session = None,
        image_b64: str | None = None,
        temp_doc_content: str | None = None
    ) -> Generator[str, None, None]:
        """
        基于检索上下文流式生成最终的问答回复 (支持多模态图片与临时上传文档)
        """
        # 如果上传了临时待分析文档，将其编织进上下文头部
        if temp_doc_content:
            context = f"【单次会话上传临时文件（不属于全局知识库，仅供本次即时问答分析）】:\n{temp_doc_content}\n\n" + context

        # 如果有图片，优先寻找多模态模型配置
        category = "multimodal" if image_b64 else "text"
        client, model_name = self._get_client_and_model(db, category)
        
        system_prompt = (
            "你是一个名为 Antigravity RAG 的企业级智能文档助手。\n"
            "请结合以下检索到的参考内容（Context），用客观专业的口吻回答用户提问（Query）。\n"
            "如果参考内容中存在相互矛盾或未涉及的信息，请优先相信参考内容并向用户说明其局限性，不要胡乱编造。\n\n"
            "参考内容 (Context):\n"
            f"\"\"\"\n{context}\n\"\"\"\n"
        )
        
        if client:
            try:
                if image_b64:
                    # 整理 Base64 格式头
                    if not image_b64.startswith("data:"):
                        image_url = f"data:image/png;base64,{image_b64}"
                    else:
                        image_url = image_b64
                        
                    user_content = [
                        {"type": "text", "text": f"问题 (Query): {query}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"问题 (Query): {query}"}
                    ]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    stream=True
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            except Exception as e:
                err_msg = f"\n[API 异常]: 大模型调用失败: {e}。已切换为系统内置回复。\n"
                for char in err_msg:
                    yield char
                    time.sleep(0.02)
                yield from self._generate_mock_answer_stream(query, context)
        else:
            yield from self._generate_mock_answer_stream(query, context)

    def _generate_mock_answer_stream(self, query: str, context: str) -> Generator[str, None, None]:
        """
        本地确定性 Mock 流式回答生成器。
        """
        mock_response = (
            f"### 🤖 离线体验模式已启动\n\n"
            f"您好！由于系统目前未配置任何激活的大模型配置，当前回答为**本地离线检索分析报告**：\n\n"
            f"1. **检索问答**：您提问了 “*{query}*”。\n"
            f"2. **知识召回**：我们已从当前知识库中为您精准检索并融合了 Dense 与 BM25 多路数据，共得到几组高度相关的文档片段（已呈现在侧边栏“引用来源”中）。\n"
            f"3. **关系编织 (UnWeaver)**：根据编织算法，我们提取并扩充了检索块中的关键实体定义，防止了语义碎裂带来的上下文缺失。\n\n"
            f"**以下是融合了召回上下文后生成的分析段落**：\n\n"
            f"> \"本系统基于 MinerU 与 pdfplumber 完成高质量 Markdown 转换，通过 Qdrant 进行向量召回，"
            f"使用 rank_bm25 对名词短语执行倒排匹配。在召回阶段借助 HyDE（假设文档）机制扩大检索面，"
            f"最终经过 LLM 进行重排序与 RRF (Reciprocal Rank Fusion) 倒数排名融合，保证首屏精度达到企业级标准。\"\n\n"
            f"*友情提示：如需获取基于大模型的真实问答和知识总结，请在“模型配置中心”配置并激活您的 API Key 和模型选型。*"
        )
        
        for chunk in [mock_response[i:i+4] for i in range(0, len(mock_response), 4)]:
            yield chunk
            time.sleep(0.01)
