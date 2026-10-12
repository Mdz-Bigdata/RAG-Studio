import { useState } from 'react';

export interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

export interface Source {
  id: string;
  fileName: string;
  content: string;
  score: number;
  isWeaved: boolean;
  weavedEntity: string;
}

export interface Relation {
  entity: string;
  definition: string;
  sourceChunkId: string;
  weavedFrom: string;
}

export interface PlaygroundData {
  denseResults: any[];
  bm25Results: any[];
  hydeResults: any[];
  rrfResults: any[];
  rerankedResults: any[];
  weavedRelations: Relation[];
  hydeDocument: string;
}

const API_BASE = 'http://localhost:8000/api/v1';

export function useChat() {
  const [messages, setMessages] = useState<ChatHistoryItem[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [relations, setRelations] = useState<Relation[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [playgroundData, setPlaygroundData] = useState<PlaygroundData | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  // 检索游乐场查询调试
  const searchPlayground = async (kbId: string, query: string, topK: number = 5) => {
    setIsSearching(true);
    try {
      const res = await fetch(`${API_BASE}/chat/${kbId}/playground`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, topK }),
      });
      if (res.ok) {
        const data = await res.json();
        setPlaygroundData(data);
        return data as PlaygroundData;
      }
    } catch (e) {
      console.error('Failed to query playground', e);
    } finally {
      setIsSearching(false);
    }
    return null;
  };

  // 发送智能对话提问 (流式返回)
  const sendChatMessage = async (kbId: string, messageText: string, imageB64?: string, tempDocContent?: string) => {
    if (!messageText.trim() || isSending) return;

    setIsSending(true);
    setSources([]);
    setRelations([]);
    
    const newUserMessage: ChatHistoryItem = { role: 'user', content: messageText };
    const updatedMessages = [...messages, newUserMessage];
    setMessages(updatedMessages);

    // 占位准备流式写入 assistant 回复
    const assistantMessageIndex = updatedMessages.length;
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    try {
      // 提取对话历史 (排除占位行)
      const chatHistory = messages.map(m => ({ role: m.role, content: m.content }));

      const res = await fetch(`${API_BASE}/chat/${kbId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          history: chatHistory,
          imageB64: imageB64 || null,
          tempDocContent: tempDocContent || null
        }),
      });

      if (!res.body) {
        throw new Error('No response body for streaming');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';
      
      // 缓存数据流
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // 按照 SSE 规范的行来切分数据
        const lines = buffer.split('\n\n');
        // 保存最后一个未完结的行回到缓存
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          // 匹配 event 和 data 字段
          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (eventMatch && dataMatch) {
            const eventName = eventMatch[1].trim();
            const dataStr = dataMatch[1].trim();

            try {
              const dataObj = JSON.parse(dataStr);

              if (eventName === 'sources') {
                setSources(dataObj);
              } else if (eventName === 'relations') {
                setRelations(dataObj);
              } else if (eventName === 'message') {
                const token = dataObj.content;
                assistantText += token;
                // 更新最新的 Assistant 回复
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    role: 'assistant',
                    content: assistantText
                  };
                  return updated;
                });
              } else if (eventName === 'error') {
                console.error('SSE backend error:', dataObj.error);
                assistantText += `\n[系统错误]: ${dataObj.error}`;
              }
            } catch (err) {
              console.error('Failed to parse SSE data', err, line);
            }
          }
        }
      }
    } catch (e) {
      console.error('Failed to send chat stream', e);
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantMessageIndex] = {
          role: 'assistant',
          content: `网络连接异常，无法获取回答。请检查后端服务是否正常运行。(${e})`
        };
        return updated;
      });
    } finally {
      setIsSending(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSources([]);
    setRelations([]);
  };

  return {
    messages,
    sources,
    relations,
    isSending,
    playgroundData,
    isSearching,
    searchPlayground,
    sendChatMessage,
    clearChat
  };
}
