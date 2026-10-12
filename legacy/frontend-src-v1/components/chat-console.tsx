import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, User, RefreshCw, FileText, Network, Trash2, HelpCircle, Paperclip, Mic, MicOff, X, File } from 'lucide-react';
import type { ChatHistoryItem, Source, Relation } from '../hooks/use-chat';
import type { KnowledgeBase } from '../hooks/use-kb';

interface ChatConsoleProps {
  currentKb: KnowledgeBase | null;
  messages: ChatHistoryItem[];
  sources: Source[];
  relations: Relation[];
  isSending: boolean;
  sendChatMessage: (kbId: string, messageText: string, imageB64?: string, tempDocContent?: string) => Promise<void>;
  clearChat: () => void;
}

const API_BASE = 'http://localhost:8000/api/v1';

export function ChatConsole({
  currentKb,
  messages,
  sources,
  relations,
  isSending,
  sendChatMessage,
  clearChat,
}: ChatConsoleProps) {
  const [input, setInput] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 附件与图片状态
  const [attachedImage, setAttachedImage] = useState<string | null>(null); // Base64
  const [attachedDoc, setAttachedDoc] = useState<{ name: string; content: string } | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // 语音输入状态
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<any>(null);

  // 自动滚动到底部
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 初始化 SpeechRecognition (ASR)
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = 'zh-CN'; // 默认中文识别

      rec.onresult = (event: any) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setInput(prev => prev + finalTranscript);
        }
      };

      rec.onerror = (e: any) => {
        console.error('Speech recognition error:', e);
        setIsRecording(false);
      };

      rec.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = rec;
    }
  }, []);

  // 开启/关闭语音录入
  const toggleRecording = () => {
    if (!recognitionRef.current) {
      alert('当前浏览器不支持语音输入接口，请使用 Chrome/Edge/Safari 浏览器。');
      return;
    }

    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
      } catch (e) {
        console.error('Failed to start speech recognition:', e);
      }
    }
  };

  // 处理附件上传
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    const file = files[0];
    const fileExt = file.name.split('.').pop()?.toLowerCase();
    
    if (!fileExt) return;

    // 1. 如果是图片类型文件 -> 转化为 Base64
    if (['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(fileExt)) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setAttachedImage(event.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    } 
    // 2. 如果是文本类型文件 -> 前端读取内容，零延迟
    else if (['txt', 'md', 'markdown'].includes(fileExt)) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setAttachedDoc({
            name: file.name,
            content: event.target.result as string
          });
        }
      };
      reader.readAsText(file);
    } 
    // 3. 如果是 PDF/Excel/PPT 类型文件 -> 发送给后端临时解析
    else if (['pdf', 'xlsx', 'xls', 'pptx', 'ppt'].includes(fileExt)) {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const res = await fetch(`${API_BASE}/documents/parse-temp`, {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          const data = await res.json();
          setAttachedDoc({
            name: file.name,
            content: data.content
          });
        } else {
          alert('临时文件解析提取文本失败，请检查文件格式是否损坏。');
        }
      } catch (err) {
        console.error('Error parsing temp document:', err);
        alert('解析失败，请检查后端网络连接。');
      } finally {
        setIsUploading(false);
      }
    } else {
      alert('不支持的文件格式！目前支持图片，以及 PDF, EXCEL, PPT, TXT, MD 常用办公文档临时解析。');
    }

    // 重置输入框 value 允许重复选相同文件
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // 提交问题
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isSending || !currentKb) return;

    // 发送消息
    sendChatMessage(
      currentKb.id, 
      input, 
      attachedImage || undefined, 
      attachedDoc?.content || undefined
    );
    
    // 清空输入与附件
    setInput('');
    setAttachedImage(null);
    setAttachedDoc(null);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px', height: '100%', overflow: 'hidden' }} className="animate-fade-in">
      {/* 左侧：智能问答主界面 */}
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* 对话头部 */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ background: 'var(--brand-gradient)', width: '32px', height: '32px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Sparkles size={16} color="white" />
            </div>
            <div>
              <h3 style={{ fontSize: '15px', fontWeight: 600 }}>智能问答控制台</h3>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                {currentKb ? `正在访问知识库: ${currentKb.name}` : '未选择知识库'}
              </span>
            </div>
          </div>
          {messages.length > 0 && (
            <button 
              className="secondary" 
              onClick={clearChat}
              style={{ padding: '6px 10px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', borderRadius: '6px' }}
            >
              <Trash2 size={12} />
              清空对话
            </button>
          )}
        </div>

        {/* 聊天气泡滚动区 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {messages.map((msg, idx) => {
            const isUser = msg.role === 'user';
            return (
              <div 
                key={idx} 
                style={{ 
                  display: 'flex', 
                  gap: '12px', 
                  alignSelf: isUser ? 'flex-end' : 'flex-start',
                  flexDirection: isUser ? 'row-reverse' : 'row',
                  maxWidth: '85%'
                }}
              >
                {/* 头像 */}
                <div style={{ 
                  width: '36px', 
                  height: '36px', 
                  borderRadius: '50%', 
                  background: isUser ? 'rgba(0, 210, 255, 0.15)' : 'var(--bg-glass-active)',
                  border: isUser ? '1px solid var(--brand-blue)' : '1px solid var(--border-light)',
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  {isUser ? <User size={16} style={{ color: 'var(--brand-blue)' }} /> : <Sparkles size={16} className="gradient-text" style={{ color: 'var(--brand-purple)' }} />}
                </div>

                {/* 气泡内容 */}
                <div style={{ 
                  padding: '12px 16px', 
                  borderRadius: isUser ? '16px 16px 2px 16px' : '16px 16px 16px 2px',
                  background: isUser ? 'var(--brand-gradient)' : 'rgba(255,255,255,0.02)',
                  border: isUser ? 'none' : '1px solid var(--border-light)',
                  boxShadow: isUser ? '0 4px 12px rgba(121, 40, 202, 0.15)' : 'none',
                  fontSize: '14px',
                  lineHeight: 1.6,
                  color: isUser ? 'white' : '#e5e7eb',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {msg.content === '' && isSending ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <RefreshCw size={14} style={{ animation: 'spin 1.5s linear infinite' }} />
                      <span style={{ color: 'var(--text-secondary)' }}>数据检索与实体关系编织中...</span>
                    </div>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            );
          })}

          {messages.length === 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
              <Sparkles size={48} style={{ opacity: 0.1, marginBottom: '12px' }} />
              <h3 style={{ fontSize: '15px', fontWeight: 500 }}>您好！我是您的 Hybrid RAG 智能助手</h3>
              <p style={{ fontSize: '13px', textAlign: 'center', maxWidth: '380px', marginTop: '6px', lineHeight: 1.5 }}>
                {currentKb ? `提一个关于“${currentKb.name}”的问题，我将融合向量、BM25 和 UnWeaver 知识图谱为您详细解答。` : '请在上方或“知识库管理”中选择一个知识库开始对话。'}
              </p>
              <div style={{ marginTop: '20px', display: 'flex', gap: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
                <span>• 支持多模态图片分析</span>
                <span>• 支持临时文档即时提取</span>
                <span>• 支持麦克风流式语音转写</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* 附件预览挂载区 */}
        {(attachedImage || attachedDoc || isUploading) && (
          <div style={{ display: 'flex', gap: '10px', padding: '10px 20px', background: 'rgba(0,0,0,0.1)', borderTop: '1px solid var(--border-light)' }}>
            {attachedImage && (
              <div style={{ position: 'relative', width: '60px', height: '60px', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-active)' }}>
                <img src={attachedImage} alt="attachment" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                <button 
                  onClick={() => setAttachedImage(null)}
                  style={{ position: 'absolute', top: 2, right: 2, background: 'rgba(0,0,0,0.6)', border: 'none', borderRadius: '50%', width: '16px', height: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', padding: 0 }}
                >
                  <X size={10} color="white" />
                </button>
              </div>
            )}

            {attachedDoc && (
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 12px 6px 8px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-light)', borderRadius: '6px', fontSize: '11px', height: '60px' }}>
                <File size={16} style={{ color: 'var(--brand-blue)' }} />
                <div style={{ maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {attachedDoc.name}
                </div>
                <button 
                  onClick={() => setAttachedDoc(null)}
                  style={{ position: 'absolute', top: 2, right: 2, background: 'rgba(0,0,0,0.6)', border: 'none', borderRadius: '50%', width: '16px', height: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', padding: 0 }}
                >
                  <X size={10} color="white" />
                </button>
              </div>
            )}

            {isUploading && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 14px', background: 'rgba(0, 210, 255, 0.05)', borderRadius: '6px', border: '1px solid rgba(0, 210, 255, 0.1)', fontSize: '11px', color: 'var(--brand-blue)' }}>
                <RefreshCw size={12} style={{ animation: 'spin 1.5s linear infinite' }} />
                <span>正在提取并解析文档内容...</span>
              </div>
            )}
          </div>
        )}

        {/* 底部输入栏 */}
        <div style={{ padding: '20px', borderTop: '1px solid var(--border-light)', position: 'relative' }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            
            {/* 隐藏的文件上传 input */}
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              onChange={handleFileChange}
              accept="image/*,.pdf,.txt,.md,.markdown,.xlsx,.xls,.pptx,.ppt"
            />

            {/* 回形针上传按钮 */}
            <button 
              type="button" 
              onClick={() => fileInputRef.current?.click()}
              disabled={!currentKb || isSending}
              className="secondary" 
              style={{ width: '42px', height: '42px', borderRadius: '8px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}
              title="上传图片或文档进行临时分析"
            >
              <Paperclip size={16} />
            </button>

            {/* 输入框主栏 */}
            <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center' }}>
              <input 
                type="text" 
                placeholder={
                  isRecording 
                    ? '正在录音识别中，请说话...' 
                    : currentKb 
                      ? '向知识库助手提问 (可上传图片/文档，可点击麦克风输入)' 
                      : '请先选择一个知识库'
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={!currentKb || isSending}
                style={{ width: '100%', fontSize: '14px', padding: '12px 46px 12px 16px' }}
              />

              {/* 麦克风录音控制按钮 */}
              <button
                type="button"
                onClick={toggleRecording}
                disabled={!currentKb || isSending}
                style={{
                  position: 'absolute',
                  right: '10px',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  padding: '6px',
                  color: isRecording ? '#ef4444' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '50%'
                }}
                className={isRecording ? 'recording-mic-active' : ''}
                title="语音输入"
              >
                {isRecording ? <Mic size={18} style={{ animation: 'pulse 1.2s infinite' }} /> : <MicOff size={18} />}
              </button>
            </div>

            {/* 发送按钮 */}
            <button 
              type="submit" 
              className="primary" 
              disabled={!currentKb || isSending || (!input.trim() && !attachedImage && !attachedDoc)}
              style={{ width: '46px', height: '46px', borderRadius: '8px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}
            >
              <Send size={16} />
            </button>
          </form>

          {/* 声波律动图形效果 (只在录音时显示) */}
          {isRecording && (
            <div style={{ display: 'flex', justifyContent: 'center', gap: '4px', height: '14px', marginTop: '10px' }}>
              <div className="voice-bar" style={{ animationDelay: '0.1s' }} />
              <div className="voice-bar" style={{ animationDelay: '0.3s' }} />
              <div className="voice-bar" style={{ animationDelay: '0.5s' }} />
              <div className="voice-bar" style={{ animationDelay: '0.2s' }} />
              <div className="voice-bar" style={{ animationDelay: '0.4s' }} />
              <div className="voice-bar" style={{ animationDelay: '0.6s' }} />
            </div>
          )}
        </div>
      </div>

      {/* 右侧：引用来源与 UnWeaver 编织看板 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', overflow: 'hidden' }}>
        {/* 引用来源卡片 */}
        <div className="glass-panel" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <h4 style={{ fontSize: '13px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
            <FileText size={14} style={{ color: 'var(--brand-blue)' }} />
            本轮引用来源 (Citations)
          </h4>

          <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {sources.map((src, idx) => (
              <div 
                key={src.id} 
                style={{ 
                  padding: '10px', 
                  borderRadius: '8px', 
                  background: src.isWeaved ? 'rgba(236, 72, 153, 0.04)' : 'rgba(255,255,255,0.01)',
                  border: src.isWeaved ? '1px solid rgba(236, 72, 153, 0.15)' : '1px solid rgba(255,255,255,0.04)',
                  fontSize: '12px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-secondary)', marginBottom: '4px', fontSize: '10px' }}>
                  <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '160px' }} title={src.fileName}>
                    [{idx + 1}] {src.fileName}
                  </span>
                  <span style={{ color: src.isWeaved ? 'var(--color-weaved)' : 'var(--color-dense)' }}>
                    {src.isWeaved ? 'Weaved' : `Score: ${src.score.toFixed(1)}`}
                  </span>
                </div>
                <div style={{ color: 'var(--text-primary)', lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' }}>
                  {src.content}
                </div>
                {src.isWeaved && (
                  <div style={{ marginTop: '4px', fontSize: '10px', color: 'var(--color-weaved)', fontWeight: 500 }}>
                    ★ 触发名词: “{src.weavedEntity}” 的定义扩展织入
                  </div>
                )}
              </div>
            ))}

            {sources.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                <HelpCircle size={24} style={{ opacity: 0.15, marginBottom: '6px' }} />
                <span style={{ fontSize: '11px' }}>等待问答检索产生引用来源...</span>
              </div>
            )}
          </div>
        </div>

        {/* UnWeaver 知识编织关系桥 */}
        <div className="glass-panel" style={{ height: '220px', padding: '16px', display: 'flex', flexDirection: 'column', overflow: 'hidden', borderLeft: '1px solid var(--border-light)' }}>
          <h4 style={{ fontSize: '13px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
            <Network size={14} style={{ color: 'var(--color-weaved)' }} />
            UnWeaver 实体关系织网
          </h4>

          <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {relations.map((rel, idx) => (
              <div key={idx} style={{ padding: '8px 10px', borderRadius: '6px', background: 'rgba(236, 72, 153, 0.05)', borderLeft: '3px solid var(--color-weaved)', fontSize: '11px' }}>
                <div style={{ fontWeight: 600, color: 'var(--color-weaved)' }}>
                  识别术语: {rel.entity}
                </div>
                <div style={{ color: 'var(--text-secondary)', marginTop: '2px', lineHeight: 1.4 }}>
                  已自动织入定义：{rel.definition}
                </div>
              </div>
            ))}

            {relations.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', textAlign: 'center', padding: '10px' }}>
                <Network size={22} style={{ opacity: 0.15, marginBottom: '4px' }} />
                <span style={{ fontSize: '10px', lineHeight: 1.3 }}>本轮对话大模型上下文未注入编织实体定义。</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
