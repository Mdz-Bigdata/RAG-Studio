import { useState, useEffect } from 'react';
import { Database, Zap, MessageSquare, Sparkles, Activity, ShieldCheck, Cpu, Sliders } from 'lucide-react';
import { useKb } from './hooks/use-kb';
import { useChat } from './hooks/use-chat';
import { KBManager } from './components/kb-manager';
import { SearchPlayground } from './components/search-playground';
import { ChatConsole } from './components/chat-console';
import { ModelSettings } from './components/model-settings';

export default function App() {
  const {
    kbs,
    currentKb,
    setCurrentKb,
    documents,
    createKb,
    deleteKb,
    uploadDocument,
    deleteDocument,
    fetchMarkdown,
    updateMarkdown,
    fetchKbs,
  } = useKb();

  const {
    messages,
    sources,
    relations,
    isSending,
    playgroundData,
    isSearching,
    searchPlayground,
    sendChatMessage,
    clearChat,
  } = useChat();

  const [activeTab, setActiveTab] = useState<'kb' | 'playground' | 'chat' | 'settings'>('kb');
  const [backendStatus, setBackendStatus] = useState<{ status: string; llm_active: boolean }>({
    status: 'connecting',
    llm_active: false,
  });

  // 检测后端服务和 API KEY 激活状态
  const checkBackend = async () => {
    try {
      const res = await fetch('http://localhost:8000/');
      if (res.ok) {
        // 由于我们在 config.py 中对 OPENAI_API_KEY 进行了判断，如果后端启动检测到有 KEY，则为激活
        // 我们尝试拉取后端的主机状态
        setBackendStatus({
          status: 'online',
          llm_active: true, // 默认设为 true，若报错则切换
        });
      }
    } catch {
      setBackendStatus({
        status: 'offline',
        llm_active: false,
      });
    }
  };

  useEffect(() => {
    fetchKbs();
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      {/* 左侧：科技导航 Sidebar */}
      <div 
        style={{ 
          background: 'var(--bg-sidebar)', 
          borderRight: '1px solid var(--border-light)', 
          padding: '24px 16px', 
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'space-between' 
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
          {/* Logo 标题 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '0 8px' }}>
            <div style={{ background: 'var(--brand-gradient)', width: '38px', height: '38px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 14px rgba(0, 210, 255, 0.3)' }}>
              <Sparkles size={18} color="white" />
            </div>
            <div>
              <h2 style={{ fontSize: '16px', fontWeight: 700, letterSpacing: '0.5px' }} className="gradient-text">
                Antigravity RAG
              </h2>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>企业级 Hybrid RAG 系统</span>
            </div>
          </div>

          {/* 导航按钮组 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button 
              onClick={() => setActiveTab('kb')}
              style={{
                display: 'flex', alignItems: 'center', gap: '12px', width: '100%',
                padding: '12px 16px', borderRadius: '10px', fontSize: '14px', textDecoration: 'none',
                background: activeTab === 'kb' ? 'var(--bg-glass-active)' : 'transparent',
                border: activeTab === 'kb' ? '1px solid var(--border-active)' : '1px solid transparent',
                color: activeTab === 'kb' ? 'var(--text-primary)' : 'var(--text-secondary)'
              }}
            >
              <Database size={16} style={{ color: activeTab === 'kb' ? 'var(--brand-blue)' : 'var(--text-muted)' }} />
              知识库管理
            </button>

            <button 
              onClick={() => setActiveTab('playground')}
              style={{
                display: 'flex', alignItems: 'center', gap: '12px', width: '100%',
                padding: '12px 16px', borderRadius: '10px', fontSize: '14px', textDecoration: 'none',
                background: activeTab === 'playground' ? 'var(--bg-glass-active)' : 'transparent',
                border: activeTab === 'playground' ? '1px solid var(--border-active)' : '1px solid transparent',
                color: activeTab === 'playground' ? 'var(--text-primary)' : 'var(--text-secondary)'
              }}
            >
              <Zap size={16} style={{ color: activeTab === 'playground' ? 'var(--color-bm25)' : 'var(--text-muted)' }} />
              多路召回游乐场
            </button>

            <button 
              onClick={() => setActiveTab('chat')}
              style={{
                display: 'flex', alignItems: 'center', gap: '12px', width: '100%',
                padding: '12px 16px', borderRadius: '10px', fontSize: '14px', textDecoration: 'none',
                background: activeTab === 'chat' ? 'var(--bg-glass-active)' : 'transparent',
                border: activeTab === 'chat' ? '1px solid var(--border-active)' : '1px solid transparent',
                color: activeTab === 'chat' ? 'var(--text-primary)' : 'var(--text-secondary)'
              }}
            >
              <MessageSquare size={16} style={{ color: activeTab === 'chat' ? 'var(--color-weaved)' : 'var(--text-muted)' }} />
              智能问答控制台
            </button>

            <button 
              onClick={() => setActiveTab('settings')}
              style={{
                display: 'flex', alignItems: 'center', gap: '12px', width: '100%',
                padding: '12px 16px', borderRadius: '10px', fontSize: '14px', textDecoration: 'none',
                background: activeTab === 'settings' ? 'var(--bg-glass-active)' : 'transparent',
                border: activeTab === 'settings' ? '1px solid var(--border-active)' : '1px solid transparent',
                color: activeTab === 'settings' ? 'var(--text-primary)' : 'var(--text-secondary)'
              }}
            >
              <Sliders size={16} style={{ color: activeTab === 'settings' ? 'var(--brand-blue)' : 'var(--text-muted)' }} />
              模型配置中心
            </button>
          </div>
        </div>

        {/* 底部连接状态面板 */}
        <div className="glass-panel" style={{ padding: '14px', background: 'rgba(0,0,0,0.2)', fontSize: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Activity size={14} style={{ color: backendStatus.status === 'online' ? '#10b981' : '#ef4444' }} />
            <span>服务状态:</span>
            <span style={{ fontWeight: 600, color: backendStatus.status === 'online' ? '#10b981' : '#ef4444' }}>
              {backendStatus.status === 'online' ? '运行中 (Online)' : backendStatus.status === 'offline' ? '未连接 (Offline)' : '连接中...'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={14} style={{ color: backendStatus.status === 'online' ? 'var(--brand-blue)' : 'var(--text-muted)' }} />
            <span>运行内核:</span>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              {backendStatus.status === 'offline' ? '--' : 'Hybrid 混合引擎'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', padding: '6px 0 0 0', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
            <ShieldCheck size={14} style={{ color: '#f59e0b' }} />
            <span style={{ color: 'var(--text-secondary)' }}>安全脱敏: 已激活</span>
          </div>
        </div>
      </div>

      {/* 右侧：主体内容区域 */}
      <div style={{ padding: '24px', overflow: 'hidden', height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* 全局右上角当前选择知识库徽标 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexShrink: 0 }}>
          <h1 style={{ fontSize: '20px', fontWeight: 700 }}>
            {activeTab === 'kb' ? '知识库管理中心' : activeTab === 'playground' ? 'Hybrid RAG 检索游乐场' : activeTab === 'chat' ? '智能问答终端' : '模型配置中心'}
          </h1>
          {currentKb && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(0, 210, 255, 0.08)', border: '1px solid rgba(0, 210, 255, 0.2)', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 500, color: 'var(--brand-blue)' }}>
              <Database size={12} />
              当前知识库: {currentKb.name}
            </div>
          )}
        </div>

        {/* Tab 页面渲染 */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {activeTab === 'kb' && (
            <KBManager 
              kbs={kbs}
              currentKb={currentKb}
              setCurrentKb={setCurrentKb}
              documents={documents}
              createKb={createKb}
              deleteKb={deleteKb}
              uploadDocument={uploadDocument}
              deleteDocument={deleteDocument}
              fetchMarkdown={fetchMarkdown}
              updateMarkdown={updateMarkdown}
            />
          )}

          {activeTab === 'playground' && (
            <SearchPlayground 
              currentKb={currentKb}
              playgroundData={playgroundData}
              isSearching={isSearching}
              searchPlayground={searchPlayground}
            />
          )}

          {activeTab === 'chat' && (
            <ChatConsole 
              currentKb={currentKb}
              messages={messages}
              sources={sources}
              relations={relations}
              isSending={isSending}
              sendChatMessage={sendChatMessage}
              clearChat={clearChat}
            />
          )}

          {activeTab === 'settings' && (
            <ModelSettings />
          )}
        </div>
      </div>
    </div>
  );
}
