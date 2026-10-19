import { useEffect, useState } from 'react';
import { BarChart3, Waves, Boxes, Brain, Compass, ChevronDown, ChevronLeft, ChevronRight, Cpu, Database, FileText, FlaskConical, HeartPulse, Images, LayoutDashboard, MessageSquare, Moon, ScanSearch, Settings2, Sun, Target, Users, X } from 'lucide-react';
import { IdentitySelect } from './components/identity-select';
import { AppProvider, useApp } from './store/app-store';
import { DashboardPage } from './pages/dashboard';
import { KnowledgeBasesPage } from './pages/knowledge-bases';
import { CoveragePage } from './pages/coverage';
import { CorpusPage } from './pages/corpus';
import { PlaygroundPage } from './pages/playground';
import { MultimodalPage } from './pages/multimodal';
import { ChatPage } from './pages/chat';
import { EvalPage } from './pages/eval';
import { ModelsPage } from './pages/models';
import { SettingsPage } from './pages/settings';
import { LakePage } from './pages/lake';
import { FeishuPage } from './pages/feishu';
import { LanceDBPage } from './pages/lancedb';
import { MemoryPage } from './pages/memory';
import { AgentsPage } from './pages/agents';
import { RecommendPage } from './pages/recommend';
import { applyTheme, currentTheme, type Theme } from './lib/theme';

export type PageKey = 'dashboard' | 'kb' | 'feishu' | 'coverage' | 'corpus' | 'playground' | 'multimodal' | 'chat' | 'eval' | 'lake' | 'lancedb' | 'memory' | 'agents' | 'recommend' | 'models' | 'settings';

// 工作流按实际调优顺序排列：先把语料与切片做对，再调检索，最后用评测确认。
const PAGES: { key: PageKey; label: string; title: string; subtitle: string; icon: React.ReactNode; section?: string }[] = [
  { key: 'dashboard', label: '总览', title: '系统总览', subtitle: '运行状态、六项核心检索能力与当前生效的管线配置', icon: <LayoutDashboard size={16} className="nav-icon" /> },
  { key: 'kb', label: '知识库', title: '知识库与文档', subtitle: 'MinerU / Markdown 解析 → 清洗 → 语义感知切片 → Parent-Child → 向量 + BM25 索引', icon: <Database size={16} className="nav-icon" />, section: '工作流' },
  { key: 'feishu', label: '飞书文档', title: '飞书文档', subtitle: '飞书文档同步进知识库（按版本增量）、远程增删改查，文档智能体用自然语言查询 / 分析 / 修改飞书文档', icon: <FileText size={16} className="nav-icon" /> },
  { key: 'coverage', label: '语料覆盖', title: '语料准备：意图 × 知识覆盖', subtitle: '把用户会问的问题与现有语料对齐：哪些知识还没准备好，哪些材料与目标无关', icon: <Target size={16} className="nav-icon" /> },
  { key: 'corpus', label: '语料治理', title: '语料优化与增强', subtitle: '质量体检与剔除 → 覆盖缺口 FAQ 增强 → 别名 / 关键词扩展 → 自动元数据', icon: <HeartPulse size={16} className="nav-icon" /> },
  { key: 'playground', label: '检索实验室', title: '检索实验室', subtitle: '查询层 → 多路召回 → RRF 融合 → 精排 → Small-to-Big，每一层可见、可按请求覆盖参数', icon: <FlaskConical size={16} className="nav-icon" /> },
  { key: 'multimodal', label: '图文检索', title: '图文检索实验室', subtitle: '文搜图 / 图搜图 / 图文混合 / 图搜文：图文统一向量 + 图片描述向量 + BM25，多路融合', icon: <Images size={16} className="nav-icon" /> },
  { key: 'chat', label: '智能问答', title: '智能问答', subtitle: '基于回答协议的流式问答：引用可追溯，回答可带图片；支持以文搜图、以图搜图、以图搜文', icon: <MessageSquare size={16} className="nav-icon" /> },
  { key: 'eval', label: '评测中心', title: '评测中心', subtitle: '黄金集 → 分阶段消融 → Hit / MRR / NDCG → 错误归因', icon: <BarChart3 size={16} className="nav-icon" /> },
  { key: 'lake', label: 'Agent 湖', title: 'Agent 湖分析', subtitle: '问题 / 检索日志 / 回答 / 反馈 / 记忆 / 黑板沉淀在同一张 LanceDB 宽表：检索质量、知识缺口、主题、SQL、分析智能体；版本 · Tag · 分支 · 派生列', icon: <Waves size={16} className="nav-icon" /> },
  { key: 'lancedb', label: 'LanceDB', title: 'LanceDB 控制台', subtitle: '嵌入式多模态数据库：知识库向量（后端 / 镜像）、向量 · 全文 · 混合检索、ANN 索引与 optimize、版本时间旅行、图文同表', icon: <Boxes size={16} className="nav-icon" />, section: 'LanceDB 应用' },
  { key: 'memory', label: '长期记忆', title: '长期记忆', subtitle: '按用户记住身份、偏好与事实：相似度 × 新近度 × 重要度召回，对话后自动抽取、去重合并、可遗忘', icon: <Brain size={16} className="nav-icon" /> },
  { key: 'agents', label: '多智能体', title: '多智能体协作', subtitle: '规划 → 并行研究 → 分析 → 审校（证据不足改写重查）→ 撰稿，共享 LanceDB 黑板，可回放', icon: <Users size={16} className="nav-icon" /> },
  { key: 'recommend', label: '个性化推荐', title: '个性化推荐', subtitle: '交互与长期记忆构成兴趣向量：LanceDB 近邻召回 → 热度加权 → MMR 多样化，每条推荐都有理由', icon: <Compass size={16} className="nav-icon" /> },
  { key: 'models', label: '模型配置', title: '模型配置中心', subtitle: '文本 / 嵌入 / 重排 / 多模态 / 图文向量五类模型自由组合，支持 Anthropic 与 OpenAI 兼容端点', icon: <Cpu size={16} className="nav-icon" />, section: '配置' },
  { key: 'settings', label: '策略配置', title: '检索策略配置', subtitle: '解析、切片、召回、融合、精排、上下文工程的全部参数', icon: <Settings2 size={16} className="nav-icon" /> },
];

function pageFromHash(): PageKey {
  const h = window.location.hash.replace('#/', '').replace('#', '') as PageKey;
  return PAGES.some((p) => p.key === h) ? h : 'dashboard';
}

/** 侧栏分组：有 section 的页面开启新的一组，后面没有 section 的页面归入这一组；总览不属于任何组 */
const NAV_GROUPS = PAGES.reduce<{ section?: string; pages: typeof PAGES }[]>((groups, p) => {
  if (p.section || !groups.length) groups.push({ section: p.section, pages: [] });
  groups[groups.length - 1].pages.push(p);
  return groups;
}, []);
const NAV_FOLD_KEY = 'rag.navFolded';

function readFolded(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(NAV_FOLD_KEY) ?? '[]');
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
  } catch { return []; }
}

function Shell() {
  const { status, backendOnline, currentKb, toasts, dismissToast, refreshKbs, refreshStatus } = useApp();
  const [page, setPage] = useState<PageKey>(pageFromHash);
  const [collapsed, setCollapsed] = useState(false);
  const [folded, setFolded] = useState<string[]>(readFolded);
  const toggleGroup = (section: string) => setFolded((cur) => {
    const next = cur.includes(section) ? cur.filter((s) => s !== section) : [...cur, section];
    try { localStorage.setItem(NAV_FOLD_KEY, JSON.stringify(next)); } catch { /* 无痕模式等不可写时只在本次会话生效 */ }
    return next;
  });
  const [theme, setTheme] = useState<Theme>(currentTheme);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  // 切页时刷新共享数据，保证各页看到的知识库 / 运行状态一致
  useEffect(() => {
    refreshKbs().catch(() => undefined);
    refreshStatus();
  }, [page, refreshKbs, refreshStatus]);

  const go = (key: PageKey) => {
    window.location.hash = `#/${key}`;
    setPage(key);
  };
  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    setTheme(next);
  };

  const meta = PAGES.find((p) => p.key === page)!;
  const rt = status?.runtime;
  const llmOk = rt ? !rt.llm.offline : false;
  const embOk = rt ? !rt.embedding.offline : false;
  const fullHeight = page === 'chat' || page === 'playground' || page === 'multimodal';

  return (
    <div className={`app ${collapsed ? 'collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo"><ScanSearch size={16} /></div>
          {!collapsed ? (
            <div>
              <div className="brand-name">RAG Agent Studio</div>
              <div className="brand-sub">hybrid · mineru · popo</div>
            </div>
          ) : null}
        </div>
        <nav className="nav">
          {NAV_GROUPS.map((g) => {
            // 收起侧栏（只剩图标）时不显示分组标题，也不折叠，所有入口都可点
            const isFolded = !collapsed && !!g.section && folded.includes(g.section);
            const hasActive = g.pages.some((p) => p.key === page);
            return (
              <div key={g.section ?? '_top'} className={`nav-group ${isFolded ? 'folded' : ''}`}>
                {g.section && !collapsed ? (
                  <button className={`nav-section ${isFolded && hasActive ? 'has-active' : ''}`} onClick={() => toggleGroup(g.section!)} aria-expanded={!isFolded} title={isFolded ? `展开${g.section}` : `收起${g.section}`}>
                    <span>{g.section}</span>
                    <ChevronDown size={12} className={`nav-chevron ${isFolded ? 'folded' : ''}`} />
                  </button>
                ) : null}
                {g.pages.map((p) => (
                  <button key={p.key} className={`nav-item ${page === p.key ? 'active' : ''}`} onClick={() => go(p.key)} title={p.label} aria-current={page === p.key ? 'page' : undefined}>
                    {p.icon}
                    {!collapsed ? <span>{p.label}</span> : null}
                  </button>
                ))}
              </div>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          {!collapsed ? (
            <div className="status-card">
              <div className="status-row"><span className={`dot ${backendOnline ? 'ok' : 'bad'}`} />后端<b>{backendOnline ? '在线' : '离线'}</b></div>
              <div className="status-row"><span className={`dot ${llmOk ? 'ok' : 'warn'}`} />LLM<b title={rt?.llm.model ?? ''}>{llmOk ? rt?.llm.model : '抽取式回答'}</b></div>
              <div className="status-row"><span className={`dot ${embOk ? 'ok' : 'warn'}`} />嵌入<b title={rt?.embedding.model ?? ''}>{rt?.embedding.model ?? '—'}</b></div>
              <div className="status-row"><span className={`dot ${rt?.rerank.offline ? 'warn' : 'ok'}`} />重排<b>{rt?.rerank.provider ?? '—'}</b></div>
              <div className="status-row"><span className={`dot ${rt?.mineruAvailable ? 'ok' : 'warn'}`} />解析<b>{rt?.mineruAvailable ? `mineru${rt.mineruMode === 'cloud' ? ' 云端' : rt.mineruMode === 'server' ? ' 服务' : ''} · ${rt.mineruTier}` : '内置解析器'}</b></div>
            </div>
          ) : null}
          <button className="nav-item" onClick={() => setCollapsed((c) => !c)} title={collapsed ? '展开侧栏' : '收起侧栏'}>
            {collapsed ? <ChevronRight size={16} className="nav-icon" /> : <ChevronLeft size={16} className="nav-icon" />}
            {!collapsed ? <span>收起侧栏</span> : null}
          </button>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{meta.title}</h1>
            <div className="subtitle">{meta.subtitle}</div>
          </div>
          <div className="topbar-actions">
            <IdentitySelect />
            {currentKb ? <span className="badge" title="当前知识库（各页共用）"><Database size={11} />{currentKb.name}</span> : null}
            {status?.runtime.offlineMode ? <span className="badge warn" title="未配置模型：哈希向量 + BM25 + 词法精排 + 抽取式回答">离线体验模式</span> : null}
            <button className="btn ghost icon" onClick={toggleTheme} title={theme === 'dark' ? '切换到浅色' : '切换到深色'} aria-label="切换主题">
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>
        <div className={`content ${fullHeight ? 'no-scroll' : ''}`} key={page}>
          {page === 'dashboard' && <DashboardPage go={go} />}
          {page === 'kb' && <KnowledgeBasesPage />}
          {page === 'feishu' && <FeishuPage />}
          {page === 'coverage' && <CoveragePage go={go} />}
          {page === 'corpus' && <CorpusPage go={go} />}
          {page === 'playground' && <PlaygroundPage />}
          {page === 'multimodal' && <MultimodalPage />}
          {page === 'chat' && <ChatPage />}
          {page === 'eval' && <EvalPage />}
          {page === 'lake' && <LakePage />}
          {page === 'lancedb' && <LanceDBPage />}
          {page === 'memory' && <MemoryPage />}
          {page === 'agents' && <AgentsPage />}
          {page === 'recommend' && <RecommendPage />}
          {page === 'models' && <ModelsPage />}
          {page === 'settings' && <SettingsPage />}
        </div>
      </div>

      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.tone}`}>
            <span className="flex-1">{t.text}</span>
            <button className="btn ghost icon xs" onClick={() => dismissToast(t.id)} aria-label="关闭"><X size={12} /></button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
