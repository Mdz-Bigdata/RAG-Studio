// 个性化推荐（LanceDB 表 rec_items / rec_events）：兴趣向量召回 → 热度加权 → MMR 多样化，每条推荐带理由；点赞 / 点踩实时生效。
import { useCallback, useEffect, useRef, useState } from 'react';
import { Compass, FileText, HelpCircle, Image as ImageIcon, Layers, MessageCircleQuestion, RefreshCw, Sparkles, ThumbsDown, ThumbsUp, TrendingUp } from 'lucide-react';
import { ApiError } from '../api/client';
import { fmtTime, recommendApi, type RecItem, type RecKind, type RecResult, type RecStats, type RecVote } from '../api/lance';
import { KbSelect } from '../components/kb-select';
import { Badge, Button, Callout, Drawer, Empty, ScoreBar, Spinner, Tabs } from '../components/ui';
import { useApp } from '../store/app-store';

const KIND: Record<RecKind, { label: string; icon: React.ReactNode; tone: string }> = {
  section: { label: '章节', icon: <Layers size={12} />, tone: 'dense' },
  question: { label: '问题', icon: <HelpCircle size={12} />, tone: 'accent' },
  doc: { label: '文档', icon: <FileText size={12} />, tone: 'info' },
  image: { label: '图片', icon: <ImageIcon size={12} />, tone: 'hyde' },
};
const EVENT_LABEL: Record<string, string> = { ask: '提问', like: '点赞', dislike: '点踩', click: '点开', cite: '被引用', view: '浏览' };
const errText = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

export function RecommendPage() {
  const { currentKb, identity, toast } = useApp();
  const userId = identity.userId || 'default';
  const [kind, setKind] = useState<'all' | RecKind>('all');
  const [res, setRes] = useState<RecResult | null>(null);
  const [stats, setStats] = useState<RecStats | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [ask, setAsk] = useState('出差住宿超标了怎么报销');
  const [similar, setSimilar] = useState<{ item: RecItem; rows: RecItem[] } | null>(null);
  // 赞 / 踩是"评价"：每人每条只能一次，赞踩二选一。voting 用 ref 同步上锁——同一渲染周期里的连点、两个一起点也挡得住
  const [votes, setVotes] = useState<Record<string, RecVote>>({});
  const voting = useRef(new Set<string>());
  useEffect(() => { setVotes({}); voting.current.clear(); }, [userId]);

  const load = useCallback(async () => {
    try {
      const [r, s] = await Promise.all([recommendApi.get({ userId, kbId: currentKb?.id, kinds: kind === 'all' ? undefined : kind, k: 8 }), recommendApi.stats()]);
      setRes(r);
      setStats(s);
    } catch (e) {
      toast(`读取推荐失败：${errText(e)}`, 'bad');
    }
  }, [userId, currentKb, kind, toast]);
  useEffect(() => { load(); }, [load]);

  const sync = async () => {
    if (!currentKb) return;
    setSyncing(true);
    try {
      const r = await recommendApi.sync(currentKb.id);
      toast(`已同步 ${r.items} 个推荐条目（${Object.entries(r.byKind).map(([k, n]) => `${KIND[k as RecKind]?.label ?? k} ${n}`).join(' · ')}）`, 'ok');
      load();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setSyncing(false);
    }
  };
  const feedback = async (item: RecItem, k: 'click') => {
    try {
      await recommendApi.event({ userId, kind: k, itemId: item.id, kbId: item.kbId });
      load();
    } catch (e) {
      toast(errText(e), 'bad');
    }
  };
  // 评价后只刷新兴趣画像与统计，列表不动：评价过的那条留在原处，显示你选的是赞还是踩
  const refreshSide = async () => {
    try {
      const [r, s] = await Promise.all([recommendApi.get({ userId, kbId: currentKb?.id, kinds: kind === 'all' ? undefined : kind, k: 8 }), recommendApi.stats()]);
      setRes((prev) => (prev ? { ...prev, mode: r.mode, profile: r.profile, memoryInterests: r.memoryInterests } : r));
      setStats(s);
    } catch {
      /* 侧栏刷新失败不影响评价本身 */
    }
  };
  const vote = async (item: RecItem, k: RecVote) => {
    if (votes[item.id] || item.myVote || voting.current.has(item.id)) return;
    voting.current.add(item.id);
    setVotes((v) => ({ ...v, [item.id]: k }));
    try {
      await recommendApi.event({ userId, kind: k, itemId: item.id, kbId: item.kbId });
      toast(k === 'like' ? '已记录：会推荐更多相似内容' : '已记录：不再推荐它', 'ok');
      refreshSide();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        // 之前已经评价过（别的页面 / 设备）：以服务器记录为准
        let prev: RecVote = k;
        try {
          prev = (JSON.parse(e.message) as { vote?: RecVote }).vote ?? k;
        } catch {
          /* 不是结构化的说明 */
        }
        setVotes((v) => ({ ...v, [item.id]: prev }));
        toast('这条已经评价过了：每人每条只能评价一次', 'warn');
      } else {
        voting.current.delete(item.id);
        setVotes((v) => {
          const next = { ...v };
          delete next[item.id];
          return next;
        });
        toast(errText(e), 'bad');
      }
    }
  };
  const simulateAsk = async () => {
    if (!ask.trim()) return;
    await recommendApi.event({ userId, kind: 'ask', text: ask, kbId: currentKb?.id });
    toast('已记录这次提问（智能问答里的提问也会自动记录）', 'ok');
    load();
  };
  const openSimilar = async (item: RecItem) => {
    try {
      setSimilar({ item, rows: await recommendApi.similar(item.id) });
    } catch (e) {
      toast(errText(e), 'bad');
    }
  };

  const kbItems = stats?.items ?? 0;
  return (
    <div className="page col gap-16 fade-in">
      <div className="card stat-strip">
        <div className="stat"><div className="label"><Compass size={12} />当前用户</div><div className="value">{userId}</div><div className="sub">在右上角「用户」里切换</div></div>
        <div className="stat"><div className="label">推荐条目</div><div className="value">{kbItems}</div><div className="sub">{stats ? Object.entries(stats.itemsByKind).map(([k, n]) => `${KIND[k as RecKind]?.label ?? k} ${n}`).join(' · ') || '先同步候选条目' : '—'}</div></div>
        <div className="stat"><div className="label">交互事件</div><div className="value">{stats?.events ?? 0}</div><div className="sub">{stats?.users ?? 0} 个用户</div></div>
        <div className="stat"><div className="label">推荐模式</div><div className="value" style={{ fontSize: 16, paddingTop: 6 }}>{res?.mode === 'personal' ? '个性化' : res?.mode === 'popular' ? '冷启动（热门）' : '—'}</div><div className="sub">{res?.memoryInterests ? `叠加了 ${res.memoryInterests} 条记忆偏好` : '兴趣向量来自交互与长期记忆'}</div></div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 320px', alignItems: 'start', gap: 16 }}>
        <div className="card">
          <div className="card-header">
            <h3><Sparkles size={14} />为你推荐</h3>
            <div className="row gap-8 wrap">
              <KbSelect />
              <Tabs value={kind} onChange={setKind} items={[{ value: 'all', label: '全部' }, ...(Object.keys(KIND) as RecKind[]).map((k) => ({ value: k, label: KIND[k].label }))]} />
              <Button size="sm" onClick={sync} loading={syncing} disabled={!currentKb}><RefreshCw size={13} />同步候选条目</Button>
            </div>
          </div>
          <div className="card-body col gap-8">
            {res === null ? <Spinner label="加载中…" /> : res.items.length === 0 ? (
              <Empty icon={<Compass size={32} />} title={res.mode === 'empty' ? '还没有候选条目' : '暂时没有可推荐的内容'} desc={res.mode === 'empty' ? '点「同步候选条目」：把当前知识库的章节、文档、生成的问题、图片写进 LanceDB。' : '看过 / 点过的不会重复推荐，可以换个类型或知识库看看。'} />
            ) : res.items.map((it) => (
              <div key={it.id} className="chunk">
                <div className="row between gap-8">
                  <button className="btn ghost sm" style={{ padding: 0, height: 'auto', textAlign: 'left' }} onClick={() => feedback(it, 'click')} title="点开（记为一次点击）">
                    <span className="row gap-6"><Badge tone={KIND[it.kind]?.tone}>{KIND[it.kind]?.icon}{KIND[it.kind]?.label}</Badge><span className="small strong">{it.title}</span></span>
                  </button>
                  <span className="mono xs faint">{(it.score ?? 0).toFixed(3)}</span>
                </div>
                {it.text ? <div className="xs muted clamp-2 mt-4">{it.text}</div> : null}
                <div className="row between gap-8 mt-8">
                  <span className="xs" style={{ color: 'var(--accent)' }}><TrendingUp size={11} /> {it.reason}</span>
                  {(() => {
                    const mine = votes[it.id] ?? it.myVote ?? null;
                    const tip = (v: RecVote) => (mine === null ? (v === 'like' ? '喜欢' : '不喜欢') : mine === v ? `已${v === 'like' ? '点赞' : '点踩'}（每人每条只能评价一次）` : `已${mine === 'like' ? '点赞' : '点踩'}，赞和踩只能选一个`);
                    return (
                      <span className="row gap-4">
                        <Button size="xs" variant="ghost" onClick={() => openSimilar(it)}>相似</Button>
                        <Button size="xs" icon variant={mine === 'like' ? 'primary' : 'ghost'} disabled={mine === 'dislike'} aria-pressed={mine === 'like'} style={mine === 'like' ? { cursor: 'default' } : undefined}
                          title={tip('like')} aria-label="喜欢" onClick={() => vote(it, 'like')}><ThumbsUp size={12} /></Button>
                        <Button size="xs" icon variant={mine === 'dislike' ? 'danger' : 'ghost'} disabled={mine === 'like'} aria-pressed={mine === 'dislike'} style={mine === 'dislike' ? { cursor: 'default' } : undefined}
                          title={tip('dislike')} aria-label="不喜欢" onClick={() => vote(it, 'dislike')}><ThumbsDown size={12} /></Button>
                      </span>
                    );
                  })()}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="col gap-12">
          <div className="card">
            <div className="card-header"><h3><MessageCircleQuestion size={14} />模拟一次提问</h3></div>
            <div className="card-body col gap-8">
              <textarea rows={2} value={ask} onChange={(e) => setAsk(e.target.value)} />
              <Button size="sm" variant="primary" onClick={simulateAsk}>记录提问并刷新推荐</Button>
              <div className="xs faint">智能问答里的每次提问、回答引用到的章节，都会自动记为交互。</div>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><h3><TrendingUp size={14} />兴趣画像</h3></div>
            <div className="card-body col gap-8">
              {res?.profile ? (
                <>
                  <div className="row gap-6 wrap">{Object.entries(res.profile.byKind).map(([k, n]) => <Badge key={k}>{EVENT_LABEL[k] ?? k} {n}</Badge>)}</div>
                  <div className="col gap-4">
                    {res.profile.recent.map((r, i) => (
                      <div key={i} className="row between gap-8 xs"><span className="ellipsis"><b>{EVENT_LABEL[r.kind] ?? r.kind}</b> {r.text || r.itemId.split(':').slice(-1)[0]}</span><span className="faint">{fmtTime(r.ts).slice(5, 16)}</span></div>
                    ))}
                  </div>
                </>
              ) : <div className="xs faint">还没有交互记录：先按热门推荐，提问 / 点赞后变成个性化推荐。</div>}
            </div>
          </div>
          <Callout>兴趣向量 = 近期交互条目向量的时间衰减加权和（半衰期 14 天，点踩扣分），再叠加长期记忆里的偏好；召回后按热度加分、用 MMR 去同质化。</Callout>
        </div>
      </div>

      {similar ? (
        <Drawer title={`与「${similar.item.title}」相似`} subtitle="LanceDB 条目向量近邻" onClose={() => setSimilar(null)}>
          <div className="col gap-8">
            {similar.rows.map((r) => (
              <div key={r.id} className="chunk">
                <div className="row between gap-8"><span className="row gap-6"><Badge tone={KIND[r.kind]?.tone}>{KIND[r.kind]?.label}</Badge><span className="small">{r.title}</span></span><span className="mono xs">{(r.similarity ?? 0).toFixed(3)}</span></div>
                <ScoreBar value={r.similarity ?? 0} />
                {r.text ? <div className="xs muted clamp-2 mt-4">{r.text}</div> : null}
              </div>
            ))}
          </div>
        </Drawer>
      ) : null}
    </div>
  );
}
