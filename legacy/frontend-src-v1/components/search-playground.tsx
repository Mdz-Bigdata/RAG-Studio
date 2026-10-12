import React, { useState } from 'react';
import { Search, Zap, Cpu, Layers, Network, ChevronDown, ChevronUp, Info, HelpCircle } from 'lucide-react';
import type { PlaygroundData, Relation } from '../hooks/use-chat';
import type { KnowledgeBase } from '../hooks/use-kb';

interface SearchPlaygroundProps {
  currentKb: KnowledgeBase | null;
  playgroundData: PlaygroundData | null;
  isSearching: boolean;
  searchPlayground: (kbId: string, query: string, topK?: number) => Promise<any>;
}

export function SearchPlayground({
  currentKb,
  playgroundData,
  isSearching,
  searchPlayground,
}: SearchPlaygroundProps) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [showHydeDoc, setShowHydeDoc] = useState(false);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !currentKb) return;
    searchPlayground(currentKb.id, query, topK);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }} className="animate-fade-in">
      {/* 顶部搜索输入栏 */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ flex: 1, position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', left: '14px', top: '13px', color: 'var(--text-muted)' }} />
            <input 
              type="text" 
              placeholder={currentKb ? `在知识库“${currentKb.name}”中测试 Hybrid 检索多路召回与融合流程...` : '请先选择知识库'}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={!currentKb || isSearching}
              style={{ width: '100%', paddingLeft: '44px', fontSize: '15px' }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>召回数 (Top K):</span>
            <select 
              value={topK} 
              onChange={(e) => setTopK(Number(e.target.value))}
              disabled={!currentKb || isSearching}
              style={{ width: '70px', padding: '9px 12px' }}
            >
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={8}>8</option>
              <option value={10}>10</option>
            </select>
          </div>
          <button 
            type="submit" 
            className="primary" 
            disabled={!currentKb || isSearching || !query.trim()}
            style={{ padding: '10px 24px', display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Zap size={16} />
            {isSearching ? '多路召回中...' : '融合检索'}
          </button>
        </form>
      </div>

      {!currentKb ? (
        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
          <Search size={48} style={{ opacity: 0.1, marginBottom: '12px' }} />
          <span>请先在“知识库管理”中创建或选择知识库。</span>
        </div>
      ) : !playgroundData ? (
        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', padding: '40px' }}>
          <Info size={40} style={{ color: 'var(--brand-blue)', opacity: 0.4, marginBottom: '14px' }} />
          <h3 style={{ fontSize: '16px', fontWeight: 500, color: 'var(--text-primary)' }}>多路召回游乐场</h3>
          <p style={{ fontSize: '13px', textAlign: 'center', maxWidth: '500px', marginTop: '6px', lineHeight: 1.5 }}>
            输入 Query 并点击“融合检索”后，系统将在后台并行执行<strong>向量召回、BM25 关键词召回、HyDE 生成式召回</strong>，并将其渲染在此处，供您直观调试 RRF 和 Rerank 精排。
          </p>
        </div>
      ) : (
        /* 主体调试面板 */
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
          {/* HyDE 假设文档浮窗 */}
          {playgroundData.hydeDocument && (
            <div className="glass-panel" style={{ padding: '14px 18px', background: 'rgba(168, 85, 247, 0.03)', border: '1px solid rgba(168, 85, 247, 0.15)' }}>
              <div 
                onClick={() => setShowHydeDoc(!showHydeDoc)}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 600, color: 'var(--color-hyde)' }}>
                  <Cpu size={14} />
                  HyDE (Hypothetical Document Embeddings) 假设回答
                </div>
                <div style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px' }}>
                  {showHydeDoc ? <><ChevronUp size={14} /> 收起假设文本</> : <><ChevronDown size={14} /> 展开查看模拟答案</>}
                </div>
              </div>
              {showHydeDoc && (
                <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '10px', lineHeight: 1.6, padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', fontFamily: 'inherit' }} className="animate-fade-in">
                  {playgroundData.hydeDocument}
                </div>
              )}
            </div>
          )}

          {/* 5 列滚动对比图谱 */}
          <div style={{ 
            flex: 1, 
            display: 'grid', 
            gridTemplateColumns: 'repeat(5, 1fr)', 
            gap: '12px', 
            overflowX: 'auto', 
            overflowY: 'hidden',
            paddingBottom: '8px'
          }}>
            {/* Column 1: Dense */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--color-dense)', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
                <Layers size={14} />
                1. 密集向量召回 (Dense)
              </div>
              <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {playgroundData.denseResults.map((item, idx) => (
                  <div key={item.id} style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.04)', borderRadius: '8px', padding: '8px 10px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', marginBottom: '4px', fontSize: '11px' }}>
                      <span># {idx + 1} (切片 {item.chunkIndex})</span>
                      <span style={{ color: 'var(--color-dense)' }}>Cosine: {item.score.toFixed(4)}</span>
                    </div>
                    <div style={{ color: 'var(--text-primary)', lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical' }}>
                      {item.content}
                    </div>
                  </div>
                ))}
                {playgroundData.denseResults.length === 0 && <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '30px' }}>无结果</div>}
              </div>
            </div>

            {/* Column 2: BM25 */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--color-bm25)', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
                <Layers size={14} />
                2. 稀疏 BM25 召回
              </div>
              <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {playgroundData.bm25Results.map((item, idx) => (
                  <div key={item.id} style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.04)', borderRadius: '8px', padding: '8px 10px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', marginBottom: '4px', fontSize: '11px' }}>
                      <span># {idx + 1} (切片 {item.chunkIndex})</span>
                      <span style={{ color: 'var(--color-bm25)' }}>Score: {item.score.toFixed(2)}</span>
                    </div>
                    <div style={{ color: 'var(--text-primary)', lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical' }}>
                      {item.content}
                    </div>
                  </div>
                ))}
                {playgroundData.bm25Results.length === 0 && <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '30px' }}>无结果</div>}
              </div>
            </div>

            {/* Column 3: HyDE */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--color-hyde)', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
                <Cpu size={14} />
                3. HyDE 向量召回
              </div>
              <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {playgroundData.hydeResults.map((item, idx) => (
                  <div key={item.id} style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.04)', borderRadius: '8px', padding: '8px 10px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', marginBottom: '4px', fontSize: '11px' }}>
                      <span># {idx + 1} (切片 {item.chunkIndex})</span>
                      <span style={{ color: 'var(--color-hyde)' }}>Cosine: {item.score.toFixed(4)}</span>
                    </div>
                    <div style={{ color: 'var(--text-primary)', lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical' }}>
                      {item.content}
                    </div>
                  </div>
                ))}
                {playgroundData.hydeResults.length === 0 && <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '30px' }}>无结果</div>}
              </div>
            </div>

            {/* Column 4: RRF & Rerank */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '12px', background: 'rgba(16, 185, 129, 0.02)', border: '1px solid rgba(16, 185, 129, 0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--color-rrf)', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
                <Layers size={14} />
                4. RRF 融合与重排 (Rerank)
              </div>
              <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {playgroundData.rerankedResults.map((item, idx) => (
                  <div key={item.id} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', padding: '8px 10px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', marginBottom: '4px', fontSize: '10px' }}>
                      <span>精排 #{idx + 1} (切片 {item.chunkIndex})</span>
                      <span style={{ color: 'var(--color-rrf)', fontWeight: 600 }}>Score: {item.score.toFixed(1)}</span>
                    </div>
                    <div style={{ color: 'var(--text-primary)', lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', marginBottom: '6px' }}>
                      {item.content}
                    </div>
                    <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                      {item.sources?.map((src: string) => {
                        const srcColor = src === 'dense' ? 'var(--color-dense)' : src === 'bm25' ? 'var(--color-bm25)' : 'var(--color-hyde)';
                        return (
                          <span key={src} style={{ fontSize: '9px', padding: '2px 4px', borderRadius: '4px', border: `1px solid ${srcColor}`, color: srcColor }}>
                            {src.toUpperCase()} 召回
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ))}
                {playgroundData.rerankedResults.length === 0 && <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '30px' }}>无结果</div>}
              </div>
            </div>

            {/* Column 5: UnWeaver */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '12px', background: 'rgba(236, 72, 153, 0.02)', border: '1px solid rgba(236, 72, 153, 0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--color-weaved)', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', marginBottom: '10px' }}>
                <Network size={14} />
                5. UnWeaver 实体织网
              </div>
              <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {playgroundData.weavedRelations.map((item, idx) => (
                  <div key={idx} style={{ background: 'rgba(236, 72, 153, 0.05)', border: '1px solid rgba(236, 72, 153, 0.15)', borderRadius: '8px', padding: '10px', fontSize: '12px' }}>
                    <div style={{ fontWeight: 600, color: 'var(--color-weaved)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Network size={12} />
                      实体: {item.entity}
                    </div>
                    <div style={{ color: 'var(--text-primary)', fontSize: '11px', lineHeight: 1.5, background: 'rgba(0,0,0,0.2)', padding: '6px', borderRadius: '4px', margin: '4px 0' }}>
                      {item.definition}
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '2px', marginTop: '6px' }}>
                      <span>织网来源: 库内定义 Chunk</span>
                      <span>检索块触发包含: “{item.weavedFrom}”</span>
                    </div>
                  </div>
                ))}
                {playgroundData.weavedRelations.length === 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', padding: '10px', textAlign: 'center' }}>
                    <HelpCircle size={28} style={{ opacity: 0.15, marginBottom: '6px' }} />
                    <span style={{ fontSize: '11px' }}>本轮未触发实体编织。上传 PDF 中含有 `**术语**是...` 格式的段落时会自动构建编织索引。</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
