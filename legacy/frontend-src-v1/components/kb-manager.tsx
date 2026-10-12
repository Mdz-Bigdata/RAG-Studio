import React, { useState } from 'react';
import { Database, Plus, Trash2, FileText, Upload, AlertCircle, Edit, Check, X, RefreshCw } from 'lucide-react';
import type { KnowledgeBase, Document } from '../hooks/use-kb';

interface KBManagerProps {
  kbs: KnowledgeBase[];
  currentKb: KnowledgeBase | null;
  setCurrentKb: (kb: KnowledgeBase) => void;
  documents: Document[];
  createKb: (name: string, description: string) => Promise<any>;
  deleteKb: (kbId: string) => Promise<boolean>;
  uploadDocument: (kbId: string, file: File) => Promise<any>;
  deleteDocument: (docId: string) => Promise<boolean>;
  fetchMarkdown: (docId: string) => Promise<string>;
  updateMarkdown: (docId: string, markdown: string) => Promise<any>;
}

export function KBManager({
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
}: KBManagerProps) {
  const [newKbName, setNewKbName] = useState('');
  const [newKbDesc, setNewKbDesc] = useState('');
  const [showAddKb, setShowAddKb] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [editDoc, setEditDoc] = useState<Document | null>(null);
  const [editMarkdown, setEditMarkdown] = useState('');
  const [isSavingMarkdown, setIsSavingMarkdown] = useState(false);

  // 处理创建 KB
  const handleCreateKb = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKbName.trim()) return;
    await createKb(newKbName, newKbDesc);
    setNewKbName('');
    setNewKbDesc('');
    setShowAddKb(false);
  };

  // 处理上传文档
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !currentKb) return;
    setIsUploading(true);
    try {
      await uploadDocument(currentKb.id, files[0]);
    } finally {
      setIsUploading(false);
      e.target.value = ''; // 重设
    }
  };

  // 格式化文件大小
  const formatBytes = (bytes: number | null) => {
    if (bytes === null) return '未知大小';
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 打开 Markdown 编辑器
  const handleOpenEditor = async (doc: Document) => {
    setEditDoc(doc);
    const md = await fetchMarkdown(doc.id);
    setEditMarkdown(md);
  };

  // 保存手工修改的 Markdown 并重新入库
  const handleSaveMarkdown = async () => {
    if (!editDoc) return;
    setIsSavingMarkdown(true);
    try {
      await updateMarkdown(editDoc.id, editMarkdown);
      setEditDoc(null);
      setEditMarkdown('');
    } finally {
      setIsSavingMarkdown(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px', height: '100%' }} className="animate-fade-in">
      {/* 左侧：知识库列表 */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '16px', fontWeight: 600 }}>
            <Database size={18} className="gradient-text" style={{ color: 'var(--brand-blue)' }} />
            知识库列表
          </h3>
          <button 
            onClick={() => setShowAddKb(!showAddKb)}
            style={{ 
              background: 'rgba(0, 210, 255, 0.1)', 
              color: 'var(--brand-blue)', 
              borderRadius: '6px', 
              padding: '6px 8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <Plus size={16} />
          </button>
        </div>

        {showAddKb && (
          <form onSubmit={handleCreateKb} className="glass-panel animate-fade-in" style={{ padding: '12px', background: 'rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <input 
              type="text" 
              placeholder="知识库名称" 
              value={newKbName}
              onChange={(e) => setNewKbName(e.target.value)}
              required
              style={{ width: '100%' }}
            />
            <textarea 
              placeholder="知识库描述" 
              value={newKbDesc}
              onChange={(e) => setNewKbDesc(e.target.value)}
              rows={2}
              style={{ width: '100%', resize: 'none' }}
            />
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
              <button type="button" className="secondary" onClick={() => setShowAddKb(false)} style={{ padding: '6px 12px', fontSize: '12px' }}>
                取消
              </button>
              <button type="submit" className="primary" style={{ padding: '6px 12px', fontSize: '12px' }}>
                创建
              </button>
            </div>
          </form>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto', flex: 1 }}>
          {kbs.map((kb) => {
            const isActive = currentKb?.id === kb.id;
            return (
              <div 
                key={kb.id} 
                onClick={() => setCurrentKb(kb)}
                style={{ 
                  padding: '12px', 
                  borderRadius: '10px', 
                  cursor: 'pointer',
                  background: isActive ? 'var(--bg-glass-active)' : 'rgba(255, 255, 255, 0.02)',
                  border: isActive ? '1px solid var(--border-active)' : '1px solid transparent',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{ overflow: 'hidden', flex: 1, marginRight: '10px' }}>
                  <div style={{ fontWeight: 500, fontSize: '14px', color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                    {kb.name}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', marginTop: '2px' }}>
                    {kb.description || '暂无描述'}
                  </div>
                </div>
                <Trash2 
                  size={14} 
                  className="trash-icon"
                  style={{ color: 'var(--text-muted)', cursor: 'pointer', transition: 'color 0.2s' }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                  onClick={async (e) => {
                    e.stopPropagation();
                    if(confirm(`确定删除知识库“${kb.name}”吗？这将级联删除其下所有文档和向量。`)) {
                      await deleteKb(kb.id);
                    }
                  }}
                />
              </div>
            );
          })}
          {kbs.length === 0 && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px', marginTop: '40px' }}>
              暂无知识库，请先创建一个。
            </div>
          )}
        </div>
      </div>

      {/* 右侧：当前知识库文档列表与解析管理 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {currentKb ? (
          <>
            {/* 上传区域 */}
            <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', position: 'relative' }}>
              <input 
                type="file" 
                id="file-upload"
                accept=".pdf"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                disabled={isUploading}
              />
              <label 
                htmlFor="file-upload"
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  gap: '12px', 
                  cursor: isUploading ? 'not-allowed' : 'pointer'
                }}
              >
                <div style={{ background: 'var(--brand-gradient)', borderRadius: '50%', width: '54px', height: '54px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(0, 210, 255, 0.2)' }}>
                  <Upload size={22} color="white" />
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '16px' }}>
                    {isUploading ? '正在上传与提交任务...' : '点击上传 PDF 知识库源文档'}
                  </div>
                  <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    系统将自动触发 MinerU (magic-pdf) 排版分析器或 Built-in 高还原文本表格解析器进行解析
                  </div>
                </div>
              </label>
            </div>

            {/* 文档列表 */}
            <div className="glass-panel" style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={16} style={{ color: 'var(--brand-blue)' }} />
                数据源文档
              </h3>
              
              <div style={{ overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {documents.map((doc) => (
                  <div 
                    key={doc.id}
                    className="glass-panel"
                    style={{ 
                      padding: '14px 18px', 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      background: 'rgba(255, 255, 255, 0.01)',
                      border: '1px solid rgba(255, 255, 255, 0.04)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flex: 1, minWidth: 0 }}>
                      <div style={{ position: 'relative' }}>
                        <FileText size={28} style={{ color: doc.status === 'FAILED' ? '#ef4444' : 'var(--brand-blue)' }} />
                        <div 
                          style={{ position: 'absolute', bottom: '-2px', right: '-2px', border: '2px solid var(--bg-main)' }}
                          className={`pulse-dot ${doc.status.toLowerCase()}`}
                        />
                      </div>
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontWeight: 500, fontSize: '14px', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={doc.fileName}>
                          {doc.fileName}
                        </div>
                        <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                          <span>大小: {formatBytes(doc.fileSize)}</span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            状态: {
                              doc.status === 'PENDING' ? '排队中' :
                              doc.status === 'PROCESSING' ? '正在执行 MinerU 高精度分析...' :
                              doc.status === 'COMPLETED' ? '解析并索引完成' : '解析失败'
                            }
                          </span>
                        </div>
                        {doc.errorMessage && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: '#ef4444', marginTop: '6px' }}>
                            <AlertCircle size={12} />
                            {doc.errorMessage}
                          </div>
                        )}
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', marginLeft: '12px' }}>
                      {doc.status === 'COMPLETED' && (
                        <button 
                          className="secondary" 
                          onClick={() => handleOpenEditor(doc)}
                          style={{ padding: '6px 12px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
                        >
                          <Edit size={12} />
                          校对
                        </button>
                      )}
                      <button 
                        className="secondary" 
                        onClick={async () => {
                          if (confirm(`确定删除文档“${doc.fileName}”吗？这将从向量库和实体库中清空对应的所有数据块。`)) {
                            await deleteDocument(doc.id);
                          }
                        }}
                        style={{ padding: '6px 8px', borderRadius: '8px' }}
                      >
                        <Trash2 size={13} style={{ color: 'var(--text-secondary)' }} />
                      </button>
                    </div>
                  </div>
                ))}

                {documents.length === 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '200px', color: 'var(--text-muted)' }}>
                    <FileText size={48} style={{ opacity: 0.1, marginBottom: '12px' }} />
                    <span style={{ fontSize: '13px' }}>当前知识库空空如也，请在上方上传一份 PDF。</span>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            <Database size={64} style={{ opacity: 0.1, marginBottom: '16px' }} />
            <h3 style={{ fontSize: '16px', fontWeight: 500 }}>请在左侧选择或创建一个知识库</h3>
            <p style={{ fontSize: '13px', marginTop: '6px' }}>企业级 Hybrid RAG 需要先关联知识库来承载切片和图谱实体。</p>
          </div>
        )}
      </div>

      {/* 隐藏的 Markdown 侧拉编辑窗 (Human-in-the-loop 人机协同校对) */}
      {editDoc && (
        <div style={{
          position: 'fixed', top: 0, right: 0, bottom: 0, left: 0,
          background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
          zIndex: 1000, display: 'flex', justifyContent: 'flex-end'
        }} className="animate-fade-in">
          <div className="glass-panel" style={{
            width: '800px', height: '100%', borderRadius: '16px 0 0 16px',
            borderRight: 'none', background: '#090d16', padding: '24px',
            display: 'flex', flexDirection: 'column', gap: '16px',
            boxShadow: '-10px 0 30px rgba(0,0,0,0.5)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 600 }}>排版校对及精准重新切片</h3>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>文档: {editDoc.fileName} (编辑提取出的结构化 Markdown)</span>
              </div>
              <button className="secondary" onClick={() => setEditDoc(null)} style={{ padding: '6px' }}>
                <X size={18} />
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', fontSize: '12px', color: 'rgba(0, 210, 255, 0.8)', background: 'rgba(0, 210, 255, 0.05)', padding: '10px 14px', borderRadius: '8px', border: '1px solid rgba(0, 210, 255, 0.1)' }}>
              <AlertCircle size={14} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>数据质量治理</strong>：大模型提取排版可能存在极少数表格列错位或专有名词漏识别。在此处手动优化 Markdown 排版和加粗实体（如 <strong>**专有名词**</strong>），保存后系统将<strong>级联清理原数据并对新版本重新生成向量和 UnWeaver 实体关系</strong>。
              </div>
            </div>

            <textarea 
              value={editMarkdown}
              onChange={(e) => setEditMarkdown(e.target.value)}
              style={{
                flex: 1, width: '100%', fontFamily: 'JetBrains Mono, monospace',
                fontSize: '13px', background: 'rgba(0,0,0,0.4)', padding: '16px',
                lineHeight: 1.6, resize: 'none'
              }}
            />

            <div style={{ display: 'flex', justifySelf: 'flex-end', gap: '12px' }}>
              <button className="secondary" onClick={() => setEditDoc(null)}>
                取消
              </button>
              <button 
                className="primary" 
                onClick={handleSaveMarkdown}
                disabled={isSavingMarkdown}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                {isSavingMarkdown ? (
                  <>
                    <RefreshCw size={14} style={{ animation: 'spin 1.5s linear infinite' }} />
                    正在重新索引入库...
                  </>
                ) : (
                  <>
                    <Check size={14} />
                    保存并重新索引
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
