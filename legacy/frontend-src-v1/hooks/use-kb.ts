import { useState, useEffect, useRef } from 'react';

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface Document {
  id: string;
  knowledgeBaseId: string;
  fileName: string;
  fileSize: number | null;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

const API_BASE = 'http://localhost:8000/api/v1';

export function useKb() {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [currentKb, setCurrentKb] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const pollingTimer = useRef<Record<string, number | NodeJS.Timeout>>({});

  // 获取知识库列表
  const fetchKbs = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/knowledge-bases/`);
      if (res.ok) {
        const data = await res.json();
        setKbs(data);
        if (data.length > 0 && !currentKb) {
          setCurrentKb(data[0]);
        }
      }
    } catch (e) {
      console.error('Failed to fetch knowledge bases', e);
    } finally {
      setIsLoading(false);
    }
  };

  // 创建知识库
  const createKb = async (name: string, description: string) => {
    try {
      const res = await fetch(`${API_BASE}/knowledge-bases/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description }),
      });
      if (res.ok) {
        const newKb = await res.json();
        setKbs((prev) => [...prev, newKb]);
        setCurrentKb(newKb);
        return newKb;
      }
    } catch (e) {
      console.error('Failed to create knowledge base', e);
    }
    return null;
  };

  // 删除知识库
  const deleteKb = async (kbId: string) => {
    try {
      const res = await fetch(`${API_BASE}/knowledge-bases/${kbId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setKbs((prev) => prev.filter((k) => k.id !== kbId));
        if (currentKb?.id === kbId) {
          const remaining = kbs.filter((k) => k.id !== kbId);
          setCurrentKb(remaining.length > 0 ? remaining[0] : null);
        }
        return true;
      }
    } catch (e) {
      console.error('Failed to delete knowledge base', e);
    }
    return false;
  };

  // 获取当前知识库下的所有文档
  const fetchDocuments = async (kbId: string) => {
    try {
      const res = await fetch(`${API_BASE}/documents/?kbId=${kbId}`);
      if (res.ok) {
        const data: Document[] = await res.json();
        setDocuments(data);
        
        // 检测是否有正在解析的文档，启动轮询
        data.forEach((doc) => {
          if ((doc.status === 'PENDING' || doc.status === 'PROCESSING') && !pollingTimer.current[doc.id]) {
            startPolling(doc.id, kbId);
          }
        });
      }
    } catch (e) {
      console.error('Failed to fetch documents', e);
    }
  };

  // 轮询单个文档状态
  const startPolling = (docId: string, kbId: string) => {
    if (pollingTimer.current[docId]) return;

    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/documents/${docId}`);
        if (res.ok) {
          const doc: Document = await res.json();
          // 更新文档列表状态
          setDocuments((prev) => 
            prev.map((d) => (d.id === docId ? doc : d))
          );

          if (doc.status === 'COMPLETED' || doc.status === 'FAILED') {
            clearInterval(timer);
            delete pollingTimer.current[docId];
          }
        } else {
          clearInterval(timer);
          delete pollingTimer.current[docId];
        }
      } catch (e) {
        console.error(`Polling doc ${docId} error`, e);
        clearInterval(timer);
        delete pollingTimer.current[docId];
      }
    }, 2000);

    pollingTimer.current[docId] = timer;
  };

  // 上传文档
  const uploadDocument = async (kbId: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_BASE}/documents/upload?kbId=${kbId}`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const newDoc: Document = await res.json();
        setDocuments((prev) => [newDoc, ...prev]);
        // 自动启动状态轮询
        startPolling(newDoc.id, kbId);
        return newDoc;
      }
    } catch (e) {
      console.error('Failed to upload document', e);
    }
    return null;
  };

  // 删除文档
  const deleteDocument = async (docId: string) => {
    // 如果有活跃的轮询，清理之
    if (pollingTimer.current[docId]) {
      clearInterval(pollingTimer.current[docId]);
      delete pollingTimer.current[docId];
    }

    try {
      const res = await fetch(`${API_BASE}/documents/${docId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setDocuments((prev) => prev.filter((d) => d.id !== docId));
        return true;
      }
    } catch (e) {
      console.error('Failed to delete document', e);
    }
    return false;
  };

  // 获取解析所得的 Markdown 文本
  const fetchMarkdown = async (docId: string) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}/markdown`);
      if (res.ok) {
        const data = await res.json();
        return data.markdownContent as string;
      }
    } catch (e) {
      console.error('Failed to fetch markdown content', e);
    }
    return '';
  };

  // 用户修改并提交 Markdown 文本重新索引入库
  const updateMarkdown = async (docId: string, markdown: string) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}/markdown`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ markdownContent: markdown }),
      });
      if (res.ok) {
        const updatedDoc = await res.json();
        setDocuments((prev) => 
          prev.map((d) => (d.id === docId ? updatedDoc : d))
        );
        return updatedDoc as Document;
      }
    } catch (e) {
      console.error('Failed to update document markdown', e);
    }
    return null;
  };

  // 卸载组件时清理所有轮询计时器
  useEffect(() => {
    return () => {
      Object.values(pollingTimer.current).forEach((timer) => clearInterval(timer));
    };
  }, []);

  // 切换知识库时自动加载文档列表
  useEffect(() => {
    if (currentKb) {
      fetchDocuments(currentKb.id);
    } else {
      setDocuments([]);
    }
  }, [currentKb]);

  return {
    kbs,
    currentKb,
    setCurrentKb,
    documents,
    isLoading,
    fetchKbs,
    createKb,
    deleteKb,
    uploadDocument,
    deleteDocument,
    fetchMarkdown,
    updateMarkdown,
    refreshDocuments: () => currentKb && fetchDocuments(currentKb.id),
  };
}
