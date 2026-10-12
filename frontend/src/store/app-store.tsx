// 全局状态：知识库列表 / 当前知识库 / 系统状态 / Toast。
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import type { KnowledgeBase, SystemStatus, UserIdentity } from '../api/types';

export type ToastTone = 'ok' | 'bad' | 'warn' | '';
interface Toast { id: number; tone: ToastTone; text: string }

interface AppStore {
  kbs: KnowledgeBase[];
  currentKb: KnowledgeBase | null;
  setCurrentKbId: (id: string | null) => void;
  refreshKbs: () => Promise<KnowledgeBase[]>;
  status: SystemStatus | null;
  backendOnline: boolean;
  refreshStatus: () => Promise<void>;
  toast: (text: string, tone?: ToastTone) => void;
  toasts: Toast[];
  dismissToast: (id: number) => void;
  identity: UserIdentity;
  setIdentity: (u: UserIdentity) => void;
}

// 权限级别：文档侧"最低可见级别"与用户侧"身份级别"共用一套刻度
export const ACCESS_LEVELS = [
  { value: 0, doc: '公开', user: '访客（仅公开）' },
  { value: 1, doc: '内部', user: '员工（公开 + 内部）' },
  { value: 2, doc: '机密', user: '主管（含机密）' },
  { value: 3, doc: '绝密', user: '管理员（全部可见）' },
];
const LS_IDENTITY = 'rag.identity';

const Ctx = createContext<AppStore | null>(null);
const LS_KB = 'rag.currentKb';

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [currentKbId, setCurrentKbIdState] = useState<string | null>(() => {
    try { return localStorage.getItem(LS_KB); } catch { return null; }
  });
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(1);

  const toast = useCallback((text: string, tone: ToastTone = '') => {
    const id = idRef.current++;
    setToasts((t) => [...t, { id, tone, text }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), tone === 'bad' ? 6000 : 3500);
  }, []);
  const dismissToast = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const setCurrentKbId = useCallback((id: string | null) => {
    setCurrentKbIdState(id);
    try { id ? localStorage.setItem(LS_KB, id) : localStorage.removeItem(LS_KB); } catch { /* ignore */ }
  }, []);

  const refreshKbs = useCallback(async () => {
    const list = await api.kb.list();
    setKbs(list);
    setCurrentKbIdState((cur) => {
      const next = cur && list.some((k) => k.id === cur) ? cur : list[0]?.id ?? null;
      try { next ? localStorage.setItem(LS_KB, next) : localStorage.removeItem(LS_KB); } catch { /* ignore */ }
      return next;
    });
    return list;
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await api.system.status();
      setStatus(s);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    refreshKbs().catch(() => undefined);
    const t = window.setInterval(refreshStatus, 15000);
    return () => window.clearInterval(t);
  }, [refreshKbs, refreshStatus]);

  const currentKb = useMemo(() => kbs.find((k) => k.id === currentKbId) ?? null, [kbs, currentKbId]);
  const [identity, setIdentityState] = useState<UserIdentity>(() => {
    try { return { level: 3, department: null, ...JSON.parse(localStorage.getItem(LS_IDENTITY) || '{}') }; } catch { return { level: 3, department: null }; }
  });
  const setIdentity = useCallback((u: UserIdentity) => {
    setIdentityState(u);
    try { localStorage.setItem(LS_IDENTITY, JSON.stringify(u)); } catch { /* ignore */ }
  }, []);
  const value = useMemo<AppStore>(() => ({ kbs, currentKb, setCurrentKbId, refreshKbs, status, backendOnline, refreshStatus, toast, toasts, dismissToast, identity, setIdentity }), [kbs, currentKb, setCurrentKbId, refreshKbs, status, backendOnline, refreshStatus, toast, toasts, dismissToast, identity, setIdentity]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppStore {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp must be used within AppProvider');
  return v;
}
