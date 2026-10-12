// 向量检索调优的效果对比：同一批查询走精确检索与所选的降维 / 量化配置，比较召回率、延迟与内存。
import { useState } from 'react';
import { Gauge } from 'lucide-react';
import { api } from '../api/client';
import type { VectorBenchmark } from '../api/types';
import { useApp } from '../store/app-store';
import { Badge, Button } from './ui';

const fmtBytes = (n?: number | null) => (n == null ? '—' : n > 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${(n / 1024).toFixed(1)} KB`);

export function VectorBenchmarkCard({ pcaDim, quantize, rescore, oversample }: { pcaDim: number | null; quantize: string; rescore: boolean; oversample: number }) {
  const { currentKb, toast } = useApp();
  const [res, setRes] = useState<VectorBenchmark | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    if (!currentKb) return;
    setLoading(true);
    try {
      setRes(await api.retrieval.vectorBenchmark(currentKb.id, { pcaDim, quantize: quantize as 'none' | 'int8', rescore, oversample }));
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), 'bad');
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="chunk col gap-8" style={{ margin: '8px 0' }}>
      <div className="row between wrap gap-8">
        <span className="small strong row gap-6"><Gauge size={13} />效果对比（{currentKb ? currentKb.name : '未选择知识库'}）</span>
        <Button size="xs" loading={loading} disabled={!currentKb} onClick={run}>用上面的参数测一下</Button>
      </div>
      <div className="xs faint">从知识库正文抽取查询，分别走精确检索与上面的配置（含未保存的修改），比较 top-10 召回重合率、单次延迟与粗筛矩阵内存。不会改动已保存的配置。</div>
      {res && !res.ok ? <div className="xs text-warning">{res.message}</div> : null}
      {res && res.ok ? (
        <div className="col gap-6">
          {[
            { label: '精确检索', dim: res.dim, bytes: res.fullBytes, ms: res.exactMs, recall: null as number | null },
            { label: '当前配置', dim: res.coarseDim ?? res.dim, bytes: res.coarseBytes, ms: res.tunedMs, recall: res.recall },
          ].map((r) => (
            <div key={r.label} className="row between wrap gap-6 xs" style={{ padding: '6px 8px', borderRadius: 6, background: 'var(--bg-1)' }}>
              <span className="strong row gap-6">{r.label}{r.recall !== null && !res.tuned ? <Badge>未启用调优</Badge> : null}</span>
              <span className="row wrap gap-8 mono">
                <span>{r.dim} 维</span>
                <span>{fmtBytes(r.bytes)}</span>
                <span>{r.ms} ms</span>
                <span className={r.recall === null ? 'faint' : r.recall >= 0.95 ? 'text-success' : 'text-warning'}>召回@{res.topK} {(r.recall ?? 1).toFixed(3)}</span>
              </span>
            </div>
          ))}
          <div className="xs faint">{res.vectors} 个向量 · {res.queries} 条查询 · 后端 {res.backend}。向量少于几万条时暴力检索本来就很快，降维 / 量化主要省内存；规模上去后延迟收益才明显。</div>
        </div>
      ) : null}
    </div>
  );
}
