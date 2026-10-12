import { Database } from 'lucide-react';
import { useApp } from '../store/app-store';

export function KbSelect({ allowAuto, value, onChange }: { allowAuto?: boolean; value?: string | null; onChange?: (id: string) => void }) {
  const { kbs, currentKb, setCurrentKbId } = useApp();
  const v = value === undefined ? currentKb?.id ?? '' : value ?? '';
  return (
    <div className="row gap-6" style={{ minWidth: 200 }}>
      <Database size={14} className="faint" />
      <select value={v} onChange={(e) => (onChange ? onChange(e.target.value) : setCurrentKbId(e.target.value || null))} style={{ padding: '6px 30px 6px 10px', fontSize: 12.5 }}>
        {allowAuto ? <option value="auto">自动路由（所有知识库）</option> : null}
        {kbs.length === 0 ? <option value="">暂无知识库</option> : null}
        {kbs.map((kb) => (
          <option key={kb.id} value={kb.id}>{kb.name}（{kb.documentCount} 文档）</option>
        ))}
      </select>
    </div>
  );
}
