// 上下文 / 引用来源卡片：编号、文件、面包屑、分数条、压缩前后切换、织入标记。
import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Network } from 'lucide-react';
import type { ContextView } from '../api/types';
import { Badge, ScoreBar } from './ui';
import { formatScore } from '../lib/format';

interface Props {
  ctx: ContextView;
  index?: number;
  highlighted?: boolean;
  defaultExpanded?: boolean;
  compact?: boolean;
  onClick?: () => void;
}

export function ContextCard({ ctx, index, highlighted, defaultExpanded = false, compact, onClick }: Props) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [showOriginal, setShowOriginal] = useState(false);
  const body = ctx.compressedText && !showOriginal ? ctx.compressedText : ctx.text;
  const pages = ctx.pageStart !== null && ctx.pageStart !== undefined ? (ctx.pageEnd !== null && ctx.pageEnd !== ctx.pageStart ? `p.${ctx.pageStart + 1}-${(ctx.pageEnd ?? ctx.pageStart) + 1}` : `p.${ctx.pageStart + 1}`) : null;
  return (
    <div className={`chunk ${highlighted ? 'hl' : ''}`} onClick={onClick} style={onClick ? { cursor: 'pointer' } : undefined} id={index !== undefined ? `ctx-${index}` : undefined}>
      <div className="chunk-head">
        <div className="row gap-6" style={{ minWidth: 0 }}>
          {index !== undefined ? <span className="cite" style={{ cursor: 'default' }}>{index}</span> : null}
          <FileText size={12} />
          <span className="ellipsis" title={ctx.fileName ?? ctx.docId} style={{ maxWidth: compact ? 140 : 220 }}>{ctx.fileName ?? ctx.docId}</span>
          {pages ? <span className="faint">{pages}</span> : null}
          {ctx.weaved ? <Badge tone="weaved"><Network size={10} />织入：{ctx.weavedEntity}</Badge> : null}
        </div>
        <div className="row gap-6">
          <span className="mono">{formatScore(ctx.score)}</span>
          <span className="faint">{ctx.tokenCount} tok</span>
        </div>
      </div>
      <div style={{ margin: '4px 0 6px' }}><ScoreBar value={ctx.score} color={ctx.weaved ? 'var(--c-weaved)' : undefined} /></div>
      {ctx.breadcrumb ? <div className="chunk-crumb ellipsis" title={ctx.breadcrumb}>{ctx.breadcrumb}</div> : null}
      <div className={`chunk-text pre ${expanded ? '' : compact ? 'clamp-3' : 'clamp-5'}`} style={{ marginTop: 4 }}>{body}</div>
      <div className="row between mt-8">
        <div className="row gap-6">
          {ctx.compressedText ? (
            <button className="btn ghost xs" onClick={(e) => { e.stopPropagation(); setShowOriginal((v) => !v); }}>
              {showOriginal ? '查看压缩后' : '查看原文'}
            </button>
          ) : null}
          {ctx.matchedChildIds?.length ? <span className="faint xs">命中 {ctx.matchedChildIds.length} 个子块</span> : null}
        </div>
        <button className="btn ghost xs" onClick={(e) => { e.stopPropagation(); setExpanded((v) => !v); }}>
          {expanded ? <><ChevronUp size={12} />收起</> : <><ChevronDown size={12} />展开</>}
        </button>
      </div>
    </div>
  );
}
