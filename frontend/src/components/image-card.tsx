// 图片检索结果：缩略图卡片（分数 / 各路排名 / 图注 / 来源）与详情弹窗（大图 + 描述 + OCR + 上下文）。
import { FileText, Link2, ScanSearch } from 'lucide-react';
import type { ImageHitView, ImageView } from '../api/types';
import { Badge, Button, Modal, ScoreBar } from './ui';
import { formatScore } from '../lib/format';

export const IMAGE_ROUTE: Record<string, { label: string; tone: string }> = {
  clip: { label: '图文向量', tone: 'dense' },
  clip_fused: { label: '图文早融合', tone: 'fused' },
  visual: { label: '视觉相似', tone: 'dense' },
  ocr_sparse: { label: '图中文字', tone: 'sparse' },
  desc_dense: { label: '描述·向量', tone: 'hyde' },
  desc_sparse: { label: '描述·BM25', tone: 'sparse' },
  'clip:image→chunk': { label: '图文直达', tone: 'dense' },
  'neighbor:image→image→chunk': { label: '相似图所在段落', tone: 'mmr' },
  'caption:dense': { label: '图转文·向量', tone: 'hyde' },
  'caption:sparse': { label: '图转文·BM25', tone: 'sparse' },
  lance_mm: { label: 'LanceDB·全文', tone: 'sparse' },
  'lance_mm:image→text': { label: 'LanceDB·图→文', tone: 'fused' },
};

export const isHit = (im: ImageView | ImageHitView): im is ImageHitView => 'score' in im;
const pageOf = (im: ImageView) => (im.page !== null && im.page !== undefined ? `p.${im.page + 1}` : null);
const titleOf = (im: ImageView) => im.caption || im.description || im.ocrText || im.contextText || im.fileName || '未命名图片';

export function ImageCard({ image, onOpen, highlighted, compact }: { image: ImageView | ImageHitView; onOpen?: () => void; highlighted?: boolean; compact?: boolean }) {
  const hit = isHit(image) ? image : null;
  return (
    <div className={`img-card ${highlighted ? 'hl' : ''} ${compact ? 'compact' : ''}`} onClick={onOpen} id={hit?.index ? `img-${hit.index}` : undefined} role={onOpen ? 'button' : undefined}>
      <div className="img-thumb">
        <img src={image.url} alt={titleOf(image)} loading="lazy" />
        {hit?.index ? <span className="cite img img-no">图{hit.index}</span> : hit && hit.rank ? <span className="rank img-no">#{hit.rank}</span> : null}
      </div>
      <div className="img-meta">
        <div className="small clamp-2" title={titleOf(image)}>{titleOf(image)}</div>
        <div className="xs faint ellipsis row gap-4" title={image.label}><FileText size={10} style={{ flexShrink: 0 }} /><span className="ellipsis">{image.fileName ?? image.docId}</span>{pageOf(image) ? <span>· {pageOf(image)}</span> : null}</div>
        {hit ? (
          <>
            <div className="row gap-6"><div className="flex-1"><ScoreBar value={hit.score} /></div><span className="mono xs">{formatScore(hit.score)}</span></div>
            {!compact ? (
              <div className="row gap-4 wrap">
                {hit.linked ? <Badge tone="weaved" title="图片所在段落被文本检索命中"><Link2 size={9} />图文关联</Badge> : null}
                {Object.entries(hit.routes).map(([k, v]) => <Badge key={k} mono tone={IMAGE_ROUTE[k]?.tone ?? ''} title={`${IMAGE_ROUTE[k]?.label ?? k}：第 ${v.rank} 名，原始分 ${v.score}`}>{IMAGE_ROUTE[k]?.label ?? k} #{v.rank}</Badge>)}
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

export function ImageDetail({ image, onClose, onSearchSimilar }: { image: ImageView | ImageHitView; onClose: () => void; onSearchSimilar?: (im: ImageView) => void }) {
  const hit = isHit(image) ? image : null;
  const Section = ({ label, text }: { label: string; text: string }) => (text ? <div><div className="xs faint mb-8">{label}</div><div className="small pre">{text}</div></div> : null);
  return (
    <Modal wide title={<span className="row gap-8">{hit?.index ? <span className="cite img">图{hit.index}</span> : null}<span className="ellipsis" style={{ maxWidth: 640 }}>{titleOf(image)}</span></span>} onClose={onClose}
      footer={<>{onSearchSimilar ? <Button onClick={() => { onSearchSimilar(image); onClose(); }}><ScanSearch size={13} />以此图搜相似</Button> : null}<a className="btn" href={image.url} target="_blank" rel="noreferrer">查看原图</a><Button variant="primary" onClick={onClose}>关闭</Button></>}>
      <div className="img-detail">
        <div className="img-detail-view"><img src={image.url} alt={titleOf(image)} /></div>
        <div className="col gap-12" style={{ minWidth: 0 }}>
          <div className="row gap-6 wrap">
            <Badge><FileText size={10} />{image.fileName ?? image.docId}</Badge>
            {pageOf(image) ? <Badge mono>{pageOf(image)}</Badge> : null}
            <Badge mono>{image.width}×{image.height}</Badge>
            <Badge mono>{image.origin}</Badge>
            {hit ? <Badge tone="info" mono>score {formatScore(hit.score)}</Badge> : null}
          </div>
          {image.breadcrumb ? <div className="chunk-crumb">{image.breadcrumb}</div> : null}
          {hit && Object.keys(hit.routes).length ? (
            <div><div className="xs faint mb-8">各路命中</div><div className="row gap-4 wrap">{Object.entries(hit.routes).map(([k, v]) => <Badge key={k} mono tone={IMAGE_ROUTE[k]?.tone ?? ''}>{IMAGE_ROUTE[k]?.label ?? k} · #{v.rank} · {formatScore(v.score, v.score > 1 ? 2 : 3)}</Badge>)}</div></div>
          ) : null}
          <Section label="图注" text={image.caption} />
          <Section label="内容描述（VLM）" text={image.description} />
          <Section label="图中文字（OCR）" text={image.ocrText} />
          <Section label="所在段落" text={image.contextText} />
          {!image.description && !image.ocrText ? <div className="xs faint">这张图没有 VLM 描述和 OCR 文字：检索只能依靠图注、上下文和图片向量。在「模型配置」激活多模态模型、或安装 rapidocr-onnxruntime 后重新解析文档即可补齐。</div> : null}
        </div>
      </div>
    </Modal>
  );
}
