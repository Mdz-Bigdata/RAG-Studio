// Markdown 渲染（GFM），并把 [n] 文本引用与 [图n] 图片引用渲染为可点击的引用胶囊。
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Props {
  text: string;
  onCite?: (n: number) => void;
  activeCite?: number | null;
  onImageCite?: (n: number) => void;
  className?: string;
}

const CITE_RE = /\[(\d{1,2})\](?!\()/g;
const IMAGE_CITE_RE = /\[图\s*(\d{1,2})\](?!\()/g;

export function Markdown({ text, onCite, activeCite, onImageCite, className = '' }: Props) {
  // 把 [3] 转成链接 [3](#cite-3)、[图2] 转成 [图2](#img-2)，再由 a 组件渲染成胶囊
  const prepared = text.replace(IMAGE_CITE_RE, (_m, n) => `[图${n}](#img-${n})`).replace(CITE_RE, (_m, n) => `[${n}](#cite-${n})`);
  return (
    <div className={`md ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // 资料原文里的图片标记多半指向本地路径或已剥离的内嵌图：渲染成文字占位，真正的图片在回答下方的 [图n] 卡片里
          img: ({ src, alt }) => (typeof src === 'string' && /^(https?:|\/api\/)/.test(src) ? <img src={src} alt={alt ?? ''} style={{ maxWidth: '100%' }} /> : <span className="faint">〔图片{alt ? `：${alt}` : ''}〕</span>),
          a: ({ href, children, ...rest }) => {
            if (href && href.startsWith('#cite-')) {
              const n = Number(href.slice(6));
              return (
                <button type="button" className={`cite ${activeCite === n ? 'active' : ''}`} onClick={() => onCite?.(n)} title={`查看来源 ${n}`}>
                  {n}
                </button>
              );
            }
            if (href && href.startsWith('#img-')) {
              const n = Number(href.slice(5));
              return (
                <button type="button" className="cite img" onClick={() => onImageCite?.(n)} title={`查看图片 ${n}`}>
                  图{n}
                </button>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer" {...rest}>{children}</a>;
          },
        }}
      >
        {prepared}
      </ReactMarkdown>
    </div>
  );
}
