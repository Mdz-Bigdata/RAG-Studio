// 基础 UI 原语：Button / Badge / Toggle / Field / Modal / Drawer / Empty / Spinner / ScoreBar / Tabs / Tag
import React, { useEffect } from 'react';
import { Inbox, Loader2, X } from 'lucide-react';

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'primary' | 'ghost' | 'danger';
  size?: 'md' | 'sm' | 'xs';
  icon?: boolean;
  loading?: boolean;
};

export function Button({ variant = 'default', size = 'md', icon, loading, className = '', children, disabled, ...rest }: ButtonProps) {
  const cls = ['btn', variant !== 'default' ? variant : '', size !== 'md' ? size : '', icon ? 'icon' : '', className].filter(Boolean).join(' ');
  return (
    <button className={cls} disabled={disabled || loading} {...rest}>
      {loading ? <Loader2 size={14} className="spin" /> : null}
      {children}
    </button>
  );
}

export function Badge({ tone = '', mono, className = '', children, title }: { tone?: string; mono?: boolean; className?: string; children: React.ReactNode; title?: string }) {
  return <span className={`badge ${tone} ${mono ? 'mono' : ''} ${className}`} title={title}>{children}</span>;
}

export function Toggle({ on, onChange, disabled }: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return <span role="switch" aria-checked={on} className={`toggle ${on ? 'on' : ''}`} style={{ opacity: disabled ? 0.5 : 1 }} onClick={() => !disabled && onChange(!on)} />;
}

export function Field({ label, hint, inline, children }: { label: React.ReactNode; hint?: React.ReactNode; inline?: boolean; children: React.ReactNode }) {
  return (
    <div className={`field ${inline ? 'inline' : ''}`}>
      <label>
        <span>{label}</span>
        {hint && !inline ? null : null}
      </label>
      {children}
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  );
}

export function Modal({ title, onClose, children, footer, wide }: { title: React.ReactNode; onClose: () => void; children: React.ReactNode; footer?: React.ReactNode; wide?: boolean }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? 'wide' : ''}`}>
        <div className="modal-header">
          <h3>{title}</h3>
          <Button variant="ghost" icon size="sm" onClick={onClose} aria-label="关闭"><X size={16} /></Button>
        </div>
        <div className="modal-body">{children}</div>
        {footer ? <div className="modal-footer">{footer}</div> : null}
      </div>
    </div>
  );
}

export function Drawer({ title, subtitle, onClose, children, footer }: { title: React.ReactNode; subtitle?: React.ReactNode; onClose: () => void; children: React.ReactNode; footer?: React.ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" style={{ justifyContent: 'flex-end', padding: 0 }} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="drawer">
        <div className="modal-header">
          <div>
            <h3>{title}</h3>
            {subtitle ? <div className="faint small">{subtitle}</div> : null}
          </div>
          <Button variant="ghost" icon size="sm" onClick={onClose} aria-label="关闭"><X size={16} /></Button>
        </div>
        <div className="modal-body flex-1" style={{ display: 'flex', flexDirection: 'column' }}>{children}</div>
        {footer ? <div className="modal-footer">{footer}</div> : null}
      </div>
    </div>
  );
}

export function Empty({ icon, title, desc, action }: { icon?: React.ReactNode; title: string; desc?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="empty">
      {icon ?? <Inbox size={36} />}
      <h4>{title}</h4>
      {desc ? <p>{desc}</p> : null}
      {action}
    </div>
  );
}

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return (
    <span className="row gap-6 muted small">
      <Loader2 size={size} className="spin" />
      {label}
    </span>
  );
}

export function ScoreBar({ value, max = 1, color }: { value: number; max?: number; color?: string }) {
  const w = Math.max(0, Math.min(100, (value / (max || 1)) * 100));
  return (
    <div className="score-bar">
      <i style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

export function Tabs<T extends string>({ value, onChange, items }: { value: T; onChange: (v: T) => void; items: { value: T; label: React.ReactNode; icon?: React.ReactNode }[] }) {
  return (
    <div className="tabs">
      {items.map((it) => (
        <button key={it.value} className={`tab ${value === it.value ? 'active' : ''}`} onClick={() => onChange(it.value)}>
          {it.icon}
          {it.label}
        </button>
      ))}
    </div>
  );
}

export function Kpi({ label, value, sub, icon }: { label: string; value: React.ReactNode; sub?: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="card kpi">
      <div className="label">{icon}{label}</div>
      <div className="value">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  );
}

export function Callout({ tone = '', icon, children }: { tone?: '' | 'warn' | 'bad'; icon?: React.ReactNode; children: React.ReactNode }) {
  return <div className={`callout ${tone}`}>{icon}<div>{children}</div></div>;
}

export function confirmDialog(message: string): boolean {
  return window.confirm(message);
}
