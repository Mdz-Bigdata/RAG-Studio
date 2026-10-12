// 主题：默认跟随系统，手动切换后记住选择。通过 html[data-theme] 切换 styles.css 里的两套变量。
export type Theme = 'light' | 'dark';

const KEY = 'rag.theme';

export function currentTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* ignore */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function applyTheme(theme: Theme, persist = true): void {
  document.documentElement.dataset.theme = theme;
  if (!persist) return;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* ignore */
  }
}
