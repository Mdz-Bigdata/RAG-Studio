import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 开发态把 /api 代理到 FastAPI，前端代码里只用相对路径 `/api/v1`。
// 后端不在 8000 端口时：VITE_API_TARGET=http://localhost:8010 npm run dev（start.sh 会自动设置）
const target = process.env.VITE_API_TARGET ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/docs': { target, changeOrigin: true },
      '/openapi.json': { target, changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
});
