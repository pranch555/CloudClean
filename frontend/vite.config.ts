import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Built straight into the Python package so `pip install` ships the UI (no Node needed on the DGX Spark).
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../cloudclean/web/static',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    // CLOUDCLEAN_API points the dev server at another backend, e.g. an SSH tunnel to the DGX Spark
    proxy: { '/api': { target: process.env.CLOUDCLEAN_API || 'http://127.0.0.1:8765', ws: true } },
  },
});
