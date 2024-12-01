import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    // Optimize build settings
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
  server: {
    port: 5173,
    // Proxy is only needed for development
    ...(process.env.NODE_ENV === 'development' ? {
      proxy: {
        '/api': {
          target: 'https://loon-stirred-terribly.ngrok-free.app',
          changeOrigin: true,
          secure: false,
        }
      }
    } : {})
  }
});
