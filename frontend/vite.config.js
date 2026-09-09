import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // Electron file:// protocol compatibility
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
