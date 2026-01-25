import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../ui', // Output to the server's ui directory
    emptyOutDir: true, // Clear the directory before building
    rollupOptions: {
      output: {
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`
      }
    }
  },
  base: './', // Use relative paths for assets so Flask/FastAPI serves them correctly
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
