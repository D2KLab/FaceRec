import path from 'path'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import nib from 'nib'
import rupture from 'rupture'
import config from './app.config'

// https://vitejs.dev/config/
export default defineConfig({
  runtimeCompiler: true,
  base: config.base,
  plugins: [vue()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, './src')
    }
  },
  css: {
    preprocessorOptions: {
      stylus: {
        use: [nib(), rupture()],
      },
    },
  }
})
