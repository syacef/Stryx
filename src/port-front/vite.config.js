import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/streams': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: true,
        cookieDomainRewrite: 'localhost',
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            const cookies = proxyRes.headers['set-cookie'];
            if (cookies) {
              proxyRes.headers['set-cookie'] = cookies.map((cookie) =>
                cookie
                  .replace(/Domain=[^;]+/i, 'Domain=localhost')
                  .replace(/; *Secure/gi, '')
              );
            }
          });
        },
      },
      '/workers': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: true,
        cookieDomainRewrite: 'localhost',
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            const cookies = proxyRes.headers['set-cookie'];
            if (cookies) {
              proxyRes.headers['set-cookie'] = cookies.map((cookie) =>
                cookie
                  .replace(/Domain=[^;]+/i, 'Domain=localhost')
                  .replace(/; *Secure/gi, '')
              );
            }
          });
        },
      },
    },
  },
})
