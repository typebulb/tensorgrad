import { defineConfig, type Plugin } from 'vite'

// Tiny plugin that exposes POST /__log so the page can stream training output
// to the dev server's stdout (which I'm tailing). Avoids the meat-relay.
function logTap(): Plugin {
  return {
    name: 'log-tap',
    configureServer(server) {
      server.middlewares.use('/__log', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return }
        const chunks: Buffer[] = []
        req.on('data', c => chunks.push(c))
        req.on('end', () => {
          try {
            const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as { msg: string }
            // eslint-disable-next-line no-console
            console.log(`[bulb] ${body.msg}`)
          } catch { /* ignore */ }
          res.statusCode = 204
          res.end()
        })
      })
    },
  }
}

export default defineConfig({
  server: { port: 4000, strictPort: true },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        index: 'index.html',
        transformer: 'transformer/index.html',
        'mlp-sin': 'mlp-sin/index.html',
        'digit-canvas': 'digit-canvas/index.html',
        'mnist-cnn': 'mnist-cnn/index.html',
      },
    },
  },
  plugins: [logTap()],
})
