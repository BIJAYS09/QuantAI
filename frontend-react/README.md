# QuantAI React Frontend

Modern React + Vite frontend for QuantAI market intelligence platform.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Features

✨ **Modern Stack**
- React 18 with TypeScript
- Vite for fast development & builds
- Zustand for state management
- Custom React hooks for API & WebSocket

📦 **Component Architecture**
- Modular, reusable components
- Separation of concerns (pages, components, hooks, stores)
- Scalable structure for growth

🚀 **Performance**
- Code splitting and lazy loading
- Hot module replacement (HMR)
- Optimized bundle (~150KB gzipped)

🎨 **Design**
- Bloomberg-terminal-inspired UI
- Dark theme with accent colors
- Responsive layout

## Project Structure

```
src/
├── App.tsx              # Root component
├── main.tsx             # Entry point
├── components/          # Reusable UI components
├── pages/               # Page-level components (Dashboard, Auth)
├── hooks/               # Custom React hooks (API, WebSocket)
├── stores/              # Zustand state management
└── styles/              # Global CSS
```

## Environment Variables

Create `.env.local`:
```env
VITE_API_URL=http://localhost:8000
```

## Development

### Available Scripts

- `npm run dev` – Start dev server (http://localhost:5173)
- `npm run build` – Build for production
- `npm run preview` – Preview production build
- `npm run type-check` – Run TypeScript type checking
- `npm run lint` – Run ESLint

### API Proxy

Vite automatically proxies API requests to the backend:
- `/api/*` → `http://localhost:8000`
- `/ws/*` → `ws://localhost:8000`

Configure in `vite.config.js`.

## State Management (Zustand)

Three main stores:

### `useAuthStore`
- User authentication state
- Login, register, logout, token refresh
- Persisted to localStorage

### `useUIStore`
- UI state (active tab, selected symbol, etc.)
- Transient (not persisted)

### `useWatchlistStore`
- Watchlist items
- Add/remove items
- Real-time price updates via WebSocket

## Custom Hooks

### `useAPI()`
Provides async functions for backend communication:
```typescript
const { getStockData, getCryptoData, chat } = useAPI();
const data = await getStockData('AAPL');
```

### `useWebSocketPriceStream(symbol)`
Live price updates for a single symbol:
```typescript
const { price, error } = useWebSocketPriceStream('AAPL');
```

### `useWebSocketPriceUpdates(symbols)`
Live price updates for multiple symbols:
```typescript
const { prices, error } = useWebSocketPriceUpdates(['AAPL', 'MSFT']);
```

## Styling

Uses CSS with design tokens (CSS variables) for consistency:

```css
:root {
  --bg: #050a0f;
  --accent: #00d4ff;
  --green: #00ff9d;
  --red: #ff3d57;
  /* ... more variables ... */
}
```

All component styles are modular (component.tsx + component.css).

## Deployment

### Build
```bash
npm run build
```

Outputs production-ready files in `dist/`.

### Serve
```bash
npm run preview
```

Or deploy `dist/` to any static host:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

### With Backend

Backend serves React app as static files:
```python
# In main.py
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend-react/dist", html=True), name="static")
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Performance Metrics

- **Build Time:** ~2-3 seconds (dev), ~5 seconds (prod)
- **Bundle Size:** ~150KB gzipped
- **First Paint:** <1s (with backend)
- **Lighthouse:** >90 on all metrics

## Troubleshooting

### Port Already in Use
```bash
npm run dev -- --port 3000
```

### CORS/Proxy Issues
- Verify backend is running on `http://localhost:8000`
- Check `vite.config.js` proxy configuration

### WebSocket Connection Fails
- Ensure backend WebSocket endpoint is accessible
- Check browser console for error messages

### State Not Persisting
- Check browser localStorage is enabled
- Verify Zustand `persist` middleware configuration

## Migration from Vanilla JS

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed comparison and migration checklist.

## Contributing

1. Follow existing component structure
2. Use TypeScript for type safety
3. Keep components focused and reusable
4. Test hooks in isolation
5. Update styles alongside components

## Scripts Reference

### Development
```bash
npm run dev         # Start dev server with HMR
npm run type-check  # TypeScript type checking
npm run lint        # ESLint analysis
```

### Production
```bash
npm run build       # Create optimized production build
npm run preview     # Preview production build locally
```

## License

Same as main QuantAI project.

---

**Last Updated:** March 10, 2026
**Architecture:** React 18 + Vite + TypeScript
