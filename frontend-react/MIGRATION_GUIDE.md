#!/usr/bin/env node

/**
 * Migration Guide: Vanilla JS → React
 *
 * This document guides migrating from the single-file vanilla JS frontend
 * to a scalable React + Vite architecture.
 */

## Overview

**Old Architecture:**
- Single `frontend/index.html` (~1882 lines)
- Vanilla JS with fetch API
- No build process
- Inline styles and logic

**New Architecture:**
- `frontend-react/` with modern build tools
- Component-based structure
- Centralized state management (Zustand)
- Type-safe hooks (TypeScript)
- Separate concerns: components, pages, hooks, stores

## Directory Structure

```
frontend-react/
├── src/
│   ├── App.tsx                 # Root component
│   ├── main.tsx                # Entry point
│   ├── components/             # Reusable UI components
│   │   ├── TopBar.tsx
│   │   ├── Sidebar.tsx
│   │   └── MainTabs.tsx
│   ├── pages/                  # Page-level components
│   │   ├── Dashboard.tsx
│   │   ├── AuthPage.tsx
│   │   ├── tabs/
│   │   │   ├── OverviewTab.tsx
│   │   │   ├── AnalysisTab.tsx
│   │   │   ├── PredictionTab.tsx
│   │   │   ├── ChatTab.tsx
│   │   │   └── HistoryTab.tsx
│   │   └── styles/
│   ├── hooks/                  # Custom React hooks
│   │   ├── useAPI.ts           # API calls
│   │   └── useWebSocket.ts     # WebSocket management
│   ├── stores/                 # Zustand state management
│   │   └── index.ts
│   └── styles/                 # Global styles
│       └── globals.css
├── index.html                  # HTML template
├── vite.config.js              # Build config
├── tsconfig.json               # TypeScript config
├── package.json                # Dependencies
└── .gitignore
```

## Key Differences

### 1. State Management

**Before (vanilla JS):**
```javascript
let watchlist = [];
function updateWatchlist(item) {
  watchlist.push(item);
  render();
}
```

**After (Zustand):**
```typescript
const useWatchlistStore = create((set) => ({
  watchlist: [],
  addWatchlistItem: (item) => set((s) => ({
    watchlist: [...s.watchlist, item]
  }))
}));
```

### 2. API Calls

**Before:**
```javascript
const response = await fetch('/api/stock/AAPL');
const data = await response.json();
```

**After:**
```typescript
const { getStockData } = useAPI();
const data = await getStockData('AAPL');
```

### 3. WebSocket Handling

**Before:** Global connection state

**After:** Custom hook with cleanup
```typescript
const { price, error } = useWebSocketPriceStream('AAPL');
```

### 4. Authentication

**Before:** Token stored in localStorage manually

**After:** Zustand persistent store with automatic serialization

## Development Workflow

### Setup
```bash
cd frontend-react
npm install
npm run dev  # Start dev server on http://localhost:5173
```

### Build for Production
```bash
npm run build  # Creates optimized dist/ folder
npm run preview  # Test production build locally
```

### Environment Variables
Create `.env.local`:
```
VITE_API_URL=http://localhost:8000
```

## Migration Checklist

- [x] Set up Vite + React + TypeScript
- [x] Create component hierarchy
- [x] Build Zustand stores (auth, UI, watchlist)
- [x] Implement custom hooks (useAPI, useWebSocket)
- [x] Migrate TopBar, Sidebar, Tabs
- [x] Migrate all 5 tabs (Overview, Analysis, Prediction, Chat, History)
- [x] Create Auth page (login/register)
- [x] Port styling (CSS variables maintained)
- [ ] Integration testing with backend
- [ ] Deploy to production

## Performance Benefits

1. **Code Splitting:** Each route/page loaded separately
2. **Hot Module Replacement:** Instant UI updates during development
3. **Tree Shaking:** Unused code removed from bundle
4. **Bundling:** Reduces HTTP requests
5. **Lazy Loading:** Tabs/components loaded on-demand

Expected bundle size: ~150KB (gzipped) vs ~2MB (vanilla).

## API Compatibility

All existing endpoints remain unchanged:
- `/api/auth/*`
- `/api/stock/{symbol}`
- `/api/crypto/{symbol}`
- `/api/predict/*`
- `/api/chat`
- `/ws/prices/{symbol}`

## Common Tasks

### Add a New Component

1. Create `src/components/MyComponent.tsx`
2. Create `src/components/MyComponent.css`
3. Export from parent

### Add a New Page

1. Create `src/pages/MyPage.tsx`
2. Import in `App.tsx`
3. Add routing logic

### Add State

1. Create store in `src/stores/index.ts`
2. Export hook
3. Use in component: `const value = useStore(s => s.value)`

### Fetch Data

```typescript
import { useAPI } from '../hooks/useAPI';
const { getStockData } = useAPI();
const data = await getStockData('AAPL');
```

## Troubleshooting

**CORS Issues:**
- Vite proxy automatically configured in `vite.config.js`

**WebSocket Connection Fails:**
- Check `useWebSocketPriceStream` hook in `hooks/useWebSocket.ts`
- Ensure backend WebSocket endpoint is accessible

**State Not Persisting:**
- Check Zustand `persist` middleware in stores
- Verify localStorage is enabled

## Next Steps

1. Install dependencies: `npm install`
2. Start dev server: `npm run dev`
3. Test login/logout flow
4. Connect watchlist to real API
5. Deploy to production

---

**Last Updated:** March 10, 2026
**Architecture:** React 18 + Vite + TypeScript + Zustand
**Styling:** CSS with design tokens (CSS variables)
