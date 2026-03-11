import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface AuthState {
  accessToken: string | null
  refreshToken: string | null
  user: { user_id: string; email: string; username: string; created_at: string } | null
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  refresh: () => Promise<void>
  initializeAuth: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,

      login: async (email: string, password: string) => {
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
        })
        if (!response.ok) throw new Error('Login failed')
        const data = await response.json()
        set({
          accessToken: data.access_token,
          refreshToken: data.refresh_token,
          user: data.user,
          isAuthenticated: true,
        })
      },

      register: async (email: string, username: string, password: string) => {
        const response = await fetch('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, username, password }),
        })
        if (!response.ok) throw new Error('Registration failed')
        const data = await response.json()
        set({
          accessToken: data.access_token,
          refreshToken: data.refresh_token,
          user: data.user,
          isAuthenticated: true,
        })
      },

      logout: async () => {
        const token = get().accessToken
        if (token) {
          try {
            await fetch('/api/auth/logout', {
              method: 'POST',
              headers: { Authorization: `Bearer ${token}` },
            })
          } catch {
            // Silent fail
          }
        }
        set({ accessToken: null, refreshToken: null, user: null, isAuthenticated: false })
      },

      refresh: async () => {
        const token = get().refreshToken
        if (!token) throw new Error('No refresh token')
        const response = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: token }),
        })
        if (!response.ok) throw new Error('Refresh failed')
        const data = await response.json()
        set({
          accessToken: data.access_token,
          refreshToken: data.refresh_token,
        })
      },

      initializeAuth: () => {
        // Check if tokens exist in localStorage (already persisted by zustand)
        const state = get()
        if (state.accessToken) {
          set({ isAuthenticated: true })
        }
      },
    }),
    { name: 'auth-store' }
  )
)

export interface UIState {
  activeTab: string
  assetType: 'stock' | 'crypto'
  selectedSymbol: string | null
  sidebarOpen: boolean
  setActiveTab: (tab: string) => void
  setAssetType: (type: 'stock' | 'crypto') => void
  setSelectedSymbol: (symbol: string | null) => void
  toggleSidebar: () => void
}

export const useUIStore = create<UIState>((set) => ({
  activeTab: 'overview',
  assetType: 'stock',
  selectedSymbol: null,
  sidebarOpen: true,
  setActiveTab: (tab) => set({ activeTab: tab }),
  setAssetType: (type) => set({ assetType: type }),
  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}))

export interface Asset {
  symbol: string
  name: string
  price: number
  change_pct: number
  type: 'stock' | 'crypto'
}

export interface WatchlistState {
  watchlist: Asset[]
  addWatchlistItem: (asset: Asset) => void
  removeWatchlistItem: (symbol: string) => void
  loadWatchlist: () => Promise<void>
}

export const useWatchlistStore = create<WatchlistState>((set, get) => ({
  watchlist: [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 192.53, change_pct: 1.23, type: 'stock' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 379.85, change_pct: 0.87, type: 'stock' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 140.23, change_pct: -0.45, type: 'stock' },
    { symbol: 'BTC', name: 'Bitcoin', price: 45230.50, change_pct: 2.15, type: 'crypto' },
    { symbol: 'ETH', name: 'Ethereum', price: 2385.60, change_pct: 1.82, type: 'crypto' },
  ],
  addWatchlistItem: (asset) =>
    set((s) => ({
      watchlist: [...s.watchlist.filter((a) => a.symbol !== asset.symbol), asset],
    })),
  removeWatchlistItem: (symbol) =>
    set((s) => ({ watchlist: s.watchlist.filter((a) => a.symbol !== symbol) })),
  loadWatchlist: async () => {
    // Implement API call to fetch watchlist from backend
  },
}))
