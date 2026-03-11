import { useCallback } from 'react'
import { useAuthStore } from '../stores'

export interface ApiError {
  detail?: string
  message?: string
}

export async function apiCall<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const token = useAuthStore.getState().accessToken
  const headers = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...options?.headers,
  }

  const response = await fetch(endpoint, {
    ...options,
    headers,
  })

  if (!response.ok) {
    const error: ApiError = await response.json()
    throw new Error(error.detail || error.message || `HTTP ${response.status}`)
  }

  return response.json() as Promise<T>
}

export interface StockData {
  symbol: string
  price: number
  change: number
  change_pct: number
  volume: number
  market_cap: string
  pe_ratio: number | null
  rsi: number
  macd: { macd: number; signal: number; histogram: number }
  sma_20: number
  sma_50: number
  bollinger: { upper: number; lower: number; middle: number }
}

export interface CryptoData {
  symbol: string
  name: string
  price: number
  market_cap: number
  volume_24h: number
  change_24h: number
  change_7d: number
  change_30d: number
  circulating_supply: number
}

export interface Prediction {
  symbol: string
  action: 'BUY' | 'SELL' | 'HOLD'
  target_price: number
  confidence: number
  reason: string
}

export function useAPI() {
  const getStockData = useCallback(async (symbol: string): Promise<StockData> => {
    return apiCall(`/api/stock/${symbol}`)
  }, [])

  const getCryptoData = useCallback(async (symbol: string): Promise<CryptoData> => {
    return apiCall(`/api/crypto/${symbol}`)
  }, [])

  const getPrediction = useCallback(async (symbol: string, assetType: string): Promise<Prediction> => {
    return apiCall(`/api/predict/${assetType}/${symbol}`)
  }, [])

  const getMarketOverview = useCallback(async () => {
    return apiCall('/api/market-overview')
  }, [])

  const chat = useCallback(
    async (message: string): Promise<{ message: string; data_type: string; asset_data?: unknown; prediction?: Prediction }> => {
      return apiCall('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ query: message }),
      })
    },
    []
  )

  const quickAnalyze = useCallback(
    async (symbol: string, assetType: string) => {
      return apiCall('/api/quick-analyze', {
        method: 'POST',
        body: JSON.stringify({ symbol, asset_type: assetType }),
      })
    },
    []
  )

  return {
    getStockData,
    getCryptoData,
    getPrediction,
    getMarketOverview,
    chat,
    quickAnalyze,
  }
}

export function refreshTokenIfNeeded() {
  return useAuthStore.getState().refresh()
}
