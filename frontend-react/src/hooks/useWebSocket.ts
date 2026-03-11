import { useEffect, useCallback, useRef, useState } from 'react'

export interface PriceUpdate {
  symbol: string
  price: number
  change_pct: number
  timestamp: string
}

type PriceUpdateHandler = (update: PriceUpdate) => void

const handlers = new Map<string, Set<PriceUpdateHandler>>()
const connections = new Map<string, WebSocket>()

export function useWebSocketPriceStream(symbol: string | null) {
  const [price, setPrice] = useState<PriceUpdate | null>(null)
  const [error, setError] = useState<string | null>(null)
  const handlerRef = useRef<PriceUpdateHandler | null>(null)

  useEffect(() => {
    if (!symbol) return

    const handler: PriceUpdateHandler = (update) => {
      if (update.symbol === symbol) {
        setPrice(update)
      }
    }

    handlerRef.current = handler
    if (!handlers.has(symbol)) {
      handlers.set(symbol, new Set())
    }
    handlers.get(symbol)!.add(handler)

    // Open connection if not already open
    if (!connections.has(symbol)) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/prices/${symbol}`)

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data)
          handlers.get(symbol)?.forEach((h) => h(update))
        } catch (err) {
          setError(`Failed to parse WebSocket message: ${err}`)
        }
      }

      ws.onerror = () => {
        setError(`WebSocket error for ${symbol}`)
      }

      ws.onclose = () => {
        connections.delete(symbol)
        handlers.delete(symbol)
      }

      connections.set(symbol, ws)
    }

    return () => {
      if (handlerRef.current) {
        handlers.get(symbol)?.delete(handlerRef.current)
      }
    }
  }, [symbol])

  return { price, error }
}

export function useWebSocketPriceUpdates(symbols: string[]) {
  const [prices, setPrices] = useState<Map<string, PriceUpdate>>(new Map())
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const newPrices = new Map(prices)

    symbols.forEach((symbol) => {
      const handler: PriceUpdateHandler = (update) => {
        newPrices.set(update.symbol, update)
        setPrices(new Map(newPrices))
      }

      if (!handlers.has(symbol)) {
        handlers.set(symbol, new Set())
      }
      handlers.get(symbol)!.add(handler)

      if (!connections.has(symbol)) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws/prices/${symbol}`)

        ws.onmessage = (event) => {
          try {
            const update = JSON.parse(event.data)
            handlers.get(symbol)?.forEach((h) => h(update))
          } catch (err) {
            setError(`Failed to parse WebSocket message`)
          }
        }

        ws.onerror = () => {
          setError(`WebSocket error for ${symbol}`)
        }

        connections.set(symbol, ws)
      }
    })

    return () => {
      symbols.forEach((symbol) => {
        const connHandlers = handlers.get(symbol)
        if (connHandlers?.size === 0) {
          connections.get(symbol)?.close()
          connections.delete(symbol)
          handlers.delete(symbol)
        }
      })
    }
  }, [symbols])

  return { prices, error }
}
