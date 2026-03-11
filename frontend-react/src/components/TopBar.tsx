import React, { useEffect, useState } from 'react'
import { useWatchlistStore, useUIStore } from '../stores'
import { useWebSocketPriceUpdates } from '../hooks/useWebSocket'
import './TopBar.css'

export function TopBar() {
  const watchlist = useWatchlistStore((s) => s.watchlist)
  const { prices } = useWebSocketPriceUpdates(watchlist.map((w) => w.symbol))
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="topbar">
      <div className="logo">
        Quant<span>AI</span>
      </div>

      <div className="ticker-tape">
        <div className="ticker-inner">
          {watchlist.map((item) => {
            const live = prices.get(item.symbol)
            const price = live?.price ?? item.price
            const change_pct = live?.change_pct ?? item.change_pct
            return (
              <div key={item.symbol} className="ticker-item">
                <span className="ticker-sym">{item.symbol}</span>
                <span className="ticker-price">${price.toFixed(2)}</span>
                <span className={`ticker-chg ${change_pct >= 0 ? 'pos' : 'neg'}`}>
                  {change_pct >= 0 ? '+' : ''}{change_pct.toFixed(2)}%
                </span>
              </div>
            )
          })}
          {watchlist.map((item) => {
            const live = prices.get(item.symbol)
            const price = live?.price ?? item.price
            const change_pct = live?.change_pct ?? item.change_pct
            return (
              <div key={`${item.symbol}-2`} className="ticker-item">
                <span className="ticker-sym">{item.symbol}</span>
                <span className="ticker-price">${price.toFixed(2)}</span>
                <span className={`ticker-chg ${change_pct >= 0 ? 'pos' : 'neg'}`}>
                  {change_pct >= 0 ? '+' : ''}{change_pct.toFixed(2)}%
                </span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="top-time">
        <strong>{currentTime.toLocaleTimeString()}</strong>
        <div>{currentTime.toLocaleDateString()}</div>
      </div>
    </div>
  )
}
