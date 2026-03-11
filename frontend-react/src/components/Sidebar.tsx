import React, { useState, useRef, useEffect } from 'react'
import { useWatchlistStore, useUIStore } from '../stores'
import { useWebSocketPriceUpdates } from '../hooks/useWebSocket'
import './Sidebar.css'

export function Sidebar() {
  const watchlist = useWatchlistStore((s) => s.watchlist)
  const addWatchlistItem = useWatchlistStore((s) => s.addWatchlistItem)
  const removeWatchlistItem = useWatchlistStore((s) => s.removeWatchlistItem)
  const selectedSymbol = useUIStore((s) => s.selectedSymbol)
  const setSelectedSymbol = useUIStore((s) => s.setSelectedSymbol)
  const assetType = useUIStore((s) => s.assetType)
  const setAssetType = useUIStore((s) => s.setAssetType)

  const { prices } = useWebSocketPriceUpdates(watchlist.map((w) => w.symbol))

  const [searchInput, setSearchInput] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const searchRef = useRef<HTMLDivElement>(null)

  const handleSearch = async (query: string) => {
    setSearchInput(query)
    if (query.length < 1) {
      setSearchResults([])
      return
    }
    // Mock search results - replace with actual API call
    setSearchResults([
      { symbol: query.toUpperCase(), name: `Search result for ${query}`, type: assetType },
    ])
  }

  const handleAddToWatchlist = (symbol: string) => {
    addWatchlistItem({
      symbol,
      name: `${symbol} ${assetType}`,
      price: 0,
      change_pct: 0,
      type: assetType,
    })
    setSearchInput('')
    setSearchResults([])
  }

  const handleRemoveFromWatchlist = (symbol: string, e: React.MouseEvent) => {
    e.stopPropagation()
    removeWatchlistItem(symbol)
  }

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <div className="sidebar-label">Asset Type</div>
        <div className="type-toggle">
          <button
            className={`type-btn ${assetType === 'stock' ? 'active' : ''}`}
            onClick={() => setAssetType('stock')}
          >
            Stocks
          </button>
          <button
            className={`type-btn ${assetType === 'crypto' ? 'active' : ''}`}
            onClick={() => setAssetType('crypto')}
          >
            Crypto
          </button>
        </div>
      </div>

      <div className="sidebar-section" ref={searchRef}>
        <div className="search-wrap">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search..."
            value={searchInput}
            onChange={(e) => handleSearch(e.target.value)}
            autoComplete="off"
          />
        </div>

        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((result) => (
              <div
                key={result.symbol}
                className="search-result-item"
                onClick={() => handleAddToWatchlist(result.symbol)}
              >
                <div className="search-result-symbol">{result.symbol}</div>
                <div className="search-result-name">{result.name}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">Watchlist</div>
        {watchlist.length === 0 ? (
          <div style={{ color: 'var(--text-dim)', fontSize: '11px', padding: '8px 10px' }}>
            No items. Search to add.
          </div>
        ) : (
          watchlist.map((item) => {
            const live = prices.get(item.symbol)
            const price = live?.price ?? item.price
            const change_pct = live?.change_pct ?? item.change_pct
            return (
              <div
                key={item.symbol}
                className={`watchlist-item ${selectedSymbol === item.symbol ? 'active' : ''}`}
                onClick={() => setSelectedSymbol(item.symbol)}
              >
                <div className="wl-left">
                  <div className="wl-sym">{item.symbol}</div>
                  <div className="wl-name">{item.name}</div>
                </div>
                <div className="wl-right">
                  <div className="wl-price">${price.toFixed(2)}</div>
                  <div className={`wl-chg ${change_pct >= 0 ? 'pos' : 'neg'}`}>
                    {change_pct >= 0 ? '+' : ''}{change_pct.toFixed(2)}%
                  </div>
                  <button
                    className="remove-btn"
                    onClick={(e) => handleRemoveFromWatchlist(item.symbol, e)}
                    title="Remove"
                  >
                    ✕
                  </button>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
