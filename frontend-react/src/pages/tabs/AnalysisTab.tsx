import React, { useState, useEffect } from 'react'
import { useAPI } from '../../hooks/useAPI'
import { useUIStore } from '../../stores'
import '../styles/panel.css'

export default function AnalysisTab({ symbol }: { symbol: string }) {
  const { getStockData, getCryptoData } = useAPI()
  const assetType = useUIStore((s) => s.assetType)
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        setLoading(true)
        const result =
          assetType === 'stock'
            ? await getStockData(symbol)
            : await getCryptoData(symbol)
        setData(result)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch analysis')
      } finally {
        setLoading(false)
      }
    }

    if (symbol) {
      fetchAnalysis()
    }
  }, [symbol, assetType, getStockData, getCryptoData])

  if (loading) return <div className="main-body"><div className="loading">Loading analysis...</div></div>
  if (error) return <div className="main-body"><div className="error">{error}</div></div>

  return (
    <div className="main-body">
      <div className="panel">
        <div className="panel-header">
          <h3>{symbol} Technical Analysis</h3>
        </div>
        <div className="panel-content">
          {data && (
            <div className="analysis-grid">
              <div className="analysis-section">
                <h4>Price Information</h4>
                <div className="info-row">
                  <span>Symbol:</span>
                  <strong>{data.symbol}</strong>
                </div>
                {data.price && <div className="info-row"><span>Price:</span><strong>${data.price.toFixed(2)}</strong></div>}
                {data.change !== undefined && <div className="info-row"><span>Change:</span><strong>{data.change.toFixed(2)}</strong></div>}
              </div>
              {data.rsi !== undefined && (
                <div className="analysis-section">
                  <h4>Technical Indicators</h4>
                  <div className="info-row"><span>RSI:</span><strong>{data.rsi.toFixed(2)}</strong></div>
                  {data.sma_20 && <div className="info-row"><span>SMA 20:</span><strong>${data.sma_20.toFixed(2)}</strong></div>}
                  {data.sma_50 && <div className="info-row"><span>SMA 50:</span><strong>${data.sma_50.toFixed(2)}</strong></div>}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
