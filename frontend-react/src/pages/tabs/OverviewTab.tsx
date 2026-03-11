import React, { useState, useEffect } from 'react'
import { useAPI } from '../../hooks/useAPI'
import '../styles/panel.css'

export default function OverviewTab() {
  const { getMarketOverview } = useAPI()
  const [marketData, setMarketData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchOverview = async () => {
      try {
        setLoading(true)
        const data = await getMarketOverview()
        setMarketData(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch market overview')
      } finally {
        setLoading(false)
      }
    }

    fetchOverview()
  }, [getMarketOverview])

  if (loading) return <div className="main-body"><div className="loading">Loading market data...</div></div>
  if (error) return <div className="main-body"><div className="error">{error}</div></div>

  return (
    <div className="main-body">
      <div className="panel">
        <div className="panel-header">
          <h3>Market Overview</h3>
        </div>
        <div className="panel-content">
          {marketData && (
            <div className="overview-grid">
              {Object.entries(marketData).map(([key, value]: [string, any]) => (
                <div key={key} className="overview-item">
                  <div className="overview-label">{key}</div>
                  <div className="overview-value">{JSON.stringify(value)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
