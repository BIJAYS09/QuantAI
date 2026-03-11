import React, { useState, useEffect } from 'react'
import { useAPI } from '../../hooks/useAPI'
import '../styles/panel.css'

export default function HistoryTab() {
  const [predictions, setPredictions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true)
        // Call API to fetch prediction history
        // const data = await API.getPredictionHistory()
        // setPredictions(data)
        setPredictions([])
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch history')
      } finally {
        setLoading(false)
      }
    }

    fetchHistory()
  }, [])

  if (loading) return <div className="main-body"><div className="loading">Loading history...</div></div>
  if (error) return <div className="main-body"><div className="error">{error}</div></div>

  return (
    <div className="main-body">
      <div className="panel">
        <div className="panel-header">
          <h3>Prediction History</h3>
        </div>
        <div className="panel-content">
          {predictions.length === 0 ? (
            <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '20px' }}>
              No prediction history yet.
            </div>
          ) : (
            <div className="history-table">
              {predictions.map((pred, idx) => (
                <div key={idx} className="history-item">
                  <div>{pred.symbol}</div>
                  <div>{pred.action}</div>
                  <div>${pred.target_price.toFixed(2)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
