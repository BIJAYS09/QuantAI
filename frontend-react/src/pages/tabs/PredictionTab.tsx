import React, { useState, useEffect } from 'react'
import { useAPI } from '../../hooks/useAPI'
import { useUIStore } from '../../stores'
import '../styles/panel.css'

export default function PredictionTab({ symbol }: { symbol: string }) {
  const { getPrediction } = useAPI()
  const assetType = useUIStore((s) => s.assetType)
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        setLoading(true)
        const result = await getPrediction(symbol, assetType)
        setPrediction(result)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch prediction')
      } finally {
        setLoading(false)
      }
    }

    if (symbol) {
      fetchPrediction()
    }
  }, [symbol, assetType, getPrediction])

  if (loading) return <div className="main-body"><div className="loading">Loading prediction...</div></div>
  if (error) return <div className="main-body"><div className="error">{error}</div></div>

  return (
    <div className="main-body">
      <div className="panel">
        <div className="panel-header">
          <h3>{symbol} Prediction</h3>
        </div>
        <div className="panel-content">
          {prediction && (
            <div className="prediction-card">
              <div className={`action ${prediction.action.toLowerCase()}`}>
                {prediction.action}
              </div>
              <div className="info-row">
                <span>Target Price:</span>
                <strong>${prediction.target_price.toFixed(2)}</strong>
              </div>
              <div className="info-row">
                <span>Confidence:</span>
                <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
              </div>
              <div className="info-row">
                <span>Reason:</span>
                <p>{prediction.reason}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
