import React from 'react'
import './MainTabs.css'

export interface MainTabsProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

export function MainTabs({ activeTab, onTabChange }: MainTabsProps) {
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analysis', label: 'Analysis' },
    { id: 'prediction', label: 'Prediction' },
    { id: 'chat', label: 'Chat' },
    { id: 'history', label: 'History' },
  ]

  return (
    <div className="main-tabs">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
