import React, { useState } from 'react'
import { TopBar } from '../components/TopBar'
import { Sidebar } from '../components/Sidebar'
import { MainTabs } from '../components/MainTabs'
import { useUIStore } from '../stores'
import OverviewTab from './tabs/OverviewTab'
import AnalysisTab from './tabs/AnalysisTab'
import PredictionTab from './tabs/PredictionTab'
import ChatTab from './tabs/ChatTab'
import HistoryTab from './tabs/HistoryTab'
import './Dashboard.css'

export default function Dashboard() {
  const activeTab = useUIStore((s) => s.activeTab)
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const selectedSymbol = useUIStore((s) => s.selectedSymbol)

  const renderTabContent = () => {
    if (!selectedSymbol && ['analysis', 'prediction'].includes(activeTab)) {
      return (
        <div className="main-body empty-state">
          <div className="empty-message">Select an asset from the watchlist to view {activeTab}</div>
        </div>
      )
    }

    switch (activeTab) {
      case 'overview':
        return <OverviewTab />
      case 'analysis':
        return <AnalysisTab symbol={selectedSymbol!} />
      case 'prediction':
        return <PredictionTab symbol={selectedSymbol!} />
      case 'chat':
        return <ChatTab />
      case 'history':
        return <HistoryTab />
      default:
        return <OverviewTab />
    }
  }

  return (
    <div className="app-layout">
      <TopBar />
      <div className="app-body">
        <Sidebar />
        <div className="main-section">
          <MainTabs activeTab={activeTab} onTabChange={setActiveTab} />
          {renderTabContent()}
        </div>
      </div>
    </div>
  )
}
