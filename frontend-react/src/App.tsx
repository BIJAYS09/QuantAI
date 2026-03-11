import React, { useState, useEffect } from 'react'
import { useAuthStore, useUIStore } from './stores'
import AuthPage from './pages/AuthPage'
import Dashboard from './pages/Dashboard'
import './styles/app.css'

export default function App() {
  const isAuthenticated = useAuthStore(s => s.isAuthenticated)
  const initializeAuth = useAuthStore(s => s.initializeAuth)

  useEffect(() => {
    initializeAuth()
  }, [initializeAuth])

  return (
    <div className="app-container">
      {isAuthenticated ? <Dashboard /> : <AuthPage />}
    </div>
  )
}
