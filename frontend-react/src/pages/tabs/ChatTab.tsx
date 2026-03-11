import React, { useState, useRef, useEffect } from 'react'
import { useAPI } from '../../hooks/useAPI'
import '../styles/panel.css'
import '../styles/chat.css'

export default function ChatTab() {
  const { chat } = useAPI()
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])

    try {
      setLoading(true)
      setError(null)
      const response = await chat(userMessage)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.message,
        },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get response')
      setMessages((prev) =>
        prev.slice(0, -1).concat([
          { role: 'error', content: 'Failed to send message. Try again.' },
        ])
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="main-body chat-container">
      {error && <div className="error-banner">{error}</div>}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-welcome">
            <h3>Market Intelligence Chat</h3>
            <p>Ask me anything about stocks, crypto, market trends, and predictions.</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-role">{msg.role === 'user' ? 'You' : msg.role === 'assistant' ? 'QuantAI' : 'Error'}</div>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        {loading && <div className="message assistant"><div className="message-role">QuantAI</div><div className="message-content">Typing...</div></div>}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input" onSubmit={handleSendMessage}>
        <input
          type="text"
          placeholder="Ask about markets, assets, predictions..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
