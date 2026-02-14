import { useState } from 'react'

const FinanceChatbot = () => {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [isOpen, setIsOpen] = useState(false)

    const sendMessage = async () => {
        if (!input.trim()) return

        const userMessage = { role: 'user', content: input }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setLoading(true)

        try {
            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input })
            })

            const data = await response.json()

            if (data.success) {
                const botMessage = { role: 'bot', content: data.reply }
                setMessages(prev => [...prev, botMessage])
            } else {
                const errorMessage = { role: 'bot', content: '❌ ' + (data.error || 'Failed to get response') }
                setMessages(prev => [...prev, errorMessage])
            }
        } catch (error) {
            const errorMessage = { role: 'bot', content: '❌ Connection error. Make sure backend is running.' }
            setMessages(prev => [...prev, errorMessage])
        } finally {
            setLoading(false)
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                style={{
                    position: 'fixed',
                    bottom: '2rem',
                    right: '2rem',
                    width: '60px',
                    height: '60px',
                    borderRadius: '50%',
                    background: '#6366f1',
                    color: 'white',
                    border: 'none',
                    fontSize: '1.5rem',
                    cursor: 'pointer',
                    boxShadow: '0 4px 12px rgba(99, 102, 241, 0.4)',
                    transition: 'transform 0.2s ease',
                    zIndex: 1000
                }}
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.1)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
            >
                💬
            </button>
        )
    }

    return (
        <div style={{
            position: 'fixed',
            bottom: '2rem',
            right: '2rem',
            width: '380px',
            height: '550px',
            background: 'white',
            borderRadius: '16px',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.15)',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 1000,
            overflow: 'hidden'
        }}>
            {/* Header */}
            <div style={{
                background: '#6366f1',
                color: 'white',
                padding: '1rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div>
                    <div style={{ fontWeight: '600', fontSize: '1rem' }}>Finance Assistant</div>
                    <div style={{ fontSize: '0.75rem', opacity: 0.9 }}>AI advice + real-time market data</div>
                </div>
                <button
                    onClick={() => setIsOpen(false)}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'white',
                        fontSize: '1.5rem',
                        cursor: 'pointer',
                        width: 'auto',
                        padding: '0',
                        margin: '0'
                    }}
                >
                    ✕
                </button>
            </div>

            {/* Messages */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                padding: '1rem',
                background: '#f8fafc'
            }}>
                {messages.length === 0 ? (
                    <div style={{
                        textAlign: 'center',
                        color: '#64748b',
                        marginTop: '2rem',
                        fontSize: '0.9rem'
                    }}>
                        <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🤖</div>
                        <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Finance Assistant</div>
                        <div style={{ fontSize: '0.85rem', marginTop: '0.5rem', textAlign: 'left', padding: '0 1rem' }}>
                            <strong>Market Data:</strong><br />
                            • "price of AAPL"<br />
                            • "bitcoin price"<br /><br />
                            <strong>Ask Anything:</strong><br />
                            • "What is a good credit score?"<br />
                            • "How to improve loan chances?"
                        </div>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <div
                            key={idx}
                            style={{
                                marginBottom: '0.75rem',
                                display: 'flex',
                                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'
                            }}
                        >
                            <div style={{
                                maxWidth: '80%',
                                padding: '0.6rem 0.9rem',
                                borderRadius: '12px',
                                fontSize: '0.85rem',
                                lineHeight: '1.5',
                                background: msg.role === 'user' ? '#6366f1' : '#ffffff',
                                color: msg.role === 'user' ? 'white' : '#1e293b',
                                boxShadow: msg.role === 'user' ? 'none' : '0 1px 3px rgba(0,0,0,0.1)'
                            }}>
                                {msg.content}
                            </div>
                        </div>
                    ))
                )}
                {loading && (
                    <div style={{ textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
                        <span className="spinner" style={{
                            borderColor: '#e2e8f0',
                            borderTopColor: '#6366f1',
                            display: 'inline-block'
                        }}></span>
                    </div>
                )}
            </div>

            {/* Input */}
            <div style={{
                padding: '1rem',
                borderTop: '1px solid #e2e8f0',
                background: 'white'
            }}>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask anything or get stock prices..."
                        disabled={loading}
                        style={{
                            flex: 1,
                            padding: '0.6rem',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px',
                            fontSize: '0.85rem',
                            outline: 'none'
                        }}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={loading || !input.trim()}
                        style={{
                            padding: '0.6rem 1rem',
                            background: '#6366f1',
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                            fontSize: '0.85rem',
                            fontWeight: '600',
                            opacity: loading || !input.trim() ? 0.5 : 1,
                            width: 'auto',
                            marginTop: '0'
                        }}
                    >
                        Send
                    </button>
                </div>
            </div>
        </div>
    )
}

export default FinanceChatbot
