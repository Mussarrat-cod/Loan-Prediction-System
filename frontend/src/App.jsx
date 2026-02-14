import { useState } from 'react'
import LoanForm from './components/LoanForm'
import PredictionResult from './components/PredictionResult'
import FinanceChatbot from './components/FinanceChatbot'

const API_URL = 'http://localhost:5000/api'

function App() {
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleSubmit = async (formData) => {
        setLoading(true)
        setError(null)
        setPrediction(null)

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed')
            }

            if (data.success) {
                setPrediction(data)
            } else {
                throw new Error(data.error || 'Unknown error occurred')
            }
        } catch (err) {
            setError(err.message)
            console.error('Prediction error:', err)
        } finally {
            setLoading(false)
        }
    }

    const handleReset = () => {
        setPrediction(null)
        setError(null)
    }

    return (
        <>
            <div className="card">
                <div className="app-header">
                    <h1>Loan Eligibility Checker</h1>
                    <p className="app-subtitle">
                        Get instant predictions powered by machine learning
                    </p>
                </div>

                {!prediction ? (
                    <>
                        <LoanForm onSubmit={handleSubmit} loading={loading} />
                        {error && (
                            <div className="error-message" style={{ justifyContent: 'center', marginTop: '1rem' }}>
                                <span>⚠️</span>
                                <span>{error}</span>
                            </div>
                        )}
                    </>
                ) : (
                    <PredictionResult prediction={prediction} onReset={handleReset} />
                )}
            </div>

            {/* Floating Finance Chatbot */}
            <FinanceChatbot />
        </>
    )
}

export default App
