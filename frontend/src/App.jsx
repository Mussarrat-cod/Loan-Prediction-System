import { useState } from 'react'
import LoanForm from './components/LoanForm'
import PredictionResult from './components/PredictionResult'
import ExplanationDashboard from './components/ExplanationDashboard'

const API_URL = 'http://localhost:5000/api'

function App() {
    const [prediction, setPrediction] = useState(null)
    const [explanation, setExplanation] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [activeTab, setActiveTab] = useState('prediction')

    const handleSubmit = async (formData) => {
        setLoading(true)
        setError(null)
        setPrediction(null)
        setExplanation(null)
        setActiveTab('prediction')

        try {
            // Get both prediction and explanation
            const [predictionResponse, explanationResponse] = await Promise.all([
                fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData),
                }),
                fetch(`${API_URL}/v1/loan/explain`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData),
                })
            ])

            const predictionData = await predictionResponse.json()
            const explanationData = await explanationResponse.json()

            if (!predictionResponse.ok) {
                throw new Error(predictionData.error || 'Prediction failed')
            }

            if (!explanationResponse.ok) {
                throw new Error(explanationData.error || 'Explanation failed')
            }

            if (predictionData.success && explanationData.success) {
                setPrediction(predictionData)
                setExplanation(explanationData)
            } else {
                throw new Error('Unknown error occurred')
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
        setExplanation(null)
        setError(null)
        setActiveTab('prediction')
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-4">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8 pt-8">
                    <h1 className="text-4xl font-bold text-gray-900 mb-2">
                        Loan Eligibility Checker
                    </h1>
                    <p className="text-lg text-gray-600">
                        Get instant predictions with AI-powered explanations
                    </p>
                </div>

                {/* Main Card */}
                <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
                    {!prediction ? (
                        <div className="p-8">
                            <LoanForm onSubmit={handleSubmit} loading={loading} />
                            {error && (
                                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2">
                                    <span className="text-red-500">⚠️</span>
                                    <span className="text-red-700 text-sm">{error}</span>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div>
                            {/* Tab Navigation */}
                            <div className="border-b border-gray-200">
                                <nav className="flex space-x-8 px-8" aria-label="Tabs">
                                    <button
                                        onClick={() => setActiveTab('prediction')}
                                        className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                            activeTab === 'prediction'
                                                ? 'border-indigo-500 text-indigo-600'
                                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                        }`}
                                    >
                                        Prediction Result
                                    </button>
                                    <button
                                        onClick={() => setActiveTab('explanation')}
                                        className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                            activeTab === 'explanation'
                                                ? 'border-indigo-500 text-indigo-600'
                                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                        }`}
                                    >
                                        AI Explanation
                                    </button>
                                </nav>
                            </div>

                            {/* Tab Content */}
                            <div className="p-8">
                                {activeTab === 'prediction' ? (
                                    <PredictionResult prediction={prediction} onReset={handleReset} />
                                ) : (
                                    <div>
                                        <div className="mb-6">
                                            <button
                                                onClick={handleReset}
                                                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
                                            >
                                                ← Back to Form
                                            </button>
                                        </div>
                                        <ExplanationDashboard explanation={explanation} loading={loading} />
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="text-center mt-8 pb-8">
                    <p className="text-sm text-gray-500">
                        Powered by Explainable AI • Transparent loan decisions
                    </p>
                </div>
            </div>
        </div>
    )
}

export default App
