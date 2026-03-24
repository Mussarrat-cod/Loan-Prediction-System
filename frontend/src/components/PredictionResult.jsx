const PredictionResult = ({ prediction, onReset }) => {
    const isApproved = prediction.prediction === 1
    const confidenceValue = parseFloat(prediction.confidence) || 0

    return (
        <div className="result">
            <div className={`result-icon ${isApproved ? 'result-approved' : 'result-rejected'}`}>
                {isApproved ? '✅' : '❌'}
            </div>

            <h2>Loan Status: {prediction.status} {isApproved ? '✅' : '❌'}</h2>
            
            {!isApproved && (
                <div style={{
                    background: 'rgba(255,255,255,0.1)',
                    padding: '1rem',
                    borderRadius: 'var(--radius-sm)',
                    marginBottom: '1.5rem'
                }}>
                    <h3 style={{ color: '#ff6b6b', marginBottom: '0.5rem' }}>Reason:</h3>
                    <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
                        <li>Low income</li>
                        <li>High debt</li>
                    </ul>
                </div>
            )}

            <div style={{
                background: 'rgba(255,255,255,0.1)',
                padding: '1rem',
                borderRadius: 'var(--radius-sm)',
                marginBottom: '1.5rem'
            }}>
                <h3 style={{ color: '#4ecdc4', marginBottom: '0.5rem' }}>Suggestions:</h3>
                <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
                    {isApproved ? (
                        <>
                            <li>Maintain good credit score</li>
                            <li>Keep stable income</li>
                            <li>Avoid new debt applications</li>
                        </>
                    ) : (
                        <>
                            <li>Reduce EMI</li>
                            <li>Increase income proof</li>
                            <li>Improve credit score</li>
                        </>
                    )}
                </ul>
            </div>

            <div style={{
                textAlign: 'center',
                padding: '0.5rem',
                background: 'linear-gradient(45deg, #667eea, #764ba2)',
                borderRadius: 'var(--radius-sm)',
                marginBottom: '1.5rem',
                color: 'white',
                fontWeight: '600'
            }}>
                 AI-powered advisor 
            </div>

            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Confidence Score</span>
                    <span style={{ fontWeight: '600', fontSize: '1.125rem' }}>
                        {confidenceValue.toFixed(1)}%
                    </span>
                </div>
                <div className="confidence-bar">
                    <div
                        className="confidence-fill"
                        style={{ width: `${confidenceValue}%` }}
                    ></div>
                </div>
            </div>

            <button onClick={onReset} className="btn-secondary" style={{ width: '100%' }}>
                ↻ Check Another Application
            </button>

            <p style={{
                marginTop: '1.5rem',
                fontSize: '0.875rem',
                color: 'var(--text-muted)',
                textAlign: 'center'
            }}>
                Prediction made at {new Date().toLocaleTimeString()}
            </p>
        </div>
    )
}

export default PredictionResult
