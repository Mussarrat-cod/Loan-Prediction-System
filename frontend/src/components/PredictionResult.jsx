const PredictionResult = ({ prediction, onReset }) => {
    const isApproved = prediction.loan_status === 1

    return (
        <div className="result">
            <div className={`result-icon ${isApproved ? 'result-approved' : 'result-rejected'}`}>
                {isApproved ? '✅' : '❌'}
            </div>

            <h2>{prediction.prediction}</h2>
            <p style={{ fontSize: '1.125rem', marginBottom: '1.5rem' }}>
                {isApproved
                    ? 'Congratulations! Your loan application is likely to be approved.'
                    : 'Unfortunately, your loan application may be rejected.'}
            </p>

            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Confidence Score</span>
                    <span style={{ fontWeight: '600', fontSize: '1.125rem' }}>
                        {prediction.confidence.toFixed(1)}%
                    </span>
                </div>
                <div className="confidence-bar">
                    <div
                        className="confidence-fill"
                        style={{ width: `${prediction.confidence}%` }}
                    ></div>
                </div>
            </div>

            <div className="advice-box">
                <p style={{ margin: 0, fontSize: '0.95rem' }}>
                    <strong>💡 Advice:</strong> {prediction.advice}
                </p>
            </div>

            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '1rem',
                marginTop: '1.5rem',
                padding: '1rem',
                background: 'rgba(0,0,0,0.2)',
                borderRadius: 'var(--radius-sm)'
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Approval</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--success)' }}>
                        {prediction.probability_approved.toFixed(1)}%
                    </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Rejection</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--error)' }}>
                        {prediction.probability_rejected.toFixed(1)}%
                    </div>
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
                Prediction made at {new Date(prediction.timestamp).toLocaleTimeString()}
            </p>
        </div>
    )
}

export default PredictionResult
