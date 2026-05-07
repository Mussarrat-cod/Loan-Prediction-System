import { CheckCircle, XCircle, RefreshCw, TrendingUp, AlertTriangle } from 'lucide-react'

const PredictionResult = ({ prediction, onReset }) => {
    const isApproved = prediction.prediction === 1
    const confidenceValue = parseFloat(prediction.confidence) || 0

    return (
        <div className="text-center space-y-6 animate-fade-in">
            {/* Status Icon and Title */}
            <div className="flex flex-col items-center space-y-4">
                <div className={`p-4 rounded-full ${
                    isApproved ? 'bg-green-100' : 'bg-red-100'
                }`}>
                    {isApproved ? (
                        <CheckCircle className="w-16 h-16 text-green-600" />
                    ) : (
                        <XCircle className="w-16 h-16 text-red-600" />
                    )}
                </div>
                <div>
                    <h2 className={`text-2xl font-bold ${
                        isApproved ? 'text-green-800' : 'text-red-800'
                    }`}>
                        Loan {prediction.status}
                    </h2>
                    <p className="text-gray-600 mt-1">
                        {isApproved 
                            ? 'Congratulations! Your loan application has been approved.' 
                            : 'Unfortunately, your loan application was not approved.'}
                    </p>
                </div>
            </div>

            {/* Confidence Score */}
            <div className="bg-gray-50 p-6 rounded-xl">
                <div className="flex justify-between items-center mb-3">
                    <span className="text-sm font-medium text-gray-600">AI Confidence Score</span>
                    <span className="text-lg font-bold text-gray-800">
                        {confidenceValue.toFixed(1)}%
                    </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all duration-1000 ease-out ${
                            confidenceValue >= 80 ? 'bg-green-500' : 
                            confidenceValue >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${confidenceValue}%` }}
                    ></div>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                    {confidenceValue >= 80 ? 'High confidence prediction' :
                     confidenceValue >= 60 ? 'Moderate confidence prediction' :
                     'Low confidence prediction'}
                </p>
            </div>

            {/* Suggestions */}
            <div className={`p-6 rounded-xl ${
                isApproved ? 'bg-green-50 border border-green-200' : 'bg-amber-50 border border-amber-200'
            }`}>
                <h3 className={`font-semibold mb-3 flex items-center ${
                    isApproved ? 'text-green-800' : 'text-amber-800'
                }`}>
                    {isApproved ? (
                        <>
                            <TrendingUp className="w-5 h-5 mr-2" />
                            Next Steps
                        </>
                    ) : (
                        <>
                            <AlertTriangle className="w-5 h-5 mr-2" />
                            Recommendations
                        </>
                    )}
                </h3>
                <ul className="space-y-2 text-sm">
                    {isApproved ? (
                        <>
                            <li className="flex items-start">
                                <span className="text-green-600 mr-2">•</span>
                                <span className="text-gray-700">Maintain good credit score for better loan terms</span>
                            </li>
                            <li className="flex items-start">
                                <span className="text-green-600 mr-2">•</span>
                                <span className="text-gray-700">Keep stable income and employment</span>
                            </li>
                            <li className="flex items-start">
                                <span className="text-green-600 mr-2">•</span>
                                <span className="text-gray-700">Avoid new debt applications before loan disbursement</span>
                            </li>
                        </>
                    ) : (
                        <>
                            <li className="flex items-start">
                                <span className="text-amber-600 mr-2">•</span>
                                <span className="text-gray-700">Consider reducing the requested loan amount</span>
                            </li>
                            <li className="flex items-start">
                                <span className="text-amber-600 mr-2">•</span>
                                <span className="text-gray-700">Work on improving your credit history</span>
                            </li>
                            <li className="flex items-start">
                                <span className="text-amber-600 mr-2">•</span>
                                <span className="text-gray-700">Increase income proof or add a co-applicant</span>
                            </li>
                        </>
                    )}
                </ul>
            </div>

            {/* AI Advisor Note */}
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-4 rounded-xl">
                <p className="text-sm font-medium text-center">
                    🤖 AI-powered advisor • View detailed explanation in the AI Explanation tab
                </p>
            </div>

            {/* Action Button */}
            <div className="flex space-x-4">
                <button
                    onClick={onReset}
                    className="flex-1 bg-gray-100 text-gray-700 py-3 px-6 rounded-lg font-medium hover:bg-gray-200 transition-colors flex items-center justify-center space-x-2"
                >
                    <RefreshCw className="w-4 h-4" />
                    <span>Check Another Application</span>
                </button>
            </div>

            {/* Timestamp */}
            <p className="text-xs text-gray-400 text-center">
                Prediction made at {new Date().toLocaleTimeString()}
            </p>
        </div>
    )
}

export default PredictionResult
