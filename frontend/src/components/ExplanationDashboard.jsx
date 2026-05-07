import React from 'react';
import { CheckCircle, XCircle, AlertTriangle, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ExplanationDashboard = ({ explanation, loading }) => {
  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-4"></div>
        <div className="h-32 bg-gray-200 rounded mb-4"></div>
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="text-center py-8">
        <Info className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-500">No explanation data available</p>
      </div>
    );
  }

  const { prediction, confidence_score, status, top_positive_factors, top_negative_factors, explanation_text, risk_assessment } = explanation;

  // Prepare data for feature importance chart
  const chartData = [
    ...top_positive_factors.map(factor => ({
      name: factor.feature_name,
      value: Math.abs(factor.contribution),
      impact: 'positive',
      contribution: factor.contribution,
      explanation: factor.explanation
    })),
    ...top_negative_factors.map(factor => ({
      name: factor.feature_name,
      value: Math.abs(factor.contribution),
      impact: 'negative',
      contribution: factor.contribution,
      explanation: factor.explanation
    }))
  ];

  const getRiskColor = (level) => {
    switch (level) {
      case 'Low': return 'text-green-600 bg-green-50';
      case 'Medium': return 'text-yellow-600 bg-yellow-50';
      case 'High': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getImpactIcon = (impact) => {
    return impact === 'positive' ? 
      <TrendingUp className="w-4 h-4 text-green-500" /> : 
      <TrendingDown className="w-4 h-4 text-red-500" />;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Decision Header */}
      <div className={`text-center p-6 rounded-xl ${
        status === 'Approved' ? 'bg-gradient-to-r from-green-50 to-emerald-50' : 'bg-gradient-to-r from-red-50 to-pink-50'
      }`}>
        <div className="flex justify-center mb-4">
          {status === 'Approved' ? (
            <CheckCircle className="w-16 h-16 text-green-500" />
          ) : (
            <XCircle className="w-16 h-16 text-red-500" />
          )}
        </div>
        <h2 className={`text-2xl font-bold mb-2 ${
          status === 'Approved' ? 'text-green-800' : 'text-red-800'
        }`}>
          Loan {status}
        </h2>
        <p className="text-gray-600 mb-4">{explanation_text?.decision_explanation}</p>
        <div className="flex items-center justify-center space-x-4">
          <span className="text-sm text-gray-500">Confidence:</span>
          <span className="text-lg font-semibold text-gray-800">{confidence_score}</span>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(risk_assessment?.risk_level)}`}>
            {risk_assessment?.risk_level} Risk
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-lg">
        <p className="text-blue-800 text-sm">{explanation_text?.summary}</p>
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Feature Impact Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip 
              formatter={(value, name) => [value.toFixed(3), 'Impact Score']}
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Bar dataKey="value" radius={[8, 8, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.impact === 'positive' ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Factors Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Positive Factors */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold mb-4 text-green-800 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2" />
            Strength Factors
          </h3>
          <div className="space-y-3">
            {top_positive_factors.map((factor, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
                <div className="flex-shrink-0 mt-1">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-800 text-sm">{factor.feature_name}</span>
                    <span className="text-xs text-green-600 font-medium">
                      +{factor.contribution.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600">{factor.explanation}</p>
                  <div className="text-xs text-gray-500 mt-1">
                    Value: {factor.value}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Negative Factors */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold mb-4 text-red-800 flex items-center">
            <TrendingDown className="w-5 h-5 mr-2" />
            Risk Factors
          </h3>
          <div className="space-y-3">
            {top_negative_factors.map((factor, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg">
                <div className="flex-shrink-0 mt-1">
                  <AlertTriangle className="w-4 h-4 text-red-500" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-800 text-sm">{factor.feature_name}</span>
                    <span className="text-xs text-red-600 font-medium">
                      {factor.contribution.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600">{factor.explanation}</p>
                  <div className="text-xs text-gray-500 mt-1">
                    Value: {factor.value}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Risk Assessment</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-800">{risk_assessment?.confidence_score}</div>
            <div className="text-xs text-gray-500 mt-1">Confidence</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-800">{risk_assessment?.adjusted_probability}</div>
            <div className="text-xs text-gray-500 mt-1">Approval Probability</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{risk_assessment?.strength_factors_count}</div>
            <div className="text-xs text-gray-500 mt-1">Strength Factors</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{risk_assessment?.risk_factors_count}</div>
            <div className="text-xs text-gray-500 mt-1">Risk Factors</div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {explanation_text?.recommendations && explanation_text.recommendations.length > 0 && (
        <div className="bg-amber-50 border-l-4 border-amber-400 p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-3 text-amber-800 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2" />
            Recommendations
          </h3>
          <ul className="space-y-2">
            {explanation_text.recommendations.map((rec, index) => (
              <li key={index} className="flex items-start space-x-2 text-sm text-amber-800">
                <span className="text-amber-600 mt-1">•</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ExplanationDashboard;
