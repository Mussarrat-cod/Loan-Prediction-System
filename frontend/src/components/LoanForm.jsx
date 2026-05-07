import { useState } from 'react'
import { Loader2, Search } from 'lucide-react'

const LoanForm = ({ onSubmit, loading }) => {
    const [formData, setFormData] = useState({
        Gender: 'Male',
        Married: 'Yes',
        Dependents: '0',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: '',
        CoapplicantIncome: '',
        LoanAmount: '',
        Loan_Amount_Term: '360',
        Credit_History: '1',
        Property_Area: 'Urban'
    })

    const handleChange = (e) => {
        const { name, value } = e.target
        setFormData(prev => ({
            ...prev,
            [name]: value
        }))
    }

    const handleSubmit = (e) => {
        e.preventDefault()

        // Convert numeric fields to numbers
        const processedData = {
            ...formData,
            ApplicantIncome: Number(formData.ApplicantIncome),
            CoapplicantIncome: Number(formData.CoapplicantIncome),
            LoanAmount: Number(formData.LoanAmount),
            Loan_Amount_Term: Number(formData.Loan_Amount_Term),
            Credit_History: Number(formData.Credit_History)
        }

        onSubmit(processedData)
    }

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
                {/* Gender */}
                <div className="space-y-2">
                    <label htmlFor="Gender" className="block text-sm font-medium text-gray-700">
                        Gender
                    </label>
                    <select
                        id="Gender"
                        name="Gender"
                        value={formData.Gender}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                {/* Married */}
                <div className="space-y-2">
                    <label htmlFor="Married" className="block text-sm font-medium text-gray-700">
                        Marital Status
                    </label>
                    <select
                        id="Married"
                        name="Married"
                        value={formData.Married}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="Yes">Married</option>
                        <option value="No">Single</option>
                    </select>
                </div>

                {/* Dependents */}
                <div className="space-y-2">
                    <label htmlFor="Dependents" className="block text-sm font-medium text-gray-700">
                        Dependents
                    </label>
                    <select
                        id="Dependents"
                        name="Dependents"
                        value={formData.Dependents}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3+">3+</option>
                    </select>
                </div>

                {/* Education */}
                <div className="space-y-2">
                    <label htmlFor="Education" className="block text-sm font-medium text-gray-700">
                        Education
                    </label>
                    <select
                        id="Education"
                        name="Education"
                        value={formData.Education}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="Graduate">Graduate</option>
                        <option value="Not Graduate">Not Graduate</option>
                    </select>
                </div>

                {/* Self Employed */}
                <div className="space-y-2">
                    <label htmlFor="Self_Employed" className="block text-sm font-medium text-gray-700">
                        Employment Type
                    </label>
                    <select
                        id="Self_Employed"
                        name="Self_Employed"
                        value={formData.Self_Employed}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="No">Salaried</option>
                        <option value="Yes">Self-Employed</option>
                    </select>
                </div>

                {/* Property Area */}
                <div className="space-y-2">
                    <label htmlFor="Property_Area" className="block text-sm font-medium text-gray-700">
                        Property Area
                    </label>
                    <select
                        id="Property_Area"
                        name="Property_Area"
                        value={formData.Property_Area}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="Urban">Urban</option>
                        <option value="Semiurban">Semi-Urban</option>
                        <option value="Rural">Rural</option>
                    </select>
                </div>

                {/* Applicant Income */}
                <div className="space-y-2">
                    <label htmlFor="ApplicantIncome" className="block text-sm font-medium text-gray-700">
                        Applicant Income ($)
                    </label>
                    <input
                        type="number"
                        id="ApplicantIncome"
                        name="ApplicantIncome"
                        value={formData.ApplicantIncome}
                        onChange={handleChange}
                        placeholder="5000"
                        min="0"
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    />
                </div>

                {/* Coapplicant Income */}
                <div className="space-y-2">
                    <label htmlFor="CoapplicantIncome" className="block text-sm font-medium text-gray-700">
                        Co-applicant Income ($)
                    </label>
                    <input
                        type="number"
                        id="CoapplicantIncome"
                        name="CoapplicantIncome"
                        value={formData.CoapplicantIncome}
                        onChange={handleChange}
                        placeholder="2000"
                        min="0"
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    />
                </div>

                {/* Loan Amount */}
                <div className="space-y-2">
                    <label htmlFor="LoanAmount" className="block text-sm font-medium text-gray-700">
                        Loan Amount (thousands $)
                    </label>
                    <input
                        type="number"
                        id="LoanAmount"
                        name="LoanAmount"
                        value={formData.LoanAmount}
                        onChange={handleChange}
                        placeholder="150"
                        min="0"
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    />
                </div>

                {/* Loan Term */}
                <div className="space-y-2">
                    <label htmlFor="Loan_Amount_Term" className="block text-sm font-medium text-gray-700">
                        Loan Term (months)
                    </label>
                    <select
                        id="Loan_Amount_Term"
                        name="Loan_Amount_Term"
                        value={formData.Loan_Amount_Term}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="12">12 months (1 year)</option>
                        <option value="36">36 months (3 years)</option>
                        <option value="60">60 months (5 years)</option>
                        <option value="84">84 months (7 years)</option>
                        <option value="120">120 months (10 years)</option>
                        <option value="180">180 months (15 years)</option>
                        <option value="240">240 months (20 years)</option>
                        <option value="360">360 months (30 years)</option>
                        <option value="480">480 months (40 years)</option>
                    </select>
                </div>

                {/* Credit History */}
                <div className="space-y-2">
                    <label htmlFor="Credit_History" className="block text-sm font-medium text-gray-700">
                        Credit History
                    </label>
                    <select
                        id="Credit_History"
                        name="Credit_History"
                        value={formData.Credit_History}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                    >
                        <option value="1">Good (Meets Guidelines)</option>
                        <option value="0">Poor (Does Not Meet)</option>
                    </select>
                </div>
            </div>

            <button
                type="submit"
                disabled={loading}
                className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 focus:ring-4 focus:ring-indigo-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2"
            >
                {loading ? (
                    <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Analyzing with AI...</span>
                    </>
                ) : (
                    <>
                        <Search className="w-5 h-5" />
                        <span>Predict Loan Approval</span>
                    </>
                )}
            </button>
        </form>
    )
}

export default LoanForm
