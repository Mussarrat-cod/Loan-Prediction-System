import { useState } from 'react'

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
        <form onSubmit={handleSubmit}>
            <div className="form-grid">
                {/* Gender */}
                <div className="form-group">
                    <label htmlFor="Gender">Gender</label>
                    <select
                        id="Gender"
                        name="Gender"
                        value={formData.Gender}
                        onChange={handleChange}
                        required
                    >
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                {/* Married */}
                <div className="form-group">
                    <label htmlFor="Married">Marital Status</label>
                    <select
                        id="Married"
                        name="Married"
                        value={formData.Married}
                        onChange={handleChange}
                        required
                    >
                        <option value="Yes">Married</option>
                        <option value="No">Single</option>
                    </select>
                </div>

                {/* Dependents */}
                <div className="form-group">
                    <label htmlFor="Dependents">Dependents</label>
                    <select
                        id="Dependents"
                        name="Dependents"
                        value={formData.Dependents}
                        onChange={handleChange}
                        required
                    >
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3+">3+</option>
                    </select>
                </div>

                {/* Education */}
                <div className="form-group">
                    <label htmlFor="Education">Education</label>
                    <select
                        id="Education"
                        name="Education"
                        value={formData.Education}
                        onChange={handleChange}
                        required
                    >
                        <option value="Graduate">Graduate</option>
                        <option value="Not Graduate">Not Graduate</option>
                    </select>
                </div>

                {/* Self Employed */}
                <div className="form-group">
                    <label htmlFor="Self_Employed">Employment Type</label>
                    <select
                        id="Self_Employed"
                        name="Self_Employed"
                        value={formData.Self_Employed}
                        onChange={handleChange}
                        required
                    >
                        <option value="No">Salaried</option>
                        <option value="Yes">Self-Employed</option>
                    </select>
                </div>

                {/* Property Area */}
                <div className="form-group">
                    <label htmlFor="Property_Area">Property Area</label>
                    <select
                        id="Property_Area"
                        name="Property_Area"
                        value={formData.Property_Area}
                        onChange={handleChange}
                        required
                    >
                        <option value="Urban">Urban</option>
                        <option value="Semiurban">Semi-Urban</option>
                        <option value="Rural">Rural</option>
                    </select>
                </div>

                {/* Applicant Income */}
                <div className="form-group">
                    <label htmlFor="ApplicantIncome">Applicant Income ($)</label>
                    <input
                        type="number"
                        id="ApplicantIncome"
                        name="ApplicantIncome"
                        value={formData.ApplicantIncome}
                        onChange={handleChange}
                        placeholder="5000"
                        min="0"
                        required
                    />
                </div>

                {/* Coapplicant Income */}
                <div className="form-group">
                    <label htmlFor="CoapplicantIncome">Co-applicant Income ($)</label>
                    <input
                        type="number"
                        id="CoapplicantIncome"
                        name="CoapplicantIncome"
                        value={formData.CoapplicantIncome}
                        onChange={handleChange}
                        placeholder="2000"
                        min="0"
                        required
                    />
                </div>

                {/* Loan Amount */}
                <div className="form-group">
                    <label htmlFor="LoanAmount">Loan Amount (thousands $)</label>
                    <input
                        type="number"
                        id="LoanAmount"
                        name="LoanAmount"
                        value={formData.LoanAmount}
                        onChange={handleChange}
                        placeholder="150"
                        min="0"
                        required
                    />
                </div>

                {/* Loan Term */}
                <div className="form-group">
                    <label htmlFor="Loan_Amount_Term">Loan Term (months)</label>
                    <select
                        id="Loan_Amount_Term"
                        name="Loan_Amount_Term"
                        value={formData.Loan_Amount_Term}
                        onChange={handleChange}
                        required
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
                <div className="form-group">
                    <label htmlFor="Credit_History">Credit History</label>
                    <select
                        id="Credit_History"
                        name="Credit_History"
                        value={formData.Credit_History}
                        onChange={handleChange}
                        required
                    >
                        <option value="1">Good (Meets Guidelines)</option>
                        <option value="0">Poor (Does Not Meet)</option>
                    </select>
                </div>
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? (
                    <>
                        <span className="spinner"></span>
                        <span>Analyzing...</span>
                    </>
                ) : (
                    <>
                        <span>🔍</span>
                        <span>Predict Loan Approval</span>
                    </>
                )}
            </button>
        </form>
    )
}

export default LoanForm
