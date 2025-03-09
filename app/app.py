from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load("artifacts/credit_risk_cat_model.pkl")  
scaler = joblib.load("artifacts/scaler.pkl") 

# Categorical mappings
home_ownership_map = {'MORTGAGE': 0, 'OTHER': 1, 'OWN': 2, 'RENT': 3}
loan_intent_map = {'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2, 'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5}
cb_person_default_on_file_map = {'N': 0, 'Y': 1}

@app.route('/')
def home():
    return render_template('index.html', 
                           home_ownership_map=home_ownership_map,
                           loan_intent_map=loan_intent_map,
                           cb_person_default_on_file_map=cb_person_default_on_file_map)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract numerical input
        person_income = float(data['person_income'])
        loan_amnt = float(data['loan_amnt'])
        person_emp_length = float(data['person_emp_length'])
        cb_person_cred_hist_length = float(data['cb_person_cred_hist_length'])
        
        # Extract categorical input
        person_home_ownership = int(data['person_home_ownership'])
        loan_intent = int(data['loan_intent'])
        cb_person_default_on_file = int(data['cb_person_default_on_file'])
        
        # Prepare DataFrame
        input_data = pd.DataFrame({
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership],
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent],
            'total_repayment': [loan_amnt * (1 + 11) ** 12],
            'cb_person_default_on_file': [cb_person_default_on_file],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length]
        })
        
        # Normalize numerical columns
        numerical_cols_to_normalize = ['person_income', 'person_emp_length', 'total_repayment', 'cb_person_cred_hist_length']
        input_data[numerical_cols_to_normalize] = scaler.transform(input_data[numerical_cols_to_normalize])
        
        # Predict
        credit_risk_prediction = model.predict_proba(input_data)
        probability_of_default = credit_risk_prediction[0][1]
        risk_classification = "High-Risk" if probability_of_default >= 0.5 else "Low-Risk"
        
        return jsonify({
            "probability_of_default": round(probability_of_default, 2),
            "risk_classification": risk_classification
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
