from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Forest.pkl')


@app.route('/')
def home():
    return render_template('Loan_Prediction.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        # Mapping form inputs to variable names
        gender = form_data.get('gender', '').lower()
        married = form_data.get('status', '').lower()
        dependents = form_data.get('dependants', '')
        education = form_data.get('education', '').lower()
        employment = form_data.get('employ', '').lower()
        annual_income = form_data.get('aincome', '')
        co_income = form_data.get('coincome', '')
        loan_amount = form_data.get('Lamount', '')
        loan_amount_term = form_data.get('Lamount_term', '')
        credit = form_data.get('credit', '')
        property_area = form_data.get('property_area', '').lower()

        # Data preprocessing
        employment = 1 if employment == 'yes' else 0
        gender = 1 if gender == 'male' else 0
        married = 1 if married == 'married' else 0
        property_area_mapping = {'rural': 0, 'semiurban': 1, 'urban': 2}
        property_area = property_area_mapping.get(property_area, 0)
        education = 0 if education == 'graduate' else 1

        # Try to convert form inputs to integers
        try:
            dependents, annual_income, co_income, loan_amount, loan_amount_term, credit = (
                int(dependents), int(annual_income), int(co_income),
                int(loan_amount), int(loan_amount_term), int(credit)
            )
        except ValueError:
            return render_template('error.html', prediction=1)

        # Prepare input array for prediction
        input_array = np.array([[gender, married, dependents, education, employment,
                                 annual_income, co_income, loan_amount, loan_amount_term,
                                 credit, property_area]])

        # Load the model and make prediction
        # model = RandomForestClassifier()
        # joblib.dump(model, 'Forest.pkl')
        prediction = model.predict(input_array)

        if prediction == 1:
            result_message = "Congratulations! You are eligible for this loan."
        else:
            result_message = "We regret to inform you that your request has not been accepted."
        # print("Result Message:", result_message)  
        return render_template('index.html', result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)
