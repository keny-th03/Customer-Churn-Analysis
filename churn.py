# churn.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    tenure = float(request.form["tenure"])
    monthly_charges = float(request.form["MonthlyCharges"])
    total_charges = float(request.form["TotalCharges"])
    contract = request.form["Contract"]
    payment_method = request.form["PaymentMethod"]
    internet_service = request.form["InternetService"]
    
    # Create dictionary with default 0 values
    input_data = dict.fromkeys(model_columns, 0)
    
    # Numerical values
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # One-hot encoded fields
    contract_col = f"Contract_{contract}"
    payment_col = f"PaymentMethod_{payment_method}"
    internet_col = f"InternetService_{internet_service}"

    if contract_col in input_data:
        input_data[contract_col] = 1

    if payment_col in input_data:
        input_data[payment_col] = 1

    if internet_col in input_data:
        input_data[internet_col] = 1
        
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    result = "Customer Will Churn" if prediction == 1 else "Customer Will Stay"
    
    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability*100,2)
    )

@app.route("/visuals")
def visuals():
    return render_template("visuals.html")

if __name__ == "__main__":
    app.run(debug=True)