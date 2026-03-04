# 📊 Customer Churn Analysis & Prediction System

This project is a complete Machine Learning-based Customer Churn Prediction system deployed using Flask.

It helps businesses predict whether a customer will churn (leave) or stay, based on key features such as tenure, monthly charges, contract type, payment method, and internet service.

---

## 🚀 Project Features

- Data Cleaning and Preprocessing
- Label Encoding and One-Hot Encoding
- Model Training:
  - Logistic Regression
  - Tuned Random Forest (GridSearchCV)
- Model Evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC Score
- ROC Curve Comparison
- Feature Importance Analysis
- Customer Segmentation Analysis
- Interactive Web Application using Flask
- Real-time Prediction Interface
- Data Visualizations stored in static folder

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Flask
- HTML & CSS

---

## 📂 Project Structure

- train_model.py → Model training and visualization
- churn.py → Flask web application
- templates/ → HTML files
- static/ → Images and plots
- churn_model.pkl → Saved trained model
- model_columns.pkl → Model feature columns

---

## ▶ How to Run the Project

1. Install dependencies:
pip install -r requirements.txt
2. Train the model:
python train_model.py
3. Run the Flask app:
python churn.py
4. Open browser:
http://127.0.0.1:5000


---

## 🎯 Purpose of the Project

This project demonstrates:
- End-to-end Machine Learning workflow
- Model optimization
- Deployment using Flask
- Business problem solving using Data Science
- Real-world predictive analytics application

---

## 👩‍💻 Author

Developed as a Machine Learning and Web Deployment project.
