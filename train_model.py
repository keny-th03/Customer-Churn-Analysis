import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

#Create sttic/plots folder automari

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

#load dataset
customer_churn = pd.read_csv("Telco_Customer_Churn_Dataset.csv")

#Data Cleaning
# Handling missing values
customer_churn['TotalCharges'] = pd.to_numeric(customer_churn['TotalCharges'], errors='coerce')

# Checking missing values
customer_churn.isnull().sum()

# Fill missing TotalCharges with median
customer_churn['TotalCharges'].fillna(customer_churn['TotalCharges'].median(), inplace=True)

# Encode Categorical Variables
# Encode target
le = LabelEncoder()
customer_churn['Churn'] = le.fit_transform(customer_churn['Churn']) # Yes= 1, No=0

# Drop customerID
customer_churn.drop('customerID', axis=1, inplace=True)

# One-hot encoding
customer_churn = pd.get_dummies(customer_churn, drop_first=True)

# Train-Test Split
X = customer_churn.drop('Churn', axis=1)
y = customer_churn['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)
rf = grid_rf.best_estimator_

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Model Evaluation
print("\nLogistic Regression Metrics")
print("Accuracy :", round(accuracy_score(y_test, y_pred_lr), 2))
print("Precision:", round(precision_score(y_test, y_pred_lr), 2))
print("Recall   :", round(recall_score(y_test, y_pred_lr), 2))
print("F1 Score :", round(f1_score(y_test, y_pred_lr), 2))
print("AUC      :", round(roc_auc_score(y_test, y_prob_lr), 2))

print("\nTuned Random Forest Metrics")
print("Accuracy :", round(accuracy_score(y_test, y_pred_rf), 2))
print("Precision:", round(precision_score(y_test, y_pred_rf), 2))
print("Recall   :", round(recall_score(y_test, y_pred_rf), 2))
print("F1 Score :", round(f1_score(y_test, y_pred_rf), 2))
print("AUC      :", round(roc_auc_score(y_test, y_prob_rf), 2))

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# Save Best Model (Random Forest)
with open("churn_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Save Column Names
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

#Feature Analysis with visualization
# Churn by Gender, Partner, Dependents
plt.figure(figsize=(5,3))
sns.countplot(
    data=customer_churn,
    x='gender_Male',
    hue='Churn',
    palette={0: 'blue', 1: 'red'}
)

plt.xticks([0, 1], ['Female', 'Male'])
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Customer Count")
plt.savefig("static/Churn_by_Gender.png", bbox_inches="tight")
plt.close()

# Tenure vs Churn
plt.figure(figsize=(6,4))
sns.histplot(
    customer_churn[customer_churn['Churn']==1]['tenure'], 
    bins=30, 
    color='red', 
    label='Churned', kde=True
)
sns.histplot(
    customer_churn[customer_churn['Churn']==0]['tenure'], 
    bins=30, 
    color='green', 
    label='Stayed', 
    kde=True
)
plt.legend()
plt.title("Tenure Distribution vs Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Customer Count")
plt.savefig("static/Tenure_Distribution_vs_Churn.png", bbox_inches="tight")
plt.close()

# Contract type & Payment method
# Contract Type column
customer_churn['Contract_Type'] = 'Month-to-month'  # default
if 'Contract_One year' in customer_churn.columns:
    customer_churn.loc[customer_churn['Contract_One year']==1, 'Contract_Type'] = 'One year'
if 'Contract_Two year' in customer_churn.columns:
    customer_churn.loc[customer_churn['Contract_Two year']==1, 'Contract_Type'] = 'Two year'

# Payment Method column
customer_churn['PaymentMethod'] = 'Other'  # default
payment_map = {
    'PaymentMethod_Electronic check': 'Electronic check',
    'PaymentMethod_Mailed check': 'Mailed check',
    'PaymentMethod_Bank transfer (automatic)': 'Bank transfer',
    'PaymentMethod_Credit card (automatic)': 'Credit card'
}

for col, name in payment_map.items():
    if col in customer_churn.columns:
        customer_churn.loc[customer_churn[col]==1, 'PaymentMethod'] = name

# Group by Contract + Payment
churn_summary = customer_churn.groupby(['Contract_Type', 'PaymentMethod'])['Churn'].mean().reset_index()
churn_summary['Churn'] = churn_summary['Churn'] * 100  # convert to percentage

# Set style and figure size
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))

# Grouped bar chart
sns.barplot(data=churn_summary, x='Contract_Type', y='Churn', hue='PaymentMethod', palette='pastel')

# Titles and labels
plt.title("Customer Churn % by Contract Type and Payment Method", fontsize=14)
plt.xlabel("Contract Type")
plt.ylabel("Churn (%)")
plt.ylim(0, 100)
plt.legend(title="Payment Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("static/Contract_Type_and_Payment_Method.png", bbox_inches="tight")
plt.close()

# Customer Segmentation based on tenure, monthly charges and contract type
# Tenure segments (example: 0-12 months, 13-24, etc.)
tenure_bins = [0, 12, 24, 48, 60, 72]
tenure_labels = ['0-12', '13-24', '25-48', '49-60', '61-72']
customer_churn['Tenure_Segment'] = pd.cut(
    customer_churn['tenure'], 
    bins=tenure_bins, 
    labels=tenure_labels, 
    include_lowest=True)

# Monthly charges segments (low, medium, high)
monthly_bins = [0, 35, 70, 150]  # adjust based on data distribution
monthly_labels = ['Low', 'Medium', 'High']
customer_churn['Monthly_Segment'] = pd.cut(
    customer_churn['MonthlyCharges'], 
    bins=monthly_bins, 
    labels=monthly_labels, 
    include_lowest=True
)

segment_churn = customer_churn.groupby(['Contract_Type', 'Tenure_Segment', 'Monthly_Segment'])['Churn'].mean().reset_index()
segment_churn['Churn'] = segment_churn['Churn'] * 100  # convert to %

# Visualization churn by Segment
plt.figure(figsize=(12,6))
sns.heatmap(
    segment_churn.pivot_table(index='Tenure_Segment', columns='Monthly_Segment', values='Churn', aggfunc='mean'),
    annot=True, fmt=".1f", cmap='PiYG'
)
plt.title('Churn % by Tenure and Monthly Charges')
plt.ylabel('Tenure Segment')
plt.xlabel('Monthly Charges Segment')
plt.savefig("static/Tenure_and_Monthly_Charges.png", bbox_inches="tight")
plt.close()

# ROC Curve Comparison
plt.figure(figsize=(6,5))
plt.plot(fpr_lr, tpr_lr, label=f"LR AUC = {roc_auc_score(y_test, y_prob_lr):.2f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC = {roc_auc_score(y_test, y_prob_rf):.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("static/ROC_Curve-Tuned_Random_Forest.png", bbox_inches="tight")
plt.close()

# Feature Importance 
# Logistic Regression Coefficients
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

colors = np.where(coef['Coefficient']>0, 'red', 'green')
plt.figure(figsize=(8,6))
plt.barh(coef["Feature"], coef["Coefficient"], color=colors)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(0, color="black")  # Center Line
plt.gca().invert_yaxis()
plt.title("Logistic Regression Feature Importance")
plt.tight_layout()
plt.savefig("static/Logistic_Regression_Feature_Importance.png", bbox_inches="tight")
plt.close()

# Random Forest Importance
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8,6))
plt.barh(feature_importance.index, feature_importance.values, color="purple")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("static/Random_Forest_Feature_Importance.png", bbox_inches="tight")
plt.close()

print("Model and Columns Saved Successfully!")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
