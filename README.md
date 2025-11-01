# 💰 Loan Approval Risk Prediction Using Machine Learning


## 📘 Overview
This project aims to predict the risk of loan approval using machine learning techniques.
It analyzes applicant demographic, financial, and credit-related features to help financial institutions make data-driven loan decisions while minimizing default risk.

---

## 🎯 Objective
To develop and evaluate a machine learning model that accurately classifies loan applications as “Approved” or “Not Approved” based on applicant profiles and loan attributes.

---

## ⚙️ Workflow

1. **Data Loading & Exploration**
   - Loaded JSON dataset using Pandas.
   - Performed exploratory data analysis (EDA) to understand variable distributions and correlations.

2. **Data Preprocessing**
   - Cleaned categorical data and removed special characters.
   - Applied feature mapping (e.g., Married/Single → 1/0).
   - Performed one-hot encoding for categorical variables.
   - Standardized numerical features using StandardScaler.

3. **Model Development**
   - Trained multiple machine learning models:
     - Random Forest Classifier
     - K-Nearest Neighbors (KNN)
     - XGBoost Classifier
   - Compared their performance to select the best model.

4. **Model Evaluation**
   - Evaluated models using the following metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC Score
     - Confusion Matrix
   - Visualized results using Matplotlib and Seaborn.

5. **Model Optimization**
   - Tuned model hyperparameters for performance improvement.
   - Selected the final model based on the highest ROC-AUC and F1-score.

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Preprocessing | StandardScaler, One-Hot Encoding |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |
| Environment | Jupyter Notebook |

---

## 🚀 Results
  - **Best Model:** XGBoost Classifier
  - **Performance:** Achieved high accuracy and AUC on test data.
  - **Key Insight:** Applicant income, credit history, and marital status were strong predictors of loan approval.

## 📊 Visualizations
  - Correlation heatmap for feature relationships
  - Distribution plots for key numerical attributes
  - Confusion matrix and ROC curve for model evaluation

---

## 📁 Project Structure
```
Loan_Approval_Risk_Prediction/
│
├── loan_approval_risk_prediction.ipynb  
├── loan_approval_dataset.json                                  
└── README.md                     
```

---

### 🧾 Requirements

Install dependencies using:
  - pip install pandas numpy scikit-learn xgboost matplotlib seaborn

---

## 🏁 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Loan-Approval-Risk-Prediction.git
   cd Loan-Approval-Risk-Prediction

2. **Open and run the notebook**
   ```bash
   jupyter notebook loan_approval_risk_prediction.ipynb

3. **Run the cells sequentially to reproduce the analysis.**

---

## 📈 Future Improvements
  - Add feature selection and SHAP interpretability.
  - Deploy the model using Flask / Streamlit for real-time predictions.
  - Integrate a database (e.g., MySQL) for storing new applicant data.

---

## 👨‍💻 Author
  **Created by AY** </br>
  Data Science & Machine Learning Enthusiast
