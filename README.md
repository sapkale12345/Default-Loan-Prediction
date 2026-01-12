# Loan Default Prediction using Machine Learning

## Project Overview
This project focuses on predicting loan default risk using statistical analysis and machine learning techniques.  
The objective is to identify high-risk borrowers while prioritizing recall for defaulters, as missing a defaulter can lead to significant financial loss for lending institutions.

The project includes extensive exploratory data analysis (EDA), handling of class imbalance using resampling techniques, model comparison, threshold tuning, and final model selection.

---

## Dataset Information
- Dataset Name: Loan Default Dataset  
- Observations: 65,259  
- Features: 16 (Numerical + Categorical)  
- Target Variable: `Default` (0 = Non-Default, 1 = Default)

### Feature Types
- Numerical: Age, Income, LoanAmount, CreditScore, InterestRate, LoanTerm, DTIRatio, MonthsEmployed, NumCreditLines  
- Categorical: Education, EmploymentType, MaritalStatus, LoanPurpose, HasMortgage, HasDependents, HasCoSigner  

---

## Tools & Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Imbalanced-learn  
- Jupyter Notebook / Google Colab  

---

## Exploratory Data Analysis (EDA)
- Class imbalance analysis (Default vs Non-Default)
- Outlier detection using IQR and boxplots
- Credit score-wise default rate analysis
- Age group-wise default behavior analysis
- DTI ratio comparison for defaulters and non-defaulters
- Correlation analysis using heatmap (label-encoded categorical variables)

### Key EDA Insights
- Default rate decreases as credit score increases (inverse relationship)
- Younger borrowers have a higher default rate
- Defaulters exhibit higher DTI ratios on average
- Dataset is highly imbalanced (~11.6% defaulters)

---

## Data Preprocessing
- Removed unnecessary identifier column (`LoanID`)
- No missing values detected
- One-hot encoding for categorical variables
- Standard scaling for numerical features
- ColumnTransformer for unified preprocessing pipeline

---

## Handling Class Imbalance
- Applied **SMOTEENN** (SMOTE + Edited Nearest Neighbours)
- Used class-weighted models where applicable
- Focused on recall for defaulters instead of accuracy

---

## Models Implemented
- Logistic Regression + SMOTEENN  
- Random Forest + SMOTEENN  
- Gradient Boosting + SMOTEENN  
- Balanced Random Forest  

---

## Model Evaluation Metrics
- Confusion Matrix  
- Precision, Recall, F1-score  
- ROC-AUC Score  
- Threshold tuning for improved recall  

---

## Model Performance Summary

| Model | ROC-AUC | Recall (Default) | Accuracy |
|------|--------|-----------------|----------|
| Logistic Regression | 0.746 | 0.77 | 0.59 |
| Random Forest | 0.738 | 0.55 | 0.75 |
| Gradient Boosting | 0.742 | 0.62 | 0.71 |
| **Balanced Random Forest (Final) | 0.744 | 0.74 | 0.63 |

---

## Final Model Selection
**Balanced Random Forest** was selected as the final model due to:
- Strong performance on imbalanced data
- High recall for defaulters
- Stable ROC-AUC after threshold tuning

Threshold was optimized to **0.48** to reduce false negatives.

---

## Final Results (Balanced Random Forest)
- ROC-AUC Score: 0.744
- Recall (Default): 0.74
- Accuracy: 0.63
- Confusion Matrix visualized for performance interpretation


