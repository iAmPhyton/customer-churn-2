# Telco Customer Churn Prediction

Project Overview: 
- Customer churn (attrition) is a critical metric for subscription-based businesses. This project focuses on predicting which customers are likely to leave a telecommunications service provider. Beyond simple prediction, this project emphasizes handling class imbalance (using SMOTE) and interpretability (identifying why customers leave) to provide actionable insights for retention strategies.

The Data:  
- Source: [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- Target Variable: `Churn` (Yes/No).
- Key Features: Customer demographics, Services (Phone, Internet), and Account details (Contract type, Payment method).

Technologies & Tools: 
- Python 3.9+
- Pandas & NumPy: Data manipulation.
- Scikit-Learn: Machine learning (Random Forest, Logistic Regression).
- Imbalanced-Learn: SMOTE (Synthetic Minority Over-sampling Technique) and Pipelines.
- Seaborn & Matplotlib: Visualization.

## Business Insights:
Based on Permutation Feature Importance, the top drivers of churn were:
1.  Month-to-Month Contracts: Customers on short-term contracts are highly volatile.
    * Recommendation: Incentivize 1-year or 2-year contracts with discounts.
2.  Fiber Optic Internet: Surprisingly, these users churned frequently.
    * Hypothesis: Technical issues or high price point. A service quality audit is recommended.
3.  Electronic Check Payment: Users paying via check had higher churn rates than those with automatic credit card payments.

Author:
- Chukwuemeka Eugene Obiyo
- www.linkedin.com/chukwuemekao
- praise609@gmail.com
