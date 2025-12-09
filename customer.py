import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
customer = pd.read_csv('Telco-Customer-Churn.csv')
customer.head()
customer.info() 

#data cleaning
#forcing converts to numeric
customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
#checking for null values
print(f"Nulls in TotalCharges: {customer['TotalCharges'].isnull().sum()}")
#filling nulls with 0 (since they are new customers) or completely drop them
customer['TotalCharges'].fillna(0, inplace=True)
#dropping 'customerID' as it has no predictive power
customer.drop('customerID', axis=1, inplace=True)
customer

#eda
#visualising churn count
sns.countplot(x='Churn', data=customer)
plt.title("Churn Class Distribution")
plt.show()
#visualise churn by contract type
sns.countplot(x='Contract', hue='Churn', data=customer)
plt.title("Churn by Contract Type")
plt.show()

#preprocessing
#binary encoding for the target
customer['Churn'] = customer['Churn'].map({'Yes': 1, 'No': 0})
#one hot encoding for other categoricals
customer_encoded = pd.get_dummies(customer, drop_first=True) 

#model building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#splitting data
x = customer_encoded.drop('Churn', axis=1)
y = customer_encoded['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#training the baseline model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred)) 

from imblearn.over_sampling import SMOTE
from collections import Counter

#standard splitting
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

#checking counts before SMOTE
print(f"Before SMOTE: {Counter(y_train)}")

#initialising SMOTE
smote = SMOTE(random_state=42)

#fitting SMOTE on the training data
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#checking counts after SMOTE
print(f"After SMOTE: {Counter(y_train_resampled)}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#initializing model
rf_model = RandomForestClassifier(random_state=42)
#training on the balanced SMOTE data
rf_model.fit(x_train_resampled, y_train_resampled)
#prediction on the original (unlanaced) data
y_pred = rf_model.predict(x_test)
#results
print(classification_report(y_test, y_pred)) 

