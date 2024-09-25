# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, clean it, and separate features (X) from the target variable (y).
2.Split the data into training and testing sets (e.g., 80% train, 20% test).
3.Create and fit a Logistic Regression model using the training data (X_train, y_train).
4.Make predictions on the test set (X_test) and evaluate performance with accuracy, confusion matrix, and classification report.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: muralitharan k m 
RegisterNumber: 212223040121  
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer 
data = pd.read_csv('Placement_Data.csv')
print(data.isnull().sum())
X = data.drop('status', axis=1)
y = data['status']
label_encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = label_encoder.fit_transform(X[col])
imputer = SimpleImputer(strategy='mean') 
X = imputer.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)
```

## Output:

![image](https://github.com/user-attachments/assets/f6e9138f-a86b-4d3d-af85-1fc9b674d914)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
