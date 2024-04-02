# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.
2. Download and upload required csv file or dataset for predecting Employee Churn.
3. Initialize variables with required features.
4. And implement Decision tree classifier to predict Employee Churn. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Hemakesh G
RegisterNumber:  212223040064
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor, plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plot_tree(dt,feature_names=x.columns,class_names=['Salary'], filled=True)
plt.show()

```

## Output:
### 1. Head:
![Screenshot 2024-04-02 182139](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870552/ec092b75-6cc7-452c-8569-cbb76864d867)


### 2. Mean square error:
![Screenshot 2024-04-02 182035](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870552/c5b95e50-ebcc-4f69-bf48-886f4d7a4e71)


### 3. Testing of Model:
![Screenshot 2024-04-02 182116](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870552/c7e8e0be-3609-4c67-a50d-5dc3d95f34b0)


### 4. Decision Tree:
![Screenshot 2024-04-02 182020](https://github.com/HEMAKESHG/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870552/1f033144-c59f-4737-ad76-d7e5571ab786)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
