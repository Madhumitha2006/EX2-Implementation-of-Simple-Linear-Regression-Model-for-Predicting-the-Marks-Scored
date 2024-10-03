# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: Madhu Mitha V 
RegisterNumber: 2305002013
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train,lr.predict(x_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')  
```

## Output:

![image](https://github.com/user-attachments/assets/edd20ca5-f4e8-4bbf-87f4-3c17aabf686f)
![image](https://github.com/user-attachments/assets/680bf797-806c-4312-b833-6342ca7190ee)
![image](https://github.com/user-attachments/assets/19fb9e8e-1e38-4b3b-be7b-43e1f80179fc)
![image](https://github.com/user-attachments/assets/3b93c912-4782-4448-a94d-44ef0356601e)
![image](https://github.com/user-attachments/assets/7292fdb5-4882-4ce9-988a-28d657971a88)
![image](https://github.com/user-attachments/assets/b808c25a-bf91-43ca-b808-4cb78c893b32)
![image](https://github.com/user-attachments/assets/631a73e5-16e1-4807-a805-d8a01616336b)
![image](https://github.com/user-attachments/assets/96480c6e-00fa-41df-bc56-9ad72f482a91)





## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
