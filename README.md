# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the data file and import numpy, matplotlib and scipy.

2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

3.Plot the decision boundary .

4.Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Udhayanithi M
RegisterNumber: 212222220054
*/
```
```
/*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:
## DATASET:
![image](https://github.com/Prasanth9025/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343686/556446f3-c90c-40e3-b81d-bce4c39b7e8b)
## DATASET.DTYPES:
![image](https://github.com/Prasanth9025/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343686/f66a3f8a-f127-49ae-97fb-ac4f9bd04220)
## DATASET:
![image](https://github.com/Prasanth9025/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343686/64edc644-95cb-4bee-832d-1debfd3a1bf9)
## ACCURACY:
![image](https://github.com/Prasanth9025/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343686/c70df073-b610-4cb6-b962-104579b98e3e)
## Y:
![image](https://github.com/Prasanth9025/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343686/50d636a0-0a1c-49a2-8a53-6c84e407bab7)
## Y_PREDNEW:
![image](https://github.com/Prasanth9025/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343686/7d1e13a2-6403-4560-8f0f-a862900be127)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

