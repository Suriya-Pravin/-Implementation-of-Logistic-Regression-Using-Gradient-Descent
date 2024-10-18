# Ex-6: Implementation-of-Logistic-Regression-Using-Gradient-Descent
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Import pandas module and import the required data set.<br>
<br>2)Find the null values and count them.<br>
<br>3)Count number of left values.<br>
<br>4)From sklearn import LabelEncoder to convert string values to numerical values.<br>
<br>5)From sklearn.model_selection import train_test_split.<br>
<br>6)Assign the train dataset and test dataset.<br>
<br>7)From sklearn.tree import DecisionTreeClassifier.<br>
<br>8)Use criteria as entropy.<br>
<br>9)From sklearn import metrics.<br>
<br>10)Find the accuracy of our model and predict the require values.<br>
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 212223230223
RegisterNumber:  Suriya Pravin M
*/

import pandas as pd 
import numpy as np

dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')


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


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return  -np.sum(y*np.log(h)+(1-y)*log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred =predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print('Accuracy:',accuracy)

print(y_pred)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:<br>


![1)](https://github.com/user-attachments/assets/648cdc1d-22b3-4c1a-ab19-76608cf44c11)


<br><br>

![2)](https://github.com/user-attachments/assets/7f3e860d-3d74-4bba-b298-3f0cc52e1a2b)


<br><br>
![3)](https://github.com/user-attachments/assets/4aa1793f-1575-4490-ae2a-4f1fceb120aa)


<br><br>
![4)](https://github.com/user-attachments/assets/e081df59-a4ad-4766-aba0-0f1109045a50)

<br><br>
![5)](https://github.com/user-attachments/assets/f73a1e6e-faed-48a9-8c03-f3d2a1f7810f)

<br><br>
![6)](https://github.com/user-attachments/assets/0d3cc0d6-335f-4203-ba9c-c387c2b89668)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
