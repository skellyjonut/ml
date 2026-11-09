import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

df = df.drop(columns=["id","Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"M":1,"B":0})

x = df.drop("diagnosis",axis=1)
y = df["diagnosis"]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

mle = LogisticRegression(C=float("inf"),max_iter=5000)
mle.fit(x_train,y_train)
pred_mle = mle.predict(x_test)


print("MLE")
print("Accuracy:",accuracy_score(y_test,pred_mle))
print("MSE:",mean_squared_error(y_test,pred_mle))


l1 = LogisticRegression(C=1.0,solver = "liblinear",max_iter=5000,l1_ratio = 1)
l1.fit(x_train,y_train)
pred_l1 = l1.predict(x_test)


print("L1")
print("Accuracy:",accuracy_score(y_test,pred_l1))
print("MSE:",mean_squared_error(y_test,pred_l1))

l2 = LogisticRegression(C=1.0,max_iter=5000,l1_ratio=0)
l2.fit(x_train,y_train)
pred_l2 = l2.predict(x_test)

print("L2")
print("Accuracy:",accuracy_score(y_test,pred_l2))
print("MSE:",mean_squared_error(y_test,pred_l2))


