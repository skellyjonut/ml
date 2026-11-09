import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,accuracy_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pima-indians-diabetes.csv")

x = df.iloc[:,:-1]
y = df.iloc[:,-1]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


print("Without Feature scaling")
model = LogisticRegression(max_iter = 1000)
model.fit(x_train,y_train)
pred = model.predict(x_test)

print("F1 Score:",f1_score(y_test,pred))


print("With Feature Scaling")
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model1 = LogisticRegression(max_iter=1000)
model1.fit(x_train,y_train)
pred1 = model1.predict(x_test)

print("F1 Score:",f1_score(y_test,pred))


