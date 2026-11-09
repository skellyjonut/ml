import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("dataset/adult.csv")

x = df.drop('income',axis=1)
y = df['income'].map({'<=50K':0,">50K":1})


x = pd.get_dummies(x,dtype=int)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

m1 = LogisticRegression(max_iter=1000)
m1.fit(x_train,y_train)
pred1 = m1.predict(x_test)

print("Logistic Regression")
print("Accuracy:",accuracy_score(y_test,pred1))
print("F1 Score:",f1_score(y_test,pred1))

m2 = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=42)
m2.fit(x_train,y_train)
pred2 = m2.predict(x_test)

print("Decision Treee")
print("Accuracy:",accuracy_score(y_test,pred2))
print("F1 Score:",f1_score(y_test,pred2))



