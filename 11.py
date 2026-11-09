import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Iris.csv")

x = df[["PetalLengthCm","PetalWidthCm"]]
y = (df["Species"] != "Iris-setosa").astype(int)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = SVC(kernel = 'linear',C=1)
model.fit(x_train,y_train)
pred = model.predict(x_test)

print("Accuracy",accuracy_score(y_test,pred))


# plt.figure(figsize=(8,6))

plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],label="Setosa")
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],label="Non - Setosa")

w = model.coef_[0]
b = model.intercept_[0]

x_val = np.linspace(x_train[:,0].min(),x_train[:,0].max(),100)
y_val = -(w[0]*x_val + b)/w[1]
y_valplus = -(w[0]*x_val + b+1)/w[1]
y_valneg = -(w[0]*x_val + b-1)/w[1]

plt.plot(x_val,y_val,label="Decision Boundary")
plt.plot(x_val,y_valplus,'--',label="Margin")
plt.plot(x_val,y_valneg,'--',label="Margin")

plt.xlabel("Scaled Petal Length")
plt.ylabel("Scaled Petal Width")
plt.title("Linear SVM on Iris")
plt.legend()
plt.show()
