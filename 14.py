import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train = pd.read_csv("dataset/fashion-mnist_train.csv").head(10000)
test = pd.read_csv("dataset/fashion-mnist_test.csv").head(2000)

x_train = train.drop("label",axis=1)/255.0
x_test = test.drop("label",axis=1)/255.0
y_train = train["label"]
y_test = test["label"]

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
acc  = []
activation = []

for act in ['relu','tanh','logistic']:
    model = MLPClassifier(random_state=42,activation=act,hidden_layer_sizes=(50,),max_iter=1500)
    st = time.time()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    et = time.time()
    a = accuracy_score(y_test,pred)
    activation.append(act)
    acc.append(a)
    print("Accuracy:",accuracy_score(y_test,pred))
    print("Time Taken:",round(et-st,2))

plt.bar(activation,acc)
plt.xlabel("Activation")
plt.ylabel("Accuracy")
# plt.legend()
plt.show()



