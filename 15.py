import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train = pd.read_csv("dataset/fashion-mnist_train.csv").head(5000)
test = pd.read_csv("dataset/fashion-mnist_test.csv").head(1000)

x_train = train.drop("label",axis=1)/255.0
x_test = test.drop("label",axis=1)/255.0
y_train = train["label"]
y_test = test["label"]

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
acc  = []
activation = []

set = [(0.001,64,500),(0.01,64,500),(0.001,128,1000),(0.01,128,1000)]
acc = []
prp = []
for lr,batch,it in set:
    model = MLPClassifier(learning_rate_init = lr,batch_size=batch,max_iter=it,hidden_layer_sizes=(50,),random_state=42)
    st = time.time()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    et = time.time()
    a = accuracy_score(y_test,pred)
    acc.append(a)
    lbl = "LearningRate:"+str(lr)+" Batch"+str(batch)+" Iteration"+str(it)
    prp.append(lbl)
    print("Classifications:")
    print("LearningRate:",lr)
    print("Batch_size:",batch)
    print("Epoch:",it)
    print("Accuracy:",a)
    print("Time:",round(et-st,2))

plt.bar(prp,acc)
plt.xlabel("Specifications")
plt.ylabel("Accuracy")
plt.show()
