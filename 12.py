import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


train = pd.read_csv("dataset/fashion-mnist_train.csv").head(10000)
test = pd.read_csv("dataset/fashion-mnist_test.csv").head(2000)

x_train = train.drop("label",axis=1).values/255.0
y_train = train["label"]

x_test = test.drop("label",axis=1).values/255.0
y_test = test["label"]


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


kernels = ['linear','poly','rbf']
acc = []

for k in kernels:
    model = SVC(kernel = k)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    a = accuracy_score(y_test,pred)
    acc.append(a)

best = np.argmax(acc)
print("Best Kernel:",kernels[best])
print("Best Accuracy:",acc[best])

plt.bar(kernels,acc)
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.title("SVM Kernel Comparison")
plt.show()

