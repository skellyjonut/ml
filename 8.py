import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train = pd.read_csv("dataset/fashion-mnist_train.csv").head(10000)
test = pd.read_csv("dataset/fashion-mnist_test.csv").head(2000)


x_train = train.drop('label',axis=1).values/255.0
y_train = train['label'].values

x_test = test.drop('label',axis=1).values/255.0
y_test=  test['label'].values


k = [1,3,5,7]
acc = []

for k_val in k:
    knn = KNeighborsClassifier(n_neighbors=k_val)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    a = accuracy_score(y_test,pred)
    acc.append(a)

    print('K=',k_val)
    print("Accuracy:",a)
    print()

best = np.argmax(acc)
print("Best K = ",k[best])
print("Best Acc = ",acc[best])


plt.plot(k,acc,marker='o')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN perfomance on Dif K values")
plt.xticks(k)
plt.grid()
plt.show()

