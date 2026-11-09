import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv("dataset/winequality-red.csv")
x = df.drop("quality",axis=1)
y = df["quality"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
label = []
acc = []
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

for l in [(10,),(20,),(10,10)]:
    model = MLPClassifier(hidden_layer_sizes = l,random_state=42,max_iter=1500)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    
    score = accuracy_score(y_test, pred)
    label.append(str(l))
    acc.append(score)
    print("Hidden Layers:",l)
    print("Accuracy:",accuracy_score(y_test,pred))
    print()


plt.bar(label,acc)
plt.xlabel("Hidden Layer")
plt.ylabel("Accuracy")
plt.title("MLP Architecture")
plt.show()


