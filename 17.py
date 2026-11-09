import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/digits.csv")

x = df.drop("label",axis=1)
sc = StandardScaler()
x = sc.fit_transform(x)

vals = [2,4,6,8,10,12,16]
iner = []
silh = []

for k in vals:
    model = KMeans(n_init=10,random_state=42,n_clusters=k)
    label = model.fit_predict(x)
    i = model.inertia_
    s = silhouette_score(x,label)
    iner.append(i)
    silh.append(s)

    print("K:",k)
    print("Slihoutte:",s)
    print("Inertia:",i)

plt.figure(figsize=(5,8))
plt.plot(vals,iner)
plt.xlabel("K Value")
plt.ylabel("Inertia Score")
plt.grid()
plt.show()

plt.figure(figsize=(5,8))
plt.plot(vals,silh)
plt.xlabel("K Value")
plt.ylabel("Silhoutte Score")
plt.grid()
plt.show()