import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv("fashion-mnist_train.csv")

data=data.sample(n=3000,random_state=42)

X=data.drop("label",axis=1)
y=data["label"]

X=X/255.0

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

for k in [3,5,7]:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    print("K=",k,"Accuracy=",accuracy)