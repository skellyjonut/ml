import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

data=pd.read_csv("fashion-mnist_train.csv")
data=data.sample(n=2000,random_state=42)

X=data.drop("label",axis=1)
y=data["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

for kernel in ["linear","poly","rbf"]:
    model=SVC(kernel=kernel)
    start=time.time()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    end=time.time()
    accuracy=accuracy_score(y_test,y_pred)

    print("Kernel:",kernel)
    print("Accuracy:",accuracy)
    print("Training and prediction time:",round(end-start,2),"seconds")
    print()