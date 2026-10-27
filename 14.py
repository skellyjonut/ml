import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

data=pd.read_csv("mnist_train.csv")

data=data.sample(n=5000,random_state=42)

X=data.drop("label",axis=1)
y=data["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

for activation in ["logistic","relu","tanh"]:
    model=MLPClassifier(
        hidden_layer_sizes=(50,),
        activation=activation,
        max_iter=1500,
        random_state=42
    )

    start=time.time()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    end=time.time()

    print("Activation:",activation)
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("Training time:",round(end-start,2),"seconds")
    print()