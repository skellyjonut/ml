import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data=pd.read_csv("fashion-mnist_train.csv")

data=data.sample(n=3000,random_state=42)

X=data.drop("label",axis=1)/255.0
y=data["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

settings=[
    (0.001,64,500),
    (0.01,64,500),
    (0.001,128,1000),
    (0.01,128,1000)
]

for lr,batch,epochs in settings:
    model=MLPClassifier(
        hidden_layer_sizes=(50,),
        learning_rate_init=lr,
        batch_size=batch,
        max_iter=epochs,
        random_state=42
    )

    start=time.time()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    end=time.time()

    print("Learning Rate:",lr)
    print("Batch Size:",batch)
    print("Epochs:",epochs)
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("Training Time:",round(end-start,2),"seconds")
    print()