import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("winequality-red.csv")
print(data.columns.tolist())

# X=data.drop("quality",axis=1)
# y=data["quality"]

# X_train,X_test,y_train,y_test=train_test_split(
#     X,y,test_size=0.2,random_state=42
# )

# scaler=StandardScaler()
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)

# for layers in [(10,),(20,),(10,10)]:
#     model=MLPClassifier(
#         hidden_layer_sizes=layers,
#         max_iter=1500,
#         random_state=42
#     )

#     model.fit(X_train,y_train)
#     y_pred=model.predict(X_test)

#     print("Hidden layers:",layers)
#     print("Accuracy:",accuracy_score(y_test,y_pred))
#     print()