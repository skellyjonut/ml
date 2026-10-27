import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("adult.csv")
data.columns = data.columns.str.strip()

data = data[["age", "education", "marital-status","hours-per-week", "income"]]

data["income"] = data["income"].astype(str).str.strip()
data["income"] = (data["income"] == ">50K").astype(int)

X=data.drop("income",axis=1)
y=data["income"]

X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)

pred1=lr.predict(X_test)

dt=DecisionTreeClassifier(max_depth=5)
dt.fit(X_train,y_train)
pred2=dt.predict(X_test)

def evaluate(name, actual, predicted):
    print("\n", name)
    print("Accuracy:", accuracy_score(actual, predicted))
    print("Precision:", precision_score(actual, predicted))
    print("Recall:", recall_score(actual, predicted))
    print("F1-Score:", f1_score(actual, predicted))

# Compare models
evaluate("Logistic Regression", y_test, pred1)
evaluate("Decision Tree", y_test, pred2)