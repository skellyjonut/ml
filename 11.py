import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("Iris.csv")

X=data[["SepalLengthCm","SepalWidthCm"]]

y=(data["Species"]!="Iris-setosa").astype(int)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=SVC(kernel="linear")
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y)

w=model.coef_[0]
b=model.intercept_[0]

x=X.iloc[:,0]

y_boundary=-(w[0]*x+b)/w[1]

#upper margin: w1*x+w2*y+b=1
y_upper=-(w[0]*x+b-1)/w[1]

#Lower margin:w1*x+w2*y+b=-1
y_lower=-(w[0]*x+b+1)/w[1]

plt.plot(x,y_boundary,color="red",label="Decision boundary")
plt.plot(x,y_upper,"k--",label="Upper Margin")
plt.plot(x,y_lower,"k--",label="Lower Margin")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM on Iris Dataset")
plt.legend()
plt.show()