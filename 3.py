import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt


data = load_diabetes()
x = data.data
y=  data.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model = LinearRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)

print("LinearRegression")
print("R2 Score:",r2_score(y_test,pred))
print("MSE:",mean_squared_error(y_test,pred))


ridge = RidgeCV(alphas=[0.01,0.1,1,10,100],cv=5)
ridge.fit(x_train,y_train)
pred_r = ridge.predict(x_test)


print("Ridge")
print("R2 Score:",r2_score(y_test,pred_r))
print("MSE:",mean_squared_error(y_test,pred_r))

lasso = LassoCV(alphas=[0.01,0.1,1,10,100])
lasso.fit(x_train,y_train)
pred_l = lasso.predict(x_test)


print("Lasso")
print("R2 Score:",r2_score(y_test,pred_l))
print("MSE:",mean_squared_error(y_test,pred_l))

print("Coeff",lasso.coef_)


plt.scatter(y_test,pred,color="red",label="Linear Regression")
plt.scatter(y_test,pred_r,color="blue",label="RidgeCV")
plt.scatter(y_test,pred_l,color="yellow",label="LassoCV")

plt.legend()
plt.show()




