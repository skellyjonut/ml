import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/housing.csv")
x = df['total_rooms'].values.reshape(-1,1)
y = df['median_house_value'].values/100000

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

m = 0
c = 0
lr = 0.01
iter = 10000
n = len(x_train)

for i in range(iter):
    y_pred = m*x_train + c
    e = y_pred - y_train.reshape(-1,1)

    dm = np.mean(e*x_train)
    dc = np.mean(e)

    m = m-lr*dm
    c = c-lr*dc

y_pred_gd = m*x_test + c

print("Gradient Descent")
print("Slope:",m)
print("Intercept:",c)
print("MSE:",mean_squared_error(y_test,y_pred_gd))
print("R2:",r2_score(y_test,y_pred_gd))

x_train_b = np.c_[np.ones(n),x_train]
x_test_b = np.c_[np.ones(len(x_test)),x_test]
theta = np.linalg.pinv(x_train_b.T@x_train_b)@x_train_b.T@y_train
pred_ne = x_test_b@theta

print("Normal Equation")
print("(X^T * X)-1 * X^T * Y")
print("Intercept",theta[0])
print("Slope:",theta[1])
print("MSE:",mean_squared_error(y_test,pred_ne))
print("R2:",r2_score(y_test,pred_ne))

plt.scatter(x_test,y_test,label="Actual Data")
plt.plot(x_test,y_pred_gd,label="Gradi")
plt.plot(x_test,pred_ne,'--',label="Normal Equation")
plt.xlabel("Total Rooms")
plt.ylabel("Median House Value")
plt.legend()
plt.show()



