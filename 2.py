import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression


df = pd.read_csv("auto-mpg.csv")
x = df[['displacement']].values
y = df['mpg']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
plt.scatter(x,y,color="blue",label="Actual Data")

for d in [1,2,3]:
    poly = PolynomialFeatures(degree=d)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_poly,y_train)
    pred = model.predict(x_test_poly)

    print("\nDegree:",d)
    print("MSE:",mean_squared_error(y_test,pred))
    print("R2:",r2_score(y_test,pred))
    idx = np.argsort(x_test.flatten())

    plt.plot(x_test[idx],pred[idx],label = "Degree"+str(d))
    
plt.xlabel("Displacment")
plt.ylabel("MPG")
plt.title("Displacmeen v mpg")
plt.legend()
plt.show()




    
