from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy",random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)

plt.figure(figsize=(12,8))
plot_tree(model,feature_names=data.feature_names,class_names=data.target_names,filled=True)
plt.show()