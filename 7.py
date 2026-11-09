import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import f1_score,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


train = pd.read_csv("dataset/20news_train.csv")
test = pd.read_csv("dataset/20news_test.csv")

x_train = train['text'].fillna('').astype(str)
y_train = train['label']
x_test = test['text'].fillna('').astype(str)
y_test = test['label']


v = CountVectorizer(stop_words = 'english',max_features=5000)
xv_train = v.fit_transform(x_train)
xv_test = v.transform(x_test)


mnb = MultinomialNB()
mnb.fit(xv_train,y_train)
pred1 = mnb.predict(xv_test)

print("Multinomail")
print("F1 score",f1_score(y_test,pred1,average='macro'))
print("Accuracy",accuracy_score(y_test,pred1))

vb = CountVectorizer(stop_words = 'english',max_features=5000,binary=true)
x_train = vb.fit_transform(x_train)
x_test = vb.transform(x_test)

bnb = BernoulliNB()
bnb.fit(x_train,y_train)
pred2 = bnb.predict(x_test)

print("Bernoulli")
print("F1 score",f1_score(y_test,pred2,average='macro'))
print("Accuracy",accuracy_score(y_test,pred2))

model = ['Multinomial','Bernoulli']
acc = [ accuracy_score(y_test,pred1),accuracy_score(y_test,pred2)]
f1 = [f1_score(y_test,pred1,average="macro"),f1_score(y_test,pred2,average="macro")]

x = np.arange(len(model))
w = 0.35

plt.bar(x-w/2,acc,w,label="Accuracy")
plt.bar(x+w/2,f1,w,label="F1 Score")

plt.xlabel("Naive Bayes Model")
plt.ylabel("Score")
plt.title("Multinomial Vs Bernoulli")
plt.xticks(x,model)
plt.legend()
plt.show()
