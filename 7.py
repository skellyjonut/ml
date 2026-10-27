from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

data = fetch_20newsgroups(subset = 'all',remove = ('headers','footers','quotes'))

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
vm = CountVectorizer(stop_words = 'english',max_features=5000)
x_train_vm = vm.fit_transform(x_train)
x_test_vm = vm.transform(x_test)

vb = CountVectorizer(stop_words = 'english',binary=True,max_features=5000)

x_train_vb = vb.fit_transform(x_train)
x_test_vb = vb.transform(x_test)

mnb = MultinomialNB()
mnb.fit(x_train_vm,y_train)
pred_mnb = mnb.predict(x_test_vm)

bnb = BernoulliNB()
bnb.fit(x_train_vb,y_train)
pred_bnb = bnb.predict(x_test_vb)

print("MultinomialNB")
print("Accurcy:",accuracy_score(y_test,pred_mnb))
print("F1 Score:",f1_score(y_test,pred_mnb,average='macro'))
print(classification_report(y_test,pred_mnb,target_names=data.target_names))

print("\nBernoulliNB")
print("Accurcy:",accuracy_score(y_test,pred_bnb))
print("F1 Score:",f1_score(y_test,pred_bnb,average='macro'))
print(classification_report(y_test,pred_bnb,target_names=data.target_names))
