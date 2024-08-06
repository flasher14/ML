from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
y_pred = MultinomialNB().fit(X_train, newsgroups_train.target).predict(X_test)

print("Accuracy:", accuracy_score(newsgroups_test.target, y_pred))
print("Precision:", precision_score(newsgroups_test.target, y_pred, average='weighted'))
print("Recall:", recall_score(newsgroups_test.target, y_pred, average='weighted'))
