
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

X_train = train_data.data
X_test = test_data.data

y_train = train_data.target
y_test = test_data.target

vectorizer1 = CountVectorizer(stop_words="english", max_features=10000)

X_train_count = vectorizer1.fit_transform(X_train)
X_test_count = vectorizer1.transform(X_test)

model1 = MultinomialNB()
model1.fit(X_train_count, y_train)

y_pred1 = model1.predict(X_test_count)

print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("F1 Score:", f1_score(y_test, y_pred1, average="weighted"))
print()

vectorizer2 = CountVectorizer(
    stop_words="english",
    max_features=10000,
    binary=True
)

X_train_binary = vectorizer2.fit_transform(X_train)
X_test_binary = vectorizer2.transform(X_test)

model2 = BernoulliNB()
model2.fit(X_train_binary, y_train)

y_pred2 = model2.predict(X_test_binary)

print("Bernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("F1 Score:", f1_score(y_test, y_pred2, average="weighted"))


