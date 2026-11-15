#python -m pip install --upgrade pip
#python -m pip install numpy pandas scikit-learn liac-arff

import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("Loading Fashion-MNIST dataset...")

data = fetch_openml(
    "Fashion-MNIST",
    version=1,
    as_frame=False,
    parser="liac-arff"
)

X = data.data.astype("float32") / 255.0
y = data.target.astype("int64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=5000,
    test_size=1000,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

k_values = [1, 3, 5, 7]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)

    print("\nK =", k)
    print("Accuracy:", accuracy)
    print("Training Time:", round(train_time, 3), "seconds")
    print("Prediction Time:", round(predict_time, 3), "seconds")