import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

data = pd.read_csv("winequality-red.csv")

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

architectures = [
    (8,),
    (16,),
    (32,),
    (16, 8),
    (32, 16),
    (32, 16, 8)
]

for architecture in architectures:

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=architecture,
            activation="relu",
            max_iter=3000,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Architecture:", architecture)
    print("Accuracy:", accuracy)
    print()