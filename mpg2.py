import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shlex

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

with open("auto-mpg.data", "r") as file:
    rows = [shlex.split(line) for line in file if line.strip()]

data = pd.DataFrame(rows, columns=columns)

data["mpg"] = pd.to_numeric(data["mpg"], errors="coerce")
data["displacement"] = pd.to_numeric(data["displacement"], errors="coerce")

data = data[["mpg", "displacement"]].dropna()

X = data[["displacement"]]
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [1, 2, 3, 4]

plt.scatter(X_test, y_test, color="blue", label="Actual Data")

for degree in degrees:
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print()

    X_plot = np.linspace(
        X.min(), X.max(), 200
    ).reshape(-1, 1)

    y_plot = model.predict(X_plot)

    plt.plot(X_plot, y_plot, label=f"Degree {degree}")

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Polynomial Regression on Auto MPG Dataset")
plt.legend()
plt.show()