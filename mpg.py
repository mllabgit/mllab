# Implement polynomial regression on the Auto MPG dataset to predict miles per gallon
# (MPG) based on engine displacement. Compare polynomial regression results with linear
# regression.
# Tasks:
# ● Load and preprocess the dataset.
# ● Implement polynomial regression of varying degrees.
# ● Compare the polynomial regression models with linear regression using metrics
# such as MSE and R-squared.
# ● Visualize the polynomial fit. 



#Polynomial Regression, Linear Regression, MSE, R², Visualization



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("auto-mpg.csv")

data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data["displacement"] = pd.to_numeric(data["displacement"], errors="coerce")
data["mpg"] = pd.to_numeric(data["mpg"], errors="coerce")

data = data.dropna()

X = data[["displacement"]].values
y = data["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [1, 2, 3, 4]
models = {}

for degree in degrees:

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    models[degree] = model

    y_pred = model.predict(X_test)

    print("Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print()

X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

plt.scatter(X, y, color="blue", label="Actual Data")

for degree, model in models.items():
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f"Degree {degree}")

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear and Polynomial Regression")
plt.legend()
plt.show()



