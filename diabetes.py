
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))
print()

alphas = np.logspace(-4, 2, 50)

ridge_pipeline = make_pipeline(
    StandardScaler(),
    Ridge()
)

ridge_grid = GridSearchCV(
    ridge_pipeline,
    {"ridge__alpha": alphas},
    cv=5,
    scoring="neg_mean_squared_error"
)

ridge_grid.fit(X_train, y_train)

y_pred_ridge = ridge_grid.predict(X_test)

print("Ridge Regression")
print("Best Alpha:", ridge_grid.best_params_["ridge__alpha"])
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score:", r2_score(y_test, y_pred_ridge))
print()

lasso_pipeline = make_pipeline(
    StandardScaler(),
    Lasso(max_iter=10000)
)

lasso_grid = GridSearchCV(
    lasso_pipeline,
    {"lasso__alpha": alphas},
    cv=5,
    scoring="neg_mean_squared_error"
)

lasso_grid.fit(X_train, y_train)

y_pred_lasso = lasso_grid.predict(X_test)

print("Lasso Regression")
print("Best Alpha:", lasso_grid.best_params_["lasso__alpha"])
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R2 Score:", r2_score(y_test, y_pred_lasso))
