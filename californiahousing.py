# Implement linear regression with one variable on the California Housing dataset to predict
# housing prices based on a single feature (e.g., the average number of rooms per dwelling).
# Tasks:
# ● Load and preprocess the datase.
# ● Implement linear regression using both gradient descent and the normal equation.
# ● Evaluate the model performance using metrics such as Mean Squared Error
# (MSE) and R-squared.
# ● Visualize the fitted line along with the data points

#Linear Regression, Gradient Descent, Normal Equation, MSE, R², Visualization

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

#Loading dataset 

housing =fetch_california_housing(as_frame=True)

X = housing.data[["AveRooms"]].values     #input feature avg num rooms
y= housing.target.values                  #output value


#Splitting the dataset

X_train,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)  #test_size=0.2 means 20% of data is used for testing
                                                                                    #random_state=42 ensures the same split every time

 
#Feature Scaling(making features into same range like between 0 and 1 )

X_test_original = X_test.copy()

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)      #fit_transform() calculates the mean and standard deviation from training data and scales it.
                                     #transform() scales test data using the training data's mean and standard deviation.

#Add the intercept

X_b_train = np.c_[np.ones((len(X_train),1)),X_train]    #[[1  3]]     3 is input/feature
X_b_test = np.c_[np.ones(((len(X_test)),1)),X_test]    #[[1  4]]
                                                        #[[1 ..]]   slope and intercept calc by matrrix multiplication 

 
#Gradient descent 

theta = np.zeros(2)   # initializes slope and intercept to zeroes 
learning_rate = 0.01
iterations=10000
m = len(X_train)

for i in range (iterations):
    predictions =X_b_train @ theta
    gradients = (2/m)* X_b_train.T @(predictions -y_train)
    theta = theta-learning_rate*gradients


#predicting using gra. descent 

y_pred_gd = X_b_test @ theta

print("Gradient descent : \n")
print("MSE :",mean_squared_error(y_test,y_pred_gd))
print("R2 :",r2_score(y_test,y_pred_gd))


#Normal equation 

theta_ne = np.linalg.pinv(X_b_train) @ y_train   #np.linalg.pinv() calculates the pseudo-inverse of the training matrix
y_pred_ne = X_b_test @ theta_ne                   #[theta=(X^TX)^{-1}X^Ty   y-->y_train


print("Normal equation :\n")
print("MSE :",mean_squared_error(y_test,y_pred_ne))
print("R2 :",r2_score(y_test,y_pred_ne))


#Visualization

X_plot = np.linspace(X.min(),X.max(),100).reshape(-1,1)
X_plot_scaled = scaler.transform(X_plot)
X_plot_b = np.c_[np.ones((len(X_plot_scaled),1)),X_plot_scaled]

y_plot_gd = X_plot_b @ theta
y_plot_ne = X_plot_b @ theta_ne

plt.scatter(X_test_original, y_test, color="blue", alpha=0.3, label="Actual values")
plt.plot(X_plot,y_plot_gd,color="red",label="Gradient Descent")
plt.plot(X_plot,y_plot_ne,color ="green",linestyle= "--",label="Normal Equation")

plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price")
plt.title("Linear Regression - California Housing")
plt.legend()  #display graph labels
plt.show()   #shows graph