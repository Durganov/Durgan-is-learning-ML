# Python for Applied Machine Learning Projects: California Housing Price Prediction
# pip install scikit-learn
# Libraries are necessary

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# --- LOADING THE CALIFORNIA HOUSING DATASET ---
print("\n--- LOADING THE CALIFORNIA HOUSING DATASET ---")
# Load the California housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# --- SPLITTING THE DATASET ---
print("\n--- SPLITTING THE DATASET ---")
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --- DATA PREPROCESSING ---
print("\n--- DATA PREPROCESSING ---")
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- CREATING AND TRAINING THE REGRESSION MODEL ---
print("\n--- CREATING AND TRAINING THE REGRESSION MODEL ---")
# Creating a Linear Regression model
lr_model = LinearRegression()
# Training the model
lr_model.fit(X_train, y_train)

# --- MAKING PREDICTIONS AND EVALUATING THE MODEL ---
print("\n--- MAKING PREDICTIONS AND EVALUATING THE MODEL ---")
# Making predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
