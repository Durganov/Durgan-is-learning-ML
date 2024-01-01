# Python for Machine Learning Fundamentals: Using Scikit-Learn
# pip install scikit-learn
# Libraries are necessary

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- LOADING THE DATASET ---
print("\n--- LOADING THE DATASET ---")
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# --- SPLITTING THE DATASET INTO TRAINING AND TESTING SETS ---
print("\n--- SPLITTING THE DATASET ---")
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --- DATA PREPROCESSING ---
print("\n--- DATA PREPROCESSING ---")
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- CREATING AND TRAINING THE MODEL ---
print("\n--- CREATING AND TRAINING THE MODEL ---")
# Creating a KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5)
# Training the classifier
classifier.fit(X_train, y_train)

# --- MAKING PREDICTIONS AND EVALUATING THE MODEL ---
print("\n--- MAKING PREDICTIONS AND EVALUATING THE MODEL ---")
# Making predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluating the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
