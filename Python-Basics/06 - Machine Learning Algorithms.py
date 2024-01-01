# Python for Machine Learning Algorithms: Using Scikit-Learn
# pip install scikit-learn
# Libraries are necessary

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- LOADING THE DATASET ---
print("\n--- LOADING THE DATASET ---")
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

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

# --- CREATING AND TRAINING MODELS ---
# Decision Tree
print("\n--- DECISION TREE ---")
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_classifier.predict(X_test)))

# Random Forest
print("\n--- RANDOM FOREST ---")
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_classifier.predict(X_test)))

# Support Vector Machine
print("\n--- SUPPORT VECTOR MACHINE ---")
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
print("SVM Classification Report:\n", classification_report(y_test, svm_classifier.predict(X_test)))

# Logistic Regression
print("\n--- LOGISTIC REGRESSION ---")
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_classifier.predict(X_test)))
