# Python for Model Selection and Evaluation: Using Scikit-Learn
# pip install scikit-learn
# Libraries are necessary

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

# --- CREATING AND EVALUATING THE MODEL ---
print("\n--- CREATING AND EVALUATING THE MODEL ---")
# Creating a Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')

# Cross-validation
cv = KFold(n_splits=5, random_state=1, shuffle=True)
scores = cross_val_score(svm_classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Cross-validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

# Training the classifier
svm_classifier.fit(X_train, y_train)

# Making predictions
y_pred = svm_classifier.predict(X_test)

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
