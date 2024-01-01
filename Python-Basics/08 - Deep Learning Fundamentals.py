# Python for Deep Learning Fundamentals: Using TensorFlow and Keras
# pip install tensorflow
# Libraries are necessary

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# --- LOADING AND PREPARING THE MNIST DATASET ---
print("\n--- LOADING AND PREPARING THE MNIST DATASET ---")
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- CREATING THE NEURAL NETWORK MODEL ---
print("\n--- CREATING THE NEURAL NETWORK MODEL ---")
# Creating a Sequential model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAINING THE MODEL ---
print("\n--- TRAINING THE MODEL ---")
# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# --- EVALUATING THE MODEL ---
print("\n--- EVALUATING THE MODEL ---")
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
