# Python for Data Analysis: Using Pandas, NumPy, Matplotlib, and Seaborn
# pip install pandas numpy matplotlib seaborn
# libraries are necessary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATA SET LOADING AND BASIC OPERATIONS ---
print("\n--- DATA SET LOADING AND BASIC OPERATIONS ---")
# Reading a CSV File with Pandas
df = pd.read_csv('example_dataset.csv')  # Replace with your CSV file

# Displaying the First 5 Rows of the Data Set
print("First 5 Rows of the Data Set:\n", df.head())

# Basic Statistics of the Data Set
print("\nBasic Statistics of the Data Set:\n", df.describe())

# Selecting and Operating on Columns
selected_column = df['Age']  # Using 'Age' column
print("\nFirst 5 Values of the Selected Column (Age):\n", selected_column.head())

# --- BASIC OPERATIONS WITH NUMPY ---
print("\n--- BASIC OPERATIONS WITH NUMPY ---")
# Creating a NumPy Array
np_array = np.array(df['Age'])  # Using 'Age' column

# Simple Statistics
average = np.mean(np_array)
median = np.median(np_array)
print("Average:", average, "Median:", median)

# --- DATA VISUALIZATION: MATPLOTLIB AND SEABORN ---
print("\n--- DATA VISUALIZATION: MATPLOTLIB AND SEABORN ---")
# Creating a Histogram
plt.hist(df['Age'])  # Using 'Age' column
plt.title('Histogram of Age')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Box Plot with Seaborn
sns.boxplot(x=df['Age'])  # Using 'Age' column
plt.title('Box Plot of Age')
plt.show()

# --- DATA MANIPULATION WITH PANDAS ---
print("\n--- DATA MANIPULATION WITH PANDAS ---")
# Filling Missing Values
df.fillna(0, inplace=True)  # Filling missing values with 0

# Adding a New Column
df['DoubleAge'] = df['Age'] * 2  # Creating a new column 'DoubleAge'

# Filtering Data
filtered_df = df[df['Age'] > 30]  # Filtering rows where 'Age' > 30
print("\nFiltered Data Set:\n", filtered_df)

# --- GROUPING AND AGGREGATION ---
print("\n--- GROUPING AND AGGREGATION ---")
# Grouping Data
grouped_df = df.groupby('Department').mean()  # Grouping by 'Department'
print("\nGrouped Data Set:\n", grouped_df)

# --- RESULTS AND SAVING ---
print("\n--- RESULTS AND SAVING ---")
# Saving the Results to a CSV File
df.to_csv('result.csv', index=False)
print("Results saved to 'result.csv'.")
