# Python for Data Visualization: Using Pandas, Matplotlib, and Seaborn
# pip install pandas matplotlib seaborn
# Libraries are necessary

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CREATING A SAMPLE DATA SET ---
print("\n--- CREATING A SAMPLE DATA SET ---")
# Sample data for visualization
np.random.seed(0)
data = {
    'Product': np.random.choice(['A', 'B', 'C'], 100),
    'Sales': np.random.randint(1, 100, 100),
    'Market Share': np.random.rand(100),
    'Satisfaction': np.random.rand(100)
}
df = pd.DataFrame(data)

# --- BASIC PLOT WITH MATPLOTLIB ---
print("\n--- BASIC PLOT WITH MATPLOTLIB ---")
# Line plot
plt.figure(figsize=(10, 4))
plt.plot(df['Sales'])
plt.title('Sales Trend')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.show()

# --- HISTOGRAM ---
print("\n--- HISTOGRAM ---")
# Histogram of Sales
plt.figure(figsize=(10, 4))
plt.hist(df['Sales'], bins=10, color='skyblue', edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# --- SCATTER PLOT ---
print("\n--- SCATTER PLOT ---")
# Scatter plot of Sales vs. Satisfaction
plt.figure(figsize=(10, 4))
plt.scatter(df['Sales'], df['Satisfaction'])
plt.title('Sales vs. Satisfaction')
plt.xlabel('Sales')
plt.ylabel('Satisfaction')
plt.show()

# --- BAR PLOT WITH SEABORN ---
print("\n--- BAR PLOT WITH SEABORN ---")
# Bar plot of average sales by product
plt.figure(figsize=(10, 4))
sns.barplot(x='Product', y='Sales', data=df)
plt.title('Average Sales by Product')
plt.xlabel('Product')
plt.ylabel('Average Sales')
plt.show()

# --- BOX PLOT WITH SEABORN ---
print("\n--- BOX PLOT WITH SEABORN ---")
# Box plot of sales by product
plt.figure(figsize=(10, 4))
sns.boxplot(x='Product', y='Sales', data=df)
plt.title('Sales by Product')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.show()

# --- HEATMAP WITH SEABORN ---
print("\n--- HEATMAP WITH SEABORN ---")
# Selecting only numerical columns for correlation matrix
numerical_df = df.select_dtypes(include=[np.number])
# Heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
