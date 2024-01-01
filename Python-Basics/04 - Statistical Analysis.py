# Python for Statistical Analysis: Using SciPy and Statsmodels
# pip install scipy statsmodels
# Libraries are necessary

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- CREATING A SAMPLE DATA SET ---
print("\n--- CREATING A SAMPLE DATA SET ---")
# Sample data for analysis
np.random.seed(0)
data = {
    'Age': np.random.randint(20, 70, 100),
    'Salary': np.random.randint(30000, 150000, 100),
    'Satisfaction': np.random.rand(100) * 10  # Satisfaction score between 0 to 10
}
df = pd.DataFrame(data)

# --- DESCRIPTIVE STATISTICS ---
print("\n--- DESCRIPTIVE STATISTICS ---")
# Basic descriptive statistics
print("Descriptive Statistics:\n", df.describe())

# --- HYPOTHESIS TESTING: T-TEST ---
print("\n--- HYPOTHESIS TESTING: T-TEST ---")
# Performing a t-test on Satisfaction scores
t_stat, p_value = stats.ttest_1samp(df['Satisfaction'], 5)
print("T-Test Results: t-statistic =", t_stat, ", p-value =", p_value)

# --- LINEAR REGRESSION ---
print("\n--- LINEAR REGRESSION ---")
# Linear regression between Age and Salary
X = sm.add_constant(df['Age'])  # adding a constant
model = sm.OLS(df['Salary'], X).fit()
print(model.summary())

# --- PLOTTING RESULTS ---
print("\n--- PLOTTING RESULTS ---")
# Scatter plot of Age vs Salary with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Salary'], alpha=0.5)
plt.plot(df['Age'], model.predict(X), color='red')  # regression line
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
