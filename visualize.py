# dataset_visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Display plots inline if using Jupyter
# %matplotlib inline

# Load the dataset
df = pd.read_csv("Dataset.csv")

# 1. Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# 2. Dataset info
print("\nDataset Information:")
print(df.info())

# 3. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Class distribution
print("\nClass Distribution:")
print(df['label'].value_counts())

# Plotting the distribution
sns.countplot(x='label', data=df)
plt.title('Distribution of Phishing (1) and Legitimate (0) URLs')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# 5. Feature Distribution Plots
# Select numeric features (or change based on actual dataset)
numerical_features = ['url_length', 'num_dots', 'num_hyphens', 'num_subdomains']

for feature in numerical_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[feature], bins=50, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
    else:
        print(f"Feature '{feature}' not found in dataset.")

# 6. Correlation Matrix
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)  # Use numeric_only=True to avoid warning
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()
