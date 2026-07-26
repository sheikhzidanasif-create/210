
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv("/content/car_purchase_data.csv")

# ===========================
# Basic Information
# ===========================
print("="*50)
print("First 5 Rows")
print(df.head())

print("="*50)
print("Last 5 Rows")
print(df.tail())

print("="*50)
print("Dataset Shape")
print(df.shape)

print("="*50)
print("Column Names")
print(df.columns.tolist())

print("="*50)
print("Data Types")
print(df.dtypes)

print("="*50)
print("Dataset Information")
df.info()

print("="*50)
print("Statistical Summary")
print(df.describe(include='all'))

# ===========================
# Missing Values
# ===========================
print("="*50)
print("Missing Values")
print(df.isnull().sum())

# ===========================
# Duplicate Values
# ===========================
print("="*50)
print("Duplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# ===========================
# Correlation Matrix
# ===========================
corr = df[numeric_cols].corr()
print("="*50)
print("Correlation Matrix")
print(corr)

# ===========================
# Histograms
# ===========================
df[numeric_cols].hist(figsize=(12,10))
plt.suptitle("Histograms")
plt.savefig("histograms.png")
plt.show()

# ===========================
# Boxplots
# ===========================
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot - {col}")
    plt.savefig(f"boxplot_{col}.png")
    plt.show()

# ===========================
# Bar Charts
# ===========================
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure(figsize=(8,4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(f"bar_{col}.png")
    plt.show()

# ===========================
# Scatter Plot
# ===========================
if len(numeric_cols) >= 2:
    plt.figure(figsize=(6,4))
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title("Scatter Plot")
    plt.savefig("scatter_plot.png")
    plt.show()

# ===========================
# Line Plot
# ===========================
if len(numeric_cols) >= 2:
    plt.figure(figsize=(8,4))
    plt.plot(df[numeric_cols[0]], label=numeric_cols[0])
    plt.plot(df[numeric_cols[1]], label=numeric_cols[1])
    plt.legend()
    plt.title("Line Plot")
    plt.savefig("line_plot.png")
    plt.show()

# ===========================
# Save Cleaned Dataset
# ===========================
df.to_csv("cleaned_dataset.csv", index=False)

print("="*50)
print("EDA Completed Successfully!")
print("Cleaned dataset saved as cleaned_dataset.csv")
