import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
df = pd.read_csv("customer_data.csv")

# Quick peek
print(df.head())          
print("\n--- Info ---")
print(df.info())         
print("\n--- Describe ---")
print(df.describe)   

print ("---- null values ----")
print(df.isnull().sum())


#handling outliers


numeric_features = ['Age', 'Income', 'Tenure', 'SupportCalls']

for col in numeric_features:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Optionally remove outliers using IQR method
for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]   #remove outliers

print("\n Data after removing outliers:")
print(df.describe())



