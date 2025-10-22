import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
#step 1
df = pd.read_csv("customer_data.csv")

# Quick peek
print(df.head())          
print("\n--- Info ---")
print(df.info())         
print("\n--- Describe ---")
print(df.describe)   

#step2
# Check missing values again
missing = df.isnull().sum()
print("Missing values before handling:\n", missing)

# Example: Fill numeric columns with mean, categorical with mode
for col in df.columns:
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':   #?we use mean here since it better for numerical value
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])    #?we use mode here since it is not a numerical value


# Verify
print("\nMissing values after handling:\n", df.isnull().sum())



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




scaler = StandardScaler() #z-score
scaled_cols = ['Age', 'Income', 'Tenure', 'SupportCalls']

df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

print("Scaled data preview:")
print(df.head())


for col in numeric_features:
    plt.figure(figsize=(5, 3))
    plt.hist(x=df[col],color='skyblue', edgecolor='black')
    plt.grid(True)
    plt.title(f"histogram {col}")
    plt.ylabel("Frequency")
    plt.show()



categorical_features =  ['Gender' , 'ProductType' , 'ChurnStatus' ] 

for col in categorical_features:
    plt.figure(figsize=(5, 3))
    counts = df[col].value_counts()
    plt.bar(counts.index, counts.values, color='red', edgecolor='black')
    plt.title(f"bar {col}")
    plt.ylabel("Frequency")
    plt.show()


for col in numeric_features :
    plt.scatter(x=df['ChurnStatus'],y=df[col])
    plt.show()


corr_matrix = df[numeric_features].corr()
print(corr_matrix)


plt.scatter(x=df['Age'],y=df['Age'])
plt.show()











