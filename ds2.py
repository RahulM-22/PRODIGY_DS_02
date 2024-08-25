import pandas as pd

# Load the Titanic dataset (replace with the actual path to your file)
data_path = 'D:/MP 1 FINAL/DS 02/titanic/train.csv'
df = pd.read_csv(data_path)

# Display the first few rows to understand the structure of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())

# Fill missing values for 'Age' with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values for 'Embarked' with the most common value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column since it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Verify that missing values have been handled
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the survival rate
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Count (0 = Not Survived, 1 = Survived)')
plt.show()

# Calculate the percentage of survivors
survival_rate = df['Survived'].mean() * 100
print(f"Survival Rate: {survival_rate:.2f}%")
# Plot survival rate by gender
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set1')
plt.title('Survival Count by Gender')
plt.show()

# Calculate survival rate by gender
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
print(gender_survival)
# Plot survival rate by passenger class
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set3')
plt.title('Survival Count by Passenger Class')
plt.show()

# Calculate survival rate by passenger class
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print(class_survival)
# Plot the distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot survival by age using a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=df, palette='coolwarm')
plt.title('Survival by Age')
plt.show()
# Plot a heatmap to show correlation between features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
