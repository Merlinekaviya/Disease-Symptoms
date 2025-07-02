import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('/content/Diseases_Symptoms.csv')
print(df.head())
# Get column names
print(df.columns)
# Get info about data types and missing values
print(df.info())
# Describe numerical columns (mean, std, etc.)
print(df.describe())

# Check if there are any missing values
print(df.isnull().sum())
if 'age' in df.columns:
    sns.histplot(df['age'], kde=True)
    plt.title('Age Distribution')
    plt.show()

# Example: countplot for disease outcome if it exists
if 'diagnosis' in df.columns:
    sns.countplot(x='diagnosis', data=df)
    plt.title('Diagnosis Count')
    plt.show()
  print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Example visualization
if 'age' in df.columns:
    sns.histplot(df['age'], kde=True)
    plt.title('Age Distribution')
    plt.show()

if 'diagnosis' in df.columns:
    sns.countplot(x='diagnosis', data=df)
    plt.title('Diagnosis Count')
    plt.show()
print(df.head())
print(df.info())

# 🟢 Encode object columns
le = LabelEncoder()

# Encode target
df['Disease_Code'] = le.fit_transform(df['Disease_Code'])

# Encode other object columns
for col in ['Name', 'Symptoms', 'Treatments']:
    df[col] = le.fit_transform(df[col])

df.dtypes
np.random.seed(42)
df['Risk_Score'] = np.random.randint(1, 100, size=df.shape[0])

# 🟢 Encode object columns
le = LabelEncoder()
for col in ['Name', 'Symptoms', 'Treatments', 'Disease_Code']:
    df[col] = le.fit_transform(df[col])

# 🟢 Prepare features and target
X = df.drop('Risk_Score', axis=1)
y = df['Risk_Score']

# 🟢 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟢 Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 🟢 Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 🟢 Example predictions
print("Predicted:", y_pred[:5])
print("Actual:", y_test[:5].values)
le = LabelEncoder()

# Encode target column
df['Disease_Code'] = le.fit_transform(df['Disease_Code'])

# Encode other object columns
for col in ['Name', 'Symptoms', 'Treatments']:
    df[col] = le.fit_transform(df[col])

# ✅ Prepare features and target
X = df.drop('Disease_Code', axis=1)
y = df['Disease_Code']

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train model with fixes
model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Example predictions
print("\nExample Predictions:", y_pred[:5])
print("Actual Values:", y_test[:5].values)
