import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Load dataset ---
df = pd.read_csv("tourism_project/data/tourism.csv")
print("✅ Dataset loaded successfully.")
print("Columns in dataset:", df.columns.tolist())
print(f"Dataset shape: {df.shape}")

# --- Drop unnecessary columns ---
if "Unnamed: 0" in df.columns or df.columns[0] == "":
    df = df.iloc[:, 1:]
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# --- Handle missing values ---
print("\nHandling missing values...")
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# --- Fix data quality issues ---
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].str.strip().replace({"Fe Male": "Female", "Fe male": "Female"})

# --- Encode categorical features ---
print("\nEncoding categorical variables...")
label_encoder = LabelEncoder()
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
                        "MaritalStatus", "Designation"]

for col in categorical_features:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# --- Define target ---
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# --- Train-test split ---
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

# --- Save splits ---
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
