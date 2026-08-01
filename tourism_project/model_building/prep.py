import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from the project data folder
df = pd.read_csv("tourism_project/data/tourism.csv")
print("Columns in dataset:", df.columns.tolist())

# ✅ Remove unnecessary columns (adjust as needed)
# Example: drop an ID column if present
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratified split to preserve class balance
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits locally as CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("✅ Data prepared: train/test splits written.")
print("Target values kept as:", sorted(y.unique()))
