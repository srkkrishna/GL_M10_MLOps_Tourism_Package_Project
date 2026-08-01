import pandas as pd
import os

def register_dataset():
    """
    Load and validate the tourism dataset.
    Ensures required columns exist and prints a summary.
    """
    data_path = os.path.join("tourism_project", "data", "tourism.csv")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # ✅ Check expected columns
    expected_columns = ["CustomerID", "Age", "Gender", "Income", "PackagePurchased"]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    # ✅ Print summary
    print("Dataset registered successfully!")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("First 5 rows:")
    print(df.head())
    
    return df
