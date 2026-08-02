import pandas as pd
import os

def register_dataset():
    """
    Load and validate the tourism dataset.
    Ensures required columns exist and prints a summary.
    """

    # Construct the path to the dataset inside the project folder
    data_path = os.path.join("tourism_project", "data", "tourism.csv")
    
    try:
        # Attempt to read the dataset
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Raise an error if the file is missing
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # ✅ Check that the dataset contains expected columns
    expected_columns = ["CustomerID", "Age", "Gender", "Income", "PackagePurchased"]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        # Raise an error if any required columns are missing
        raise ValueError(f"Missing expected columns: {missing}")
    
    # ✅ Print a quick summary of the dataset
    print("Dataset registered successfully!")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("First 5 rows:")
    print(df.head())
    
    # Return the dataframe for downstream use
    return df
