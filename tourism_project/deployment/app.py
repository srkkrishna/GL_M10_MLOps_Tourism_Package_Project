from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import sys

# --- Base directory and dataset/model paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tourism.csv"
MODEL_PATH = BASE_DIR / "deployment" / "best_tourism_model.joblib"
print("DEBUG: BASE_DIR =", BASE_DIR, file=sys.stderr)  ## goes to Streamlit logs

# --- Load dataset once and cache ---
@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# --- Build sidebar form using dataset values ---
def build_input_form(df: pd.DataFrame):
    st.sidebar.header("Customer details")

    numeric_cols = [
        "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
        "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
        "Passport", "PitchSatisfactionScore", "OwnCar",
        "NumberOfChildrenVisiting", "MonthlyIncome",
    ]
    integer_cols = [
        "CityTier", "NumberOfPersonVisiting", "NumberOfFollowups",
        "Passport", "OwnCar", "NumberOfChildrenVisiting",
    ]
    categorical_cols = [
        "TypeofContact", "Occupation", "Gender",
        "MaritalStatus", "Designation", "ProductPitched",
    ]

    values = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if col in integer_cols:
            values[col] = st.sidebar.number_input(
                f"{col}",
                min_value=int(series.min()) if not series.empty else 0,
                max_value=int(series.max()) if not series.empty else 10,
                value=int(series.median()) if not series.empty else 0,
                step=1,
            )
        else:
            values[col] = st.sidebar.number_input(
                f"{col}",
                min_value=float(series.min()) if not series.empty else 0.0,
                max_value=float(series.max()) if not series.empty else 100.0,
                value=float(series.median()) if not series.empty else 0.0,
                step=0.1,
            )

    for col in categorical_cols:
        options = sorted([str(x) for x in df[col].dropna().unique()])
        values[col] = st.sidebar.selectbox(f"{col}", options)

    return values

# --- Convert form values into model-ready DataFrame ---
def prepare_input_frame(values: dict, df: pd.DataFrame) -> pd.DataFrame:
    input_columns = [
        "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
        "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
        "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
        "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
        "Designation", "MonthlyIncome",
    ]
    row = {col: values[col] for col in input_columns}
    input_df = pd.DataFrame([row], columns=input_columns)

    numeric_columns = [
        "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
        "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
        "Passport", "PitchSatisfactionScore", "OwnCar",
        "NumberOfChildrenVisiting", "MonthlyIncome",
    ]
    for col in numeric_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    for col in ["TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation", "ProductPitched"]:
        input_df[col] = input_df[col].astype(str)

    return input_df

# --- Main Streamlit app ---
def main():
    st.set_page_config(page_title="Tourism Purchase Predictor", page_icon="✈️", layout="wide")
    st.title("Tourism Package Prediction App")
    st.write("This app reads the tourism dataset from the Data folder and lets you enter customer details for a purchase prediction.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    df = load_dataset(DATA_PATH)
    st.caption(f"Loaded {len(df)} rows from {DATA_PATH.name}")

    values = build_input_form(df)
    input_df = prepare_input_frame(values, df)

    col1, col2 = st.columns([1.4, 0.8])
    with col1:
        st.subheader("Entered details")
        st.dataframe(input_df, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        # --- Load model committed by pipeline ---
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                prediction = int(model.predict(input_df)[0])
                probability = float(model.predict_proba(input_df)[0][1])
                st.metric("Prediction", "Purchased" if prediction == 1 else "Not Purchased")
                st.metric("Probability of Purchase", f"{probability:.2%}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            baseline_prob = float(df["ProdTaken"].mean())
            fallback_prediction = int(baseline_prob >= 0.5)
            st.info("No trained model file was found, so a simple fallback prediction is being used based on the dataset average.")
            st.metric("Prediction", "Purchased" if fallback_prediction == 1 else "Not Purchased")
            st.metric("Probability of Purchase", f"{baseline_prob:.2%}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
