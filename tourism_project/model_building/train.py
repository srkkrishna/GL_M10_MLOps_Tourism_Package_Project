import pandas as pd
import os
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder   # Preprocessing tools
from sklearn.compose import make_column_transformer               # Combine preprocessing steps
from sklearn.pipeline import make_pipeline                        # Build ML pipeline
from sklearn.model_selection import GridSearchCV                  # Hyperparameter tuning
from sklearn.metrics import (                                      # Evaluation metrics
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb                                              # Gradient boosting model

# --- Load train/test splits ---
# Read the pre-saved train/test CSVs
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

print("Training set shape:", Xtrain.shape)
print("Test set shape:", Xtest.shape)

# --- Identify feature types ---
# Numeric features: int/float columns
numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
# Categorical features: object/category columns
categorical_features = Xtrain.select_dtypes(include=["object","category"]).columns.tolist()

# --- Build preprocessing pipeline ---
# Scale numeric features and one-hot encode categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# --- Base model ---
# XGBoost classifier with reproducibility and parallelism enabled
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False
)

# --- Hyperparameter grid ---
# Define search space for GridSearchCV
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__scale_pos_weight": [1, 2]  # handle class imbalance
}

# --- Pipeline ---
# Combine preprocessing and model into one pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- MLflow setup ---
# Track experiments locally or on MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

print("\nStarting MLflow run...")
with mlflow.start_run():
    # --- Hyperparameter tuning ---
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=1,
        error_score="raise" 
    )
    grid_search.fit(Xtrain, ytrain)

    # --- Log best parameters ---
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # --- Predictions ---
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    # --- Metrics ---
    metrics = {
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "train_precision": precision_score(ytrain, y_pred_train, zero_division=0),
        "test_precision": precision_score(ytest, y_pred_test, zero_division=0),
        "train_recall": recall_score(ytrain, y_pred_train, zero_division=0),
        "test_recall": recall_score(ytest, y_pred_test, zero_division=0),
        "train_f1": f1_score(ytrain, y_pred_train, zero_division=0),
        "test_f1": f1_score(ytest, y_pred_test, zero_division=0),
        "train_roc_auc": roc_auc_score(ytrain, y_pred_train_proba),
        "test_roc_auc": roc_auc_score(ytest, y_pred_test_proba),
    }
    mlflow.log_metrics(metrics)

    # --- Print results ---
    print("\nMODEL PERFORMANCE")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification Report:\n", classification_report(ytest, y_pred_test))
    print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred_test))

    # --- Save best model ---
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model.joblib"
    #joblib.dump(best_model, model_path)
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n✅ Best model saved at: {model_path}")

    # --- Log artifact to MLflow ---
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model logged to MLflow")

print("\n=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")
