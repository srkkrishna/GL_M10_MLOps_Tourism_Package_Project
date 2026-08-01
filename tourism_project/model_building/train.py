import pandas as pd
import os
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb

# --- Load train/test splits ---
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

print("Training set shape:", Xtrain.shape)
print("Test set shape:", Xtest.shape)

# --- Preprocessor ---
numeric_features = Xtrain.columns.tolist()
preprocessor = make_column_transformer((StandardScaler(), numeric_features))

# --- Base model ---
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False
)

# --- Hyperparameter grid ---
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__scale_pos_weight": [1, 2]  # handle imbalance
}

# --- Pipeline ---
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- MLflow setup ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

print("\nStarting MLflow run...")
with mlflow.start_run():
    # Grid search
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log best params
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
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

    # Print results
    print("\nMODEL PERFORMANCE")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification Report:\n", classification_report(ytest, y_pred_test))
    print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred_test))

    # --- Save best model into deployment folder ---
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n✅ Best model saved at: {model_path}")

    # Log artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model logged to MLflow")

print("\n=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")
