import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from src.preprocessing import load_data, split_and_preprocess

def train_and_log():
    # 1. Load & preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)

    # 2. Start MLflow run
    mlflow.set_experiment("faircredit_baseline")
    with mlflow.start_run():
        # 3. Log preprocessing pipeline as artifact
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

        # 4. Train model
        params = {"n_estimators": 100, "max_depth": None, "random_state": 42}
        mlflow.log_params(params)
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        # 5. Predict & evaluate
        preds_proba = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        auc = roc_auc_score(y_test, preds_proba)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", acc)

        # 6. Log model
        mlflow.sklearn.log_model(clf, "model")

        print(f"Run done — AUC: {auc:.3f}, Acc: {acc:.3f}")

if __name__ == "__main__":
    train_and_log()
