import pandas as pd
import numpy as np
import optuna
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


def calculate_precision_k(y_true, y_prob, k=50):
    if len(y_true) < k:
        k = len(y_true)
    sorted_indices = np.argsort(y_prob)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_true = np.array(y_true)[top_k_indices]
    return sum(top_k_true) / k


def trainexport_model(csv_path, target_count=10, k_val=50):
    df = pd.read_csv(csv_path)
    drop_cols = ['appointment_code', 'patient_name']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])
    for col in ['gender', 'country']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unknown")
    if "age" in df_clean.columns:
        df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

    treatment_cols = [col for col in df_clean.columns if col not in ["age", "gender", "country"]]
    top_targets = df_clean[treatment_cols].sum().sort_values(ascending=False).head(target_count).index.tolist()
    exported_models = {"targets": top_targets, "models": {}}

    for target in top_targets:
        X = df_clean.drop(columns=[target])
        y = df_clean[target]
        cat_features = [col for col in ['gender', 'country'] if col in X.columns]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        def objective(trial):
            parameters = {
                "iterations": 100,
                "learning_rate": trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                "depth": trial.suggest_int('depth', 1, 10),
                "loss_function": "Logloss",
                "auto_class_weights": "Balanced",
                "cat_features": cat_features,
                "verbose": 0
            }

            model = CatBoostClassifier(**parameters, random_seed=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            return precision_score(y_test, predictions, zero_division=0)

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=10)

        best_parameters = study.best_params
        best_parameters.update({'iterations': 200, 'loss_function': 'Logloss', 'auto_class_weights': 'Balanced',
                                'cat_features': cat_features, 'verbose': 0})

        son_model = CatBoostClassifier(**best_parameters, random_seed=42)
        son_model.fit(X_train, y_train)

        y_probs = son_model.predict_proba(X_test)[:, 1]
        p_at_k = calculate_precision_k(y_test, y_probs, k= k_val)
        print(f"{target} - Precision@{k_val}: {p_at_k:.4f}")

        exported_models["models"][target] = {
            "model": son_model,
            "features": X_train.columns.tolist()
        }
    joblib.dump(exported_models, 'cross_sell_models.joblib')
    print("\nModeller 'cross_sell_models.joblib' olarak başarıyla kaydedildi.")


if __name__ == "__main__":
    trainexport_model("filtrelenmemiş veri seti.csv", target_count=5, k_val=50)
