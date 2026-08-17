import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def prepare_and_train(df, target_treatment, dataset_name):
    drop_cols = ['appointment_code', 'patient_name']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if 'gender' in df_clean.columns:
        df_clean['gender'] = df_clean['gender'].fillna('Unknown')
    if 'country' in df_clean.columns:
        df_clean['country'] = df_clean['country'].fillna('Unknown')
    if 'age' in df_clean.columns:
        df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

    if target_treatment not in df_clean.columns:
        return {"Dataset": dataset_name, "Error": f"Hedef kolon '{target_treatment}' bulunamadı."}

    X = df_clean.drop(columns=[target_treatment])
    y = df_clean[target_treatment]

    cat_features = [col for col in ['gender', 'country'] if col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        auto_class_weights='Balanced',
        verbose=0,
        random_seed=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        'Veri Seti': dataset_name,
        'Accuracy': round(accuracy_score(y_test, preds), 4),
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'Recall': round(recall_score(y_test, preds, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, preds, zero_division=0), 4)
    }



df_filtered = pd.read_csv("filtrelenmiş veri seti.csv")
df_unfiltered = pd.read_csv("filtrelenmemiş veri seti.csv")

hedef_tedavi = 'HARMONYCA'

sonuc_filtrelenmis = prepare_and_train(df_filtered, hedef_tedavi, 'Filtrelenmiş')
sonuc_filtrelenmemis = prepare_and_train(df_unfiltered, hedef_tedavi, 'Filtrelenmemiş')

karsilastirma_tablosu = pd.DataFrame([sonuc_filtrelenmis, sonuc_filtrelenmemis])
print(f"Hedef Tedavi: {hedef_tedavi}\n")
print(karsilastirma_tablosu.to_markdown(index=False))
