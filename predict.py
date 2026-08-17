import joblib
import pandas as pd

print("Model yükleniyor...")
models_data = joblib.load('cross_sell_models.joblib')

ornek_hasta = pd.DataFrame([{
    "gender": "female",
    "country": "Türkiye",
    "age" : 40,
    "Botoks Alın": 0,
    "Saç Ekimi": 0,
    "NCTF-YÜZ" : 0
}])

print("\n--- ÇAPRAZ SATIŞ TAHMİNLERİ ---")

for hedef in models_data["targets"]:
    model = models_data["models"][hedef]["model"]
    gerekli_kolonlar = models_data["models"][hedef]["features"]

    model_girdisi = ornek_hasta.reindex(columns=gerekli_kolonlar, fill_value=0)

    ihtimal = model.predict_proba(model_girdisi)[0][1]

    print(f"{hedef} Satın Alma İhtimali: % {ihtimal * 100:.1f}")
    hedef = "HARMONYCA"
    model = models_data["models"][hedef]["model"]
    gerekli_kolonlar = models_data["models"][hedef]["features"]

