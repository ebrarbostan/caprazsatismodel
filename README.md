# Flextell Çapraz Satış Öneri Modeli

Bu repo, Makine öğrenmesi tabanlı çapraz satış (cross-sell) modelini, anonimleştirilmiş eğitim veri setini ve kullanım/eğitim betiklerini içerir.

## Veri Seti Karşılaştırması (Filtrelenmiş vs. Filtrelenmemiş)
Model eğitimi öncesinde veri setindeki kontrol, hediye vb. randevularının silinip silinmemesi gerektiğine dair performans testi yapılmıştır. Filtrelenmemiş veri seti tüm metriklerde daha yüksek başarı gösterdiği için model eğitiminde bu set kullanılmıştır.

| Veri Seti | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Filtrelenmiş | 0.7068 | 0.2708 | 0.7629 | 0.3997 |
| **Filtrelenmemiş** | **0.8366** | **0.4316** | **0.8697** | **0.5769** |

## Kurulum

1. Repoyu bilgisayarınıza klonlayın.
2. Gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:

```
pip install -r requirements.txt
```

## Modeli Test Etme (Tahmin Yapma)
Modelin nasıl çalıştığını ve yeni bir hasta için nasıl öneri ürettiğini görmek için `predict.py` dosyasını kullanabilirsiniz.

Terminalde şu komutu çalıştırın:

```
python predict.py
```

Bu betik; örnek bir hasta profili oluşturur, `cross_sell_models.joblib` dosyasını okur ve bu hasta için 5 ana tedavinin satın alınma olasılıklarını yüzdelik (%) olarak ekrana yazdırır. Gerçek sisteme entegrasyon yaparken bu dosyadaki mantığı referans alabilirsiniz.

## Modeli Yeniden Eğitme
Anonimleştirilmiş veri seti repoda hazır bulunduğu için modeli istediğiniz zaman baştan eğitebilirsiniz.

Terminalde eğitim betiğini çalıştırın:

```bash
python train.py
```

Kod çalışacak, Optuna ile hiperparametre optimizasyonu yapacak ve yeni `cross_sell_models.joblib` dosyasını oluşturup mevcut dosyanın üzerine yazacaktır.
