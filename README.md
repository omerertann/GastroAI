# Gastro AI



Gastrointestinal endoskopi görüntülerinden derin öğrenme ile hastalık teşhisi ve polip segmentasyonu yapan masaüstü uygulaması. Ensemble DenseNet-121 ile 8 farklı GI durumunu sınıflandırır, Residual U-Net ile polip segmentasyonu gerçekleştirir. Bitirme projesi olarak CustomTkinter arayüzü ile geliştirildi.



**Geliştirici:** Ömer ERTAN



## Özellikler



- 8 sınıflı teşhis (polip, ülseratif kolit, özofajit, normal çekum vb.)

- 3-Fold ensemble ağırlıklı oylama

- Grad-CAM açıklanabilir yapay zeka ısı haritaları

- Binary polip segmentasyonu (U-Net / TFLite)

- IoU/Dice skorlamalı hibrit analiz

- SQLite + SHA-256 ile hekim giriş sistemi

- ReportLab ile PDF rapor oluşturma



## Kurulum



**Gereksinimler:** Python 3.9+



```bash

git clone https://github.com/omerertann/GastroAI.git

cd GastroAI

pip install -r requirements.txt

python main_app_gi.py

```



## Bağımlılıklar



- TensorFlow >= 2.10

- CustomTkinter >= 5.2

- OpenCV >= 4.8

- NumPy, Pillow, Matplotlib, scikit-learn, ReportLab



## Proje Yapısı



```

├── main_app_gi.py              # Ana masaüstü uygulaması (CustomTkinter)

├── ensemble_predictor_gi.py    # Sınıflandırıcı + Grad-CAM motoru

├── U_Net.py                    # Residual U-Net mimarisi

├── report_generator.py         # PDF rapor oluşturucu

├── config.py                   # Genel ayarlar ve yol tanımları

├── Tablo.py                    # Model doğrulama ve metrikler

├── models/                     # Eğitilmiş Keras modelleri (DenseNet121 + U-Net)

├── tflite_models/              # Optimize edilmiş TFLite modelleri

├── Assets/                     # Uygulama ikon ve logoları

└── requirements.txt            # Python bağımlılıkları

```



## Lisans



Tüm hakları saklıdır.
