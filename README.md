# 🩺 Gastro AI - Gastrointestinal Hastalık Teşhis ve Segmentasyon Sistemi

Bu proje, gastrointestinal (Gİ) endoskopi görüntülerinden derin öğrenme (Deep Learning) yöntemleriyle **8 farklı hastalık/anatomik yapının teşhisini (Classification)** ve **polip segmentasyonunu (Binary Polyp Segmentation)** gerçekleştiren entegre bir masaüstü klinik karar destek sistemidir.

---

## 🚀 Özellikler

- **Çoklu Sınıflandırma (Ensemble DenseNet-121):**
  - 8 sınıf: *Polip, Aktif Ülseratif Kolit, Ezofajit, Boya ile Kaldırılmış Polip (EMR), Polip Rezeksiyon Sınırı, Normal Çekum, Normal Pilor, Normal Z-Çizgisi*.
  - 3-Fold Ensemble ağırlıklı oylama ile yüksek doğruluk oranı.
- **Açıklanabilir Yapay Zeka (Grad-CAM):**
  - Modelin görüntünün neresine odaklandığını gösteren ısı haritası görselleştirmesi.
- **Lezyon Segmentasyonu (Residual U-Net / TFLite):**
  - Şüpheli polip alanlarının sınırlarını piksel düzeyinde hassas maskeleme.
- **Hibrid Analiz & Hata Haritası:**
  - Grad-CAM + U-Net kontur bindirme ve Ground Truth ile IoU/Dice skor kıyaslaması.
- **Hekim Giriş & Kayıt Sistemi:**
  - Yerel SQLite veritabanı ve SHA-256 şifreleme ile hekim oturum yönetimi.
- **Profesyonel PDF Raporlama:**
  - ReportLab ile hasta bilgileri, tahmin olasılık grafiği, orijinal/maske/overlay görselleri ve hekim klinik notlarını içeren resmi rapor çıktısı.

---

## 📁 Proje Dizin Yapısı

```text
├── Assets/                     # Uygulama ikon ve logoları
├── models/                     # Eğitilmiş Keras modelleri (DenseNet121 Folds & U-Net)
├── tflite_models/              # Optimize edilmiş TFLite modelleri
├── main_app_gi.py              # Ana masaüstü arayüzü (CustomTkinter)
├── ensemble_predictor_gi.py    # Sınıflandırıcı ve Grad-CAM motoru
├── U_Net.py                    # Residual U-Net segmentasyon mimarisi
├── report_generator.py         # PDF raporlama motoru (ReportLab)
├── Tablo.py                    # Model doğrulama ve metrik test scripti
├── config.py                   # Global ayarlar ve yol tanımları
├── requirements.txt            # Python bağımlılıkları
└── .gitignore                  # Git hariç tutma kuralları
```

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/GI_Entegre_Proje.git
cd GI_Entegre_Proje
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
```bash
python main_app_gi.py
```

---

## 👨‍💻 Lisans & Geliştirici
- Geliştirici: Ömer Faruk
- Bitirme Projesi / Klinik Karar Destek Sistemi
