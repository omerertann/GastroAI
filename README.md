# Gastro AI

Gastrointestinal endoscopy image analysis application. Classifies 8 different GI conditions using ensemble DenseNet-121 and performs polyp segmentation with Residual U-Net. Built as a graduation project with CustomTkinter desktop interface.

**Developer:** Ömer ERTAN

## Features

- 8-class classification (polyp, ulcerative colitis, esophagitis, normal cecum, etc.)
- 3-Fold ensemble weighted voting
- Grad-CAM explainability heatmaps
- Binary polyp segmentation (U-Net / TFLite)
- Hybrid analysis with IoU/Dice scoring
- Doctor login system with SQLite + SHA-256
- PDF report generation with ReportLab

## Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/omerertann/GastroAI.git
cd GastroAI
pip install -r requirements.txt
python main_app_gi.py
```

## Dependencies

- TensorFlow >= 2.10
- CustomTkinter >= 5.2
- OpenCV >= 4.8
- NumPy, Pillow, Matplotlib, scikit-learn, ReportLab

## Project Structure

```
├── main_app_gi.py              # Main desktop app (CustomTkinter GUI)
├── ensemble_predictor_gi.py    # Classifier + Grad-CAM engine
├── U_Net.py                    # Residual U-Net architecture
├── report_generator.py         # PDF report generator
├── config.py                   # Global settings and paths
├── Tablo.py                    # Model validation and metrics
├── models/                     # Trained Keras models (DenseNet121 + U-Net)
├── tflite_models/              # Optimized TFLite models
├── Assets/                     # App icons and logos
└── requirements.txt            # Python dependencies
```

## License

All rights reserved.
