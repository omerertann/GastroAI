# Gastro AI



A desktop application that performs disease diagnosis and polyp segmentation from gastrointestinal endoscopy images using deep learning. It classifies 8 different GI conditions with an Ensemble DenseNet-121 and performs polyp segmentation with a Residual U-Net. Developed with a CustomTkinter interface as a graduation capstone project.



**Developer:** Ömer ERTAN



## Features



- 8-class diagnosis (polyp, ulcerative colitis, esophagitis, normal cecum, etc.)
- 3-Fold ensemble weighted voting
- Grad-CAM explainable AI heatmaps
- Binary polyp segmentation (U-Net / TFLite)
- Hybrid analysis with IoU/Dice scoring
- Physician login system with SQLite + SHA-256
- PDF report generation with ReportLab

## Installation



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

├── main_app_gi.py              # Main desktop application (CustomTkinter)
├── ensemble_predictor_gi.py    # Classifier + Grad-CAM engine
├── U_Net.py                    # Residual U-Net architecture
├── report_generator.py         # PDF report generator
├── config.py                   # General settings and path definitions
├── Tablo.py                    # Model validation and metrics
├── models/                     # Trained Keras models (DenseNet121 + U-Net)
├── tflite_models/              # Optimized TFLite models
├── Assets/                     # Application icons and logos
└── requirements.txt            # Python dependencies
```



