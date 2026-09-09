"""
Project Configuration — GI_Entegre_Proje
FINAL VERSION (Binary Polyp Segmentation + 8-Class Classification)
Compatible with EXE, CPU, and your dataset structure
"""

import os
import sys

if not getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()

def get_asset_path(filename):
    """Find asset path across PyInstaller bundled mode and regular Python mode."""
    candidates = []
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            candidates.append(os.path.join(sys._MEIPASS, "Assets", filename))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "_internal", "Assets", filename))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "Assets", filename))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Assets", filename))
    candidates.append(os.path.join(BASE_DIR, "Assets", filename))
    candidates.append(os.path.join(BASE_DIR, "_internal", "Assets", filename))
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

CLASSIFICATION_DATASET_PATH = os.path.join(DATASETS_DIR, "classification")
SEGMENTATION_DATASET_PATH   = os.path.join(DATASETS_DIR, "segmentation")

SEGMENTATION_IMAGES_PATH = os.path.join(SEGMENTATION_DATASET_PATH, "images")

FIXED_MASK_DIR = os.path.join(SEGMENTATION_DATASET_PATH, "masks_fixed")

if os.path.exists(FIXED_MASK_DIR) and len(os.listdir(FIXED_MASK_DIR)) > 0:
    SEGMENTATION_MASKS_PATH = FIXED_MASK_DIR
    print(f"[INFO] Fixed masks aktif -> {SEGMENTATION_MASKS_PATH}")
else:
    SEGMENTATION_MASKS_PATH = os.path.join(SEGMENTATION_DATASET_PATH, "masks")
    print(f"[INFO] masks_fixed yok -> Normal masks kullanilacak: {SEGMENTATION_MASKS_PATH}")

CLASSIFIER_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "gi_classifier_densenet121_fold1_v7.keras"),
    os.path.join(MODELS_DIR, "gi_classifier_densenet121_fold2_v7.keras"),
    os.path.join(MODELS_DIR, "gi_classifier_densenet121_fold3_v7.keras"),
]

CLASSIFICATION_MODEL_PATH = CLASSIFIER_MODEL_PATHS[0]

SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, "gi_segmenter_unet_binary_polyp.keras")
TFLITE_MODELS_DIR = os.path.join(BASE_DIR, "tflite_models")
SEGMENTATION_TFLITE_PATH = os.path.join(TFLITE_MODELS_DIR, "gi_segmenter_unet_binary_polyp.tflite")

CLS_IMAGE_SIZE = (224, 224)
SEG_IMAGE_SIZE = (256, 256)

CLS_BATCH_SIZE = 16
SEG_BATCH_SIZE = 4

CLS_EPOCHS = 25
SEG_EPOCHS = 100

CLS_LEARNING_RATE_START = 1e-4
SEG_LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

CLASS_TRANSLATIONS = {
    "dyed-lifted-polyps": "Boya ile Kaldırılmış Polip (EMR)",
    "dyed-resection-margins": "Polip Rezeksiyon Sınırı (Girişim)",
    "esophagitis": "Ezofajit (Reflü Enflamasyonu)",
    "normal-cecum": "Normal Çekum Anatomisi",
    "normal-pylorus": "Normal Pilor Anatomisi",
    "normal-z-line": "Normal Z-Çizgisi (Mide-Yemek Borusu Bileşkesi)",
    "polyps": "Polip (Adenomatöz Yapı)",
    "ulcerative-colitis": "Aktif Ülseratif Kolit"
}

SEGMENTABLE_CLASSES = ["polyp"]
SEGMENTATION_OUTPUT_CHANNELS = 1

POLYP_COLOR   = (255, 0, 0)
OVERLAY_ALPHA = 0.55
GRADCAM_ALPHA = 0.4

OVERLAY_COLOR = POLYP_COLOR
EPSILON = 1e-6

print("[INFO] config.py loaded (Binary Polyp Segmentation + 8-Class Classification)")
