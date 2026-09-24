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
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Model directories — PyInstaller bundles data into _internal/ subdirectory
def _find_dir(name):
    """Find a data directory, checking multiple locations for EXE compatibility."""
    candidates = [
        os.path.join(BASE_DIR, name),
    ]
    if getattr(sys, 'frozen', False):
        candidates.insert(0, os.path.join(BASE_DIR, "_internal", name))
        if hasattr(sys, '_MEIPASS'):
            candidates.insert(0, os.path.join(sys._MEIPASS, name))
    for c in candidates:
        if os.path.exists(c) and os.listdir(c):
            return c
    # Fallback: return first candidate and create it
    os.makedirs(candidates[-1], exist_ok=True)
    return candidates[-1]

def _find_model_file(filename, *search_dirs):
    """Search for a model file across multiple directories."""
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    # Also check _internal root for PyInstaller flat bundling
    if getattr(sys, 'frozen', False):
        p = os.path.join(BASE_DIR, "_internal", filename)
        if os.path.exists(p):
            return p
    return os.path.join(search_dirs[0], filename)

TFLITE_MODELS_DIR = _find_dir("tflite_models")
_models_dir_candidate = _find_dir("models")

# In EXE mode, PyInstaller may bundle all models into tflite_models/
# If models/ is empty or doesn't have .keras files, use tflite_models/ as fallback
_has_keras = any(f.endswith('.keras') for f in os.listdir(_models_dir_candidate)) if os.path.exists(_models_dir_candidate) else False
MODELS_DIR = _models_dir_candidate if _has_keras else TFLITE_MODELS_DIR

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
    _find_model_file("gi_classifier_densenet121_fold1_v7.keras", MODELS_DIR, TFLITE_MODELS_DIR),
    _find_model_file("gi_classifier_densenet121_fold2_v7.keras", MODELS_DIR, TFLITE_MODELS_DIR),
    _find_model_file("gi_classifier_densenet121_fold3_v7.keras", MODELS_DIR, TFLITE_MODELS_DIR),
]

CLASSIFICATION_MODEL_PATH = CLASSIFIER_MODEL_PATHS[0]

SEGMENTATION_MODEL_PATH = _find_model_file("gi_segmenter_unet_binary_polyp.keras", MODELS_DIR, TFLITE_MODELS_DIR)
SEGMENTATION_TFLITE_PATH = _find_model_file("gi_segmenter_unet_binary_polyp.tflite", TFLITE_MODELS_DIR, MODELS_DIR)

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
