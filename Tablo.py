import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2

# =========================
# AYARLAR
# =========================

MODEL_PATH = "models/gi_segmenter_unet_binary_polyp.keras"
IMAGE_DIR = "datasets/segmentation/images"
MASK_DIR  = "datasets/segmentation/masks"

IMG_SIZE = (256, 256)
THRESHOLD = 0.15  # GUI ile uyumlu optimize esik degeri (Polip hassasiyeti için)

# =========================
# CUSTOM METRIC & LOSS
# =========================

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    return bce + dloss

# =========================
# MODEL YÜKLE (KRİTİK)
# =========================

TFLITE_PATH = "tflite_models/gi_segmenter_unet_binary_polyp.tflite"

if os.path.exists(TFLITE_PATH):
    print(f"[INFO] TFLite modeli yukleniyor: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    class TFLiteWrapper:
        def predict(self, x, verbose=0):
            interpreter.set_tensor(inp_idx, x.astype(np.float32))
            interpreter.invoke()
            return interpreter.get_tensor(out_idx)

    model = TFLiteWrapper()
else:
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coef": dice_coef
        }
    )

# =========================
# METRİK FONKSİYONLARI
# =========================

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-7)

def pixel_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# =========================
# VERİYİ OKU & DEĞERLENDİR
# =========================

image_files = sorted(os.listdir(IMAGE_DIR))
mask_files  = sorted(os.listdir(MASK_DIR))

ious, dices, accs = [], [], []

for img_name, mask_name in zip(image_files, mask_files):

    # Görüntü
    img_bgr = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Model RGB ile egitildigi icin donusturuldu
    img = cv2.resize(img_rgb, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Maske
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = (mask > 0).astype(np.uint8)

    # Tahmin
    pred = model.predict(img, verbose=0)[0, :, :, 0]
    pred = (pred > THRESHOLD).astype(np.uint8)

    # Metrikler
    ious.append(iou_score(mask, pred))
    dices.append(dice_coef(mask, pred))
    accs.append(pixel_accuracy(mask, pred))

# =========================
# SONUÇLAR
# =========================

print("\n===== SEGMENTASYON MODELİ PERFORMANSI =====\n")
print(f"IoU (Jaccard)      : {np.mean(ious):.4f}")
print(f"Dice (F1-score)    : {np.mean(dices):.4f}")
print(f"Pixel Accuracy     : {np.mean(accs):.4f}")
