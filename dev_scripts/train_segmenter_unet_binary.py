

import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa   # ✅ HATA ÇÖZÜLDÜ

from config import (
    BASE_DIR, DATASETS_DIR, MODELS_DIR,
    SEG_IMAGE_SIZE, SEGMENTATION_MASKS_PATH,
    SEGMENTATION_OUTPUT_CHANNELS,
    SEG_EPOCHS, SEG_BATCH_SIZE, SEG_LEARNING_RATE
)

from U_Net import build_unet_binary, dice_coef, bce_dice_loss



IMAGE_DIR = os.path.join(DATASETS_DIR, "segmentation", "images")
MASK_DIR  = SEGMENTATION_MASKS_PATH

image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.*")))
mask_paths  = sorted(glob(os.path.join(MASK_DIR, "*.*")))

print("📁 Images:", IMAGE_DIR)
print("📁 Masks :", MASK_DIR)
print(f"🔍 Toplam görüntü   : {len(image_paths)}")
print(f"🔍 Toplam maske      : {len(mask_paths)}")

assert len(image_paths) == len(mask_paths), \
    "❌ image ve mask sayısı birebir eşleşmiyor!"



def read_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(SEG_IMAGE_SIZE)
    img = np.array(img) / 255.0
    return img.astype(np.float32)


def read_mask(path):
    m = Image.open(path).convert("L")
    m = m.resize(SEG_IMAGE_SIZE, Image.NEAREST)
    m = (np.array(m) > 0).astype(np.float32)
    m = np.expand_dims(m, axis=-1)
    return m



def augment(img, mask):
    """En kaliteli augmentation seti."""

    
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

   
    if tf.random.uniform(()) > 0.75:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    
    angle = tf.random.uniform((), -10, 10) * (3.14159265 / 180)
    img = tfa.image.rotate(img, angle, fill_mode="reflect")
    mask = tfa.image.rotate(mask, angle, fill_mode="nearest")

    
    img = tf.image.random_brightness(img, 0.2)

    
    img = tf.image.random_contrast(img, 0.7, 1.3)

    
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform((), 1.0, 1.15)
        new_size = tf.cast(scale * SEG_IMAGE_SIZE[0], tf.int32)
        img = tf.image.resize(img, (new_size, new_size))
        img = tf.image.resize_with_crop_or_pad(img, SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1])
        mask = tf.image.resize(mask, (new_size, new_size), method="nearest")
        mask = tf.image.resize_with_crop_or_pad(mask, SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1])

    
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)

    return img, mask



def tf_parse(img_path, mask_path):
    img = tf.numpy_function(read_image, [img_path], tf.float32)
    mask = tf.numpy_function(read_mask, [mask_path], tf.float32)
    img.set_shape([SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1], 3])
    mask.set_shape([SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1], 1])
    return img, mask


def train_augment(img, mask):
    img, mask = augment(img, mask)
    return img, mask



def get_dataset(img_paths, mask_paths, batch=8, training=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.shuffle(buffer_size=512, reshuffle_each_iteration=True)
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



train_x, val_x, train_y, val_y = train_test_split(
    image_paths, mask_paths,
    test_size=0.2, random_state=42
)

train_ds = get_dataset(train_x, train_y, SEG_BATCH_SIZE, training=True)
val_ds   = get_dataset(val_x, val_y, SEG_BATCH_SIZE, training=False)

print(f"📊 Train: {len(train_x)} görüntü")
print(f"📊 Val  : {len(val_x)} görüntü")



print("\n🧠 U-Net modeli oluşturuluyor...")
model = build_unet_binary(
    input_shape=(SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1], 3),
    num_classes=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(SEG_LEARNING_RATE),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

model.summary()



save_path = os.path.join(MODELS_DIR, "gi_segmenter_unet_binary_polyp.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        save_path, monitor="val_dice_coef", save_best_only=True,
        verbose=1, mode="max"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef", factor=0.5,
        patience=5, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef", patience=15,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(MODELS_DIR, "segmenter_training_log.csv")
    )
]



print("\n🚀 Eğitim başlıyor...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=SEG_EPOCHS,
    callbacks=callbacks
)

print("\n🎉 Eğitim tamamlandı!")
print(f"📦 En iyi model kaydedildi → {save_path}")
