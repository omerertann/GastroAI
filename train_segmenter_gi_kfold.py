import os
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from config import *

print("✅ config.py başarıyla yüklendi.")

def load_dataset():
    IMG_PATH = os.path.join(DATASETS_DIR, "segmentation", "images")
    MSK_PATH = os.path.join(DATASETS_DIR, "segmentation", "masks")

    X, y = [], []
    for img_name in sorted(os.listdir(IMG_PATH)):
        if not img_name.lower().endswith(".png"):
            continue
        mask_name = img_name
        img = load_img(os.path.join(IMG_PATH, img_name), target_size=SEG_IMAGE_SIZE)
        mask = load_img(os.path.join(MSK_PATH, mask_name), target_size=SEG_IMAGE_SIZE, color_mode="grayscale")
        X.append(img_to_array(img) / 255.0)
        y.append(img_to_array(mask) / 255.0)
    return np.array(X), np.array(y)

def build_unet(input_shape):
    inputs = Input(input_shape)
    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 1024)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 64)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs, outputs)
    return model

X, y = load_dataset()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_idx, val_idx in kf.split(X, y):
    print(f"\n🔁 Fold {fold_no}/5 başlıyor...")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = build_unet(SEG_IMAGE_SIZE + (3,))
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
        ModelCheckpoint(os.path.join(MODELS_DIR, f"gi_segmenter_unet_fold{fold_no}.h5"),
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=25, batch_size=8, callbacks=callbacks, verbose=1)
    print(f"✅ Fold {fold_no} tamamlandı.")
    fold_no += 1

print("\n🎉 Segmentasyon eğitimi tamamlandı.")
