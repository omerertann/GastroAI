import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    CSVLogger
)



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "classification")
FOLD_PATH = lambda f: os.path.join(DATASET_PATH, f"fold{f}")

MODEL_SAVE_PATH = lambda f: os.path.join(
    BASE_DIR, "models", f"gi_classifier_densenet121_fold{f}_v7.keras"
)

IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 30

CLASS_NAMES = sorted(os.listdir(os.path.join(DATASET_PATH, "original")))
NUM_CLASSES = len([c for c in CLASS_NAMES if os.path.isdir(os.path.join(DATASET_PATH, "original", c))])


train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255.)


def build_model():

    base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base.trainable = False   # Önce freeze → daha stabil

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_fold(fold):

    print(f"\n==============================")
    print(f"🎯 FOLD {fold} EĞİTİMİ BAŞLIYOR")
    print(f"==============================")

    train_dir = os.path.join(FOLD_PATH(fold), "train")
    val_dir = os.path.join(FOLD_PATH(fold), "val")

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH
    )

    model = build_model()


    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH(fold),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    logger = CSVLogger(f"fold{fold}_training_log.csv")

    # ---------------- TRAIN ---------------- #
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, reduce_lr, early_stop, logger],
        verbose=1
    )

    print(f"✅ Fold {fold} eğitimi tamamlandı!")



def main():
    print("\n🔥 K-Fold V7.3 Ultra-Stable Trainer Başlıyor...\n")

    for fold in range(1, 6):
        train_fold(fold)

    print("\n🎉 TÜM FOLD EĞİTİMLERİ TAMAMLANDI!")
    print("📌 En iyi modeller 'models/' klasöründe kayıtlı.")

if __name__ == "__main__":
    main()
