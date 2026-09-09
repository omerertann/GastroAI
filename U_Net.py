# ===========================================
# resunetpp.py  →  U-Net Binary Polyp Segmentation
# Proje: GI_Entegre_Proje
#
# - U-Net mimarisi (hafif, CPU/EXE uyumlu)
# - Çıkış: 1 kanal, sigmoid (binary polip)
# - Kayıp: BCE + Dice Loss
# - Metrik: Dice Coefficient
#
# config.py:
#   SEG_IMAGE_SIZE = (256, 256)
#   SEGMENTATION_OUTPUT_CHANNELS = 1
#   SEGMENTATION_MODEL_PATH = "models/gi_segmenter_unet_binary_polyp.keras"
# ===========================================

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# config'ten input boyutu ve çıkış kanal sayısını al
try:
    from config import SEG_IMAGE_SIZE, SEGMENTATION_OUTPUT_CHANNELS
    _H, _W = SEG_IMAGE_SIZE
    _C_OUT = SEGMENTATION_OUTPUT_CHANNELS
except Exception:
    # config bulunamazsa defaultlara düş
    _H, _W = 256, 256
    _C_OUT = 1


# ===========================================
#                METRİK / LOSS
# ===========================================

def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """
    Binary Dice Coefficient
    y_true, y_pred: [B, H, W, 1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denom + smooth)


def bce_dice_loss(y_true, y_pred):
    """
    Binary Cross-Entropy + (1 - Dice) birleşik kayıp
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    return bce + (1.0 - dice)


# ===========================================
#                U-NET BLOKLARI
# ===========================================
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    Residual Conv Block (ResBlock)
    x -> [Conv+BN+ReLU] -> [Conv+BN] -> (+) -> ReLU
      |                                  ^
      |__________________________________|
      (Projection if shape mismatch)
    """
    shortcut = x
    
    # 1. Conv Layer
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # 2. Conv Layer
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=1)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut Projection (Identity Mapping)
    # Eğer input kanalı sayısı eşit değilse (örn. 64 -> 128) 1x1 Conv ile eşitle.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding=padding, strides=strides)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Toplama (Add) ve Aktivasyon
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    
    return x


def encoder_block(x, filters):
    """
    Encoder: conv_block + maxpool
    """
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(f)
    return f, p


def decoder_block(x, skip, filters):
    """
    Decoder: upsample + concat + conv_block
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


# ===========================================
#                U-NET MODELİ
# ===========================================

def build_unet_binary(
    input_shape=( _H, _W, 3 ),
    num_classes: int = _C_OUT,
):
    """
    Hafif U-Net Binary Polyp Segmentasyon
    - input_shape: (H, W, 3)  → config.SEG_IMAGE_SIZE ile uyumlu
    - num_classes: 1 (binary)
    """

    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 32)   # 256 → 128
    s2, p2 = encoder_block(p1, 64)       # 128 → 64
    s3, p3 = encoder_block(p2, 128)      # 64  → 32
    s4, p4 = encoder_block(p3, 256)      # 32  → 16

    # Bottleneck
    b = conv_block(p4, 512)
    b = layers.Dropout(0.3)(b)

    # Decoder
    d1 = decoder_block(b, s4, 256)       # 16 → 32
    d2 = decoder_block(d1, s3, 128)      # 32 → 64
    d3 = decoder_block(d2, s2, 64)       # 64 → 128
    d4 = decoder_block(d3, s1, 32)       # 128 → 256

    # Çıkış katmanı
    if num_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    outputs = layers.Conv2D(num_classes, (1, 1), padding="same", activation=activation)(d4)

    model = models.Model(inputs, outputs, name="unet_binary_polyp")

    return model


# ===========================================
#         Geriye Dönük Uyum Aliases
# ===========================================

def build_resunetpp(input_shape=( _H, _W, 3 ), num_classes: int = _C_OUT):
    """
    Eski script'ler için alias:
    build_resunetpp(...) çağrılırsa U-Net modelini döndürür.
    """
    return build_unet_binary(input_shape=input_shape, num_classes=num_classes)


# Modeli hızlı test etmek için:
if __name__ == "__main__":
    m = build_unet_binary()
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )
    m.summary()
