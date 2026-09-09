import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

from config import (
    CLASSIFIER_MODEL_PATHS,
    CLS_IMAGE_SIZE,
    CLASS_NAMES,
    CLASSIFICATION_MODEL_PATH,
)

tf.get_logger().setLevel('ERROR')

class GIEnsembleClassifier:
    def __init__(self):
        print("\n[INFO] Ensemble Classifier Initializing...\n")

        self.models = []

        missing = [p for p in CLASSIFIER_MODEL_PATHS if not os.path.exists(p)]
        if missing:
            print("\n[WARNING] Ensemble modeller eksik! Fallback -> Tek model kullanilacak.\n")
            if not os.path.exists(CLASSIFICATION_MODEL_PATH):
                print(f"[ERROR] Hicbir model bulunamadi!\nNB: Tahmin fonksiyonlari calismayacak.")
            else:
                m = load_model(CLASSIFICATION_MODEL_PATH, compile=False)
                self.models = [m]
                print(f"[SUCCESS] Fallback model yuklendi -> {CLASSIFICATION_MODEL_PATH}")
        else:
            print("[INFO] Ensemble modeller yukleniyor...\n")
            for p in CLASSIFIER_MODEL_PATHS:
                try:
                    m = load_model(p, compile=False)
                    self.models.append(m)
                    print(f"   [LOADED]: {os.path.basename(p)}")
                except Exception as e:
                    print(f"   [ERROR] Failed to load {os.path.basename(p)}: {e}")

        self.class_names = CLASS_NAMES
        self.gradcam_model = None
        
        if self.models:
            try:
                base_model = self.models[0]
                self.gradcam_model = base_model
            except:
                pass

        print(f"\n[INFO] Ensemble hazir - {len(self.models)} model aktif.\n")

    def predict(self, image_path):
        image_path = os.path.abspath(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[ERROR] Goruntu bulunamadi: {image_path}")

        img = load_img(image_path, target_size=CLS_IMAGE_SIZE)
        x = img_to_array(img)
        
        return self.predict_single_image_array(x)

    def predict_single_image_array(self, img_array):
        if img_array.shape[:2] != CLS_IMAGE_SIZE:
            img_resized = cv2.resize(img_array, CLS_IMAGE_SIZE)
        else:
            img_resized = img_array

        x = img_resized.astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        if not self.models:
             return np.zeros(len(self.class_names), dtype=float)

        results = []
        for model in self.models:
            p = model.predict(x, verbose=0)[0]
            results.append(p)

        avg_pred = np.mean(results, axis=0)
        return avg_pred

    def get_gradcam(self, img_array, layer_name="conv5_block16_2_conv"):
        if not self.models:
            return None
        
        model = self.models[0]
        
        if img_array.shape[:2] != CLS_IMAGE_SIZE:
            img_resized = cv2.resize(img_array, CLS_IMAGE_SIZE)
        else:
            img_resized = img_array

        x = img_resized.astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        try:
            grad_model = Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer_name).output, model.output]
            )
        except ValueError:
            try:
                target_layer = None
                for layer in reversed(model.layers):
                    if 'conv' in layer.name or 'pool' in layer.name:
                        target_layer = layer
                        break
                
                if target_layer:
                    grad_model = Model(
                        inputs=model.inputs,
                        outputs=[target_layer.output, model.output]
                    )
                else:
                    print("[WARNING] GradCAM icin uygun katman bulunamadi.")
                    return None
            except Exception as e:
                print(f"[WARNING] GradCAM model hatasi: {e}")
                return None

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, np.argmax(predictions[0])]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

        cam = cam.numpy()
        cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)

        return heatmap
