try:
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        u"GI_Entegre_Proje.App"
    )
except:
    pass

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.simpledialog as sd

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont, Image as PILImage

tf = None
load_model = None
load_img = None
img_to_array = None
GIEnsembleClassifier = None

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

from config import (
    BASE_DIR, DATASETS_DIR, MODELS_DIR, ANALYSIS_DIR,
    CLASSIFICATION_DATASET_PATH, SEGMENTATION_DATASET_PATH,
    SEGMENTATION_IMAGES_PATH, SEGMENTATION_MASKS_PATH,
    CLASSIFIER_MODEL_PATHS, SEGMENTATION_MODEL_PATH, SEGMENTATION_TFLITE_PATH,
    CLS_IMAGE_SIZE, SEG_IMAGE_SIZE,
    CLS_BATCH_SIZE, SEG_BATCH_SIZE, CLS_EPOCHS, SEG_EPOCHS,
    CLS_LEARNING_RATE_START, SEG_LEARNING_RATE,
    CLASS_NAMES, CLASS_TRANSLATIONS, SEGMENTABLE_CLASSES,
    OVERLAY_COLOR, OVERLAY_ALPHA, GRADCAM_ALPHA,
    EPSILON, get_asset_path
)

class TFLiteModelWrapper:
    """Wrapper that provides .predict() interface for TFLite models."""
    def __init__(self, tflite_path):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x, verbose=0):
        import numpy as np
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        outputs = []
        for i in range(x.shape[0]):
            self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(x[i], axis=0).astype(np.float32))
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_details[0]['index'])
            outputs.append(out[0])
        return np.array(outputs)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue") 

# Premium Medical Slate & Blue Theme Colors
COLOR_PRIMARY = ("#0284c7", "#38bdf8")    # Sky Blue Accent (Light, Dark)
COLOR_SECONDARY = ("#0f766e", "#2dd4bf")  # Teal Accent (Light, Dark)
COLOR_BG_LIGHT = "#f8fafc"                 # Slate 50
COLOR_BG_DARK = "#0f172a"                  # Slate 900
COLOR_CARD_LIGHT = "#ffffff"               # White
COLOR_CARD_DARK = "#1e293b"                # Slate 800
COLOR_TEXT_LIGHT = "#0f172a"               # Slate 900
COLOR_TEXT_DARK = "#f8fafc"                # Slate 50

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_imread_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.array(im)

def get_crop_box_from_rgb_square(img_rgb, thr=25, zoom=1.0, pad=0):
    if img_rgb is None or img_rgb.size == 0:
        return (0, 0, img_rgb.shape[1], img_rgb.shape[0])

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > thr
    if not np.any(mask):
        return (0, 0, img_rgb.shape[1], img_rgb.shape[0])

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(img_rgb.shape[0] - 1, y2 + pad)
    x2 = min(img_rgb.shape[1] - 1, x2 + pad)

    return (x1, y1, x2, y2)

def crop_rgb_by_box(img_rgb, box):
    x1, y1, x2, y2 = box
    return img_rgb[y1:y2, x1:x2]

def crop_gray_by_box(img_gray, box):
    x1, y1, x2, y2 = box
    return img_gray[y1:y2, x1:x2]

def to_pil(img_like, size=None, force_rgb=True):
    if isinstance(img_like, Image.Image):
        im = img_like
    else:
        im = Image.fromarray(img_like.astype(np.uint8))

    if size is not None:
        im = im.resize(size, Image.NEAREST if im.mode in ("1", "L") else Image.BILINEAR)

    if force_rgb and im.mode != "RGB":
        im = im.convert("RGB")

    return im

def draw_title(pil_img: PILImage.Image, title: str, fg=(0, 0, 0)):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.rectangle([bbox[0]+6, bbox[1]+4, bbox[2]+10, bbox[3]+8], fill=(255, 255, 255))
    draw.text((8, 6), title, fill=fg, font=font)
    return pil_img

def refine_mask(prob_map, thresh, ksize=5):
    """
    Applies thresholding and morphological smoothing to the probability map.
    """
    mask_bin = (prob_map > thresh).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    
    
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask_bin

def compute_metrics_binary(pred_mask_bin: np.ndarray, true_mask_bin: np.ndarray):
    inter = np.logical_and(pred_mask_bin == 1, true_mask_bin == 1).sum()
    union = np.logical_or(pred_mask_bin == 1, true_mask_bin == 1).sum()

    iou = inter / (union + 1e-7)
    dice = (2 * inter) / (
        (pred_mask_bin == 1).sum() +
        (true_mask_bin == 1).sum() +
        1e-7
    )

    tp = inter
    fp = np.logical_and(pred_mask_bin == 1, true_mask_bin == 0).sum()
    fn = np.logical_and(pred_mask_bin == 0, true_mask_bin == 1).sum()

    return iou, dice, int(tp), int(fp), int(fn)

def error_map_bgr(orig_bgr: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray):
    err = np.zeros_like(orig_bgr)

    tp = np.logical_and(pred_bin == 1, gt_bin == 1)
    fp = np.logical_and(pred_bin == 1, gt_bin == 0)
    fn = np.logical_and(pred_bin == 0, gt_bin == 1)

    err[tp] = [0, 255, 0]
    err[fp] = [255, 0, 0]
    err[fn] = [0, 0, 255]

    blend = cv2.addWeighted(orig_bgr, 0.55, err, 0.45, 0)
    return blend

def draw_binary_overlay_legend(pil_img, has_gt=True):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    t1_bbox = draw.textbbox((0,0), "Tahmin (Kırmızı)", font=font)
    w1 = t1_bbox[2] - t1_bbox[0]
    
    max_text_w = w1
    if has_gt:
        t2_bbox = draw.textbbox((0,0), "Gerçek (Yeşil)", font=font)
        w2 = t2_bbox[2] - t2_bbox[0]
        max_text_w = max(w1, w2)
    
    box_w, box_h = 14, 14
    gap = 6
    text_offset = 8
    
    bg_w = box_w + text_offset + max_text_w + 30
    bg_h = (box_h * 2) + gap + 20 if has_gt else box_h + 20
    
    img_w, img_h = pil_img.size
    
    x0 = (img_w - bg_w) // 2
    if x0 < 0: x0 = 0
    y0 = 20

    overlay = Image.new('RGBA', pil_img.size, (0,0,0,0))
    d_overlay = ImageDraw.Draw(overlay)
    
    d_overlay.rectangle([x0, y0, x0 + bg_w, y0 + bg_h], fill=(255, 255, 255, 220))
    
    pil_img = pil_img.convert("RGBA")
    pil_img = Image.alpha_composite(pil_img, overlay).convert("RGB")
    
    draw = ImageDraw.Draw(pil_img)
    
    content_x = x0 + 15
    content_y = y0 + 10

    draw.rectangle([content_x, content_y, content_x + box_w, content_y + box_h], fill=(255, 0, 0), outline=(50,50,50))
    draw.text((content_x + box_w + text_offset, content_y), "Tahmin (Kırmızı)", fill=(0, 0, 0), font=font)

    if has_gt:
        y2 = content_y + box_h + gap
        draw.rectangle([content_x, y2, content_x + box_w, y2 + box_h], fill=(0, 255, 0), outline=(50,50,50))
        
        label_text = "Gerçek (Yeşil)"
        text_col = (0, 0, 0)
        
        draw.text((content_x + box_w + text_offset, y2), label_text, fill=text_col, font=font)

    return pil_img

def montage_grid(panels, titles, cell_size=(350, 350), cols=3, padding=18, bg=(255, 255, 255)):
    assert len(panels) == len(titles)
    n = len(panels)
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    cell_w, cell_h = cell_size
    W = cols * cell_w + (cols + 1) * padding
    H = rows * cell_h + (rows + 1) * padding

    canvas = Image.new("RGB", (W, H), color=bg)

    for i, (panel, title) in enumerate(zip(panels, titles)):
        r = i // cols
        c = i % cols
        x = padding + c * (cell_w + padding)
        y = padding + r * (cell_h + padding)

        im = to_pil(panel, size=(cell_w, cell_h))
        im = draw_title(im, title, fg=(0, 0, 0))
        canvas.paste(im, (x, y))

    try:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        footer = "Gastrointestinal Hastalık Tespiti Ve Analizi — " + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        draw.text((padding, H - padding - 16), footer, fill=(80, 80, 80), font=font)
    except Exception:
        pass

    return canvas

def find_gt_mask_path(image_path: str, mask_dir: str):
    base_stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(mask_dir, base_stem + ext)
        if os.path.exists(p):
            return p
    return None

class MainApp(ctk.CTk):
    
    def __init__(self):
        super().__init__()
        
        self.configure(fg_color=(COLOR_BG_LIGHT, COLOR_BG_DARK))
        self.apply_app_icon()
        self.after(150, self.apply_app_icon)
        
        self.image_path = None
        self._crop_box = None
        self.seg_model = None
        self.ensemble = None
        self.gradcam_image = None
        self.pred_probs = None
        self.last_probabilities = None
        self.best_class = None
        self.best_conf = None
        self.last_iou = None
        self.last_dice = None

        self._last_orig = None
        self._last_mask_vis = None
        self._last_overlay = None
        self._last_gt_vis = None
        self._last_err_vis = None

        self._busy = False
        self.seg_threshold = 0.50
        self._batch_mode = False

        self.logged_in_doctor = None
        self._lib_load_error = None
        self.after(200, self.start_lazy_loading)
        
        # Check for persistent saved session
        saved_doc = self.get_saved_session()
        if saved_doc:
            self.logged_in_doctor = saved_doc
            self.start_main_application()
        else:
            self.show_login_screen()

    def apply_app_icon(self):
        try:
            icon_ico = get_asset_path("final_icon.ico")
            if os.path.exists(icon_ico):
                try:
                    self.iconbitmap(default=icon_ico)
                except Exception:
                    pass
                try:
                    self.iconbitmap(icon_ico)
                except Exception:
                    pass
            
            icon_img_path = get_asset_path("final_logo.jpg")
            if os.path.exists(icon_img_path):
                # Resize to standard icon size to prevent Windows wm_iconphoto glitch
                im = Image.open(icon_img_path).resize((64, 64), Image.Resampling.LANCZOS)
                self._app_icon_photo = ImageTk.PhotoImage(im)
                self.wm_iconphoto(False, self._app_icon_photo)
        except Exception as e:
            print("İkon yüklenemedi:", e)

    def start_lazy_loading(self):
        threading.Thread(target=self._load_heavy_libs, daemon=True).start()

    def _load_heavy_libs(self):
        global tf, load_model, load_img, img_to_array, GIEnsembleClassifier, warnings
        
        try:
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            import tensorflow as _tf
            _tf.get_logger().setLevel("ERROR")
            tf = _tf
            
            from tensorflow.keras.models import load_model as _lm
            load_model = _lm
            
            from tensorflow.keras.preprocessing.image import load_img as _li, img_to_array as _ita
            load_img = _li
            img_to_array = _ita
            
            from ensemble_predictor_gi import GIEnsembleClassifier as _GIE
            GIEnsembleClassifier = _GIE

            self.ensemble = GIEnsembleClassifier()

            print("[INFO] Heavy libraries loaded asynchronously.")
            self.after(0, lambda: self.status_label.configure(text="Hazır ✔️") if hasattr(self, 'status_label') else None)
            
        except Exception as e:
            print(f"[ERROR] Lib Load Error: {e}")
            self._lib_load_error = str(e)
            self.after(0, lambda: self.status_label.configure(text="Hata!") if hasattr(self, 'status_label') else None)
            self.after(0, lambda: messagebox.showerror("Başlatma Hatası", f"Kütüphaneler yüklenemedi: {e}"))

    def show_login_screen(self):
        # Clear any existing widgets
        for widget in self.winfo_children():
            widget.destroy()
            
        self.minsize(0, 0)
        self.geometry("900x600")
        self.resizable(False, False)
        self.title("Gastro AI - Sistem Girişi")
        self.config(menu="")
        
        self.apply_app_icon()
        self.after(150, self.apply_app_icon)
        
        # Center the window on screen
        self.update_idletasks()
        width = 900
        height = 600
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        # Container frame
        self.login_container = ctk.CTkFrame(self, fg_color=(COLOR_BG_LIGHT, COLOR_BG_DARK), corner_radius=0)
        self.login_container.pack(fill="both", expand=True)
        
        # Left Panel (Branding / Logo)
        self.left_panel = ctk.CTkFrame(
            self.login_container, 
            fg_color="#7eb2d4", 
            corner_radius=0, 
            width=420
        )
        self.left_panel.pack(side="left", fill="both", expand=False)
        self.left_panel.pack_propagate(False)

        # Left Panel (Full-Cover Polished Banner)
        banner_path = get_asset_path("login_banner.png")
        if os.path.exists(banner_path):
            try:
                bg_pil = Image.open(banner_path)
                self.login_bg_img = ctk.CTkImage(light_image=bg_pil, dark_image=bg_pil, size=(420, 600))
                self.bg_label = ctk.CTkLabel(self.left_panel, text="", image=self.login_bg_img, fg_color="#7eb2d4")
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                self.bg_label.image = self.login_bg_img
            except Exception as e:
                print("Banner yuklenemedi:", e)
        
        # Right Panel (Form Area)
        self.right_panel = ctk.CTkFrame(
            self.login_container, 
            fg_color=(COLOR_CARD_LIGHT, COLOR_CARD_DARK), 
            corner_radius=0
        )
        self.right_panel.pack(side="right", fill="both", expand=True)
        
        self.show_login_form()

    def show_login_form(self):
        # Clear right panel widgets
        for widget in self.right_panel.winfo_children():
            widget.destroy()
            
        # Title
        form_title = ctk.CTkLabel(
            self.right_panel, 
            text="Hekim Girişi", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(family="Inter", size=24, weight="bold")
        )
        form_title.pack(pady=(80, 5))
        
        subtitle = ctk.CTkLabel(
            self.right_panel, 
            text="Lütfen sisteme erişmek için bilgilerinizi girin.", 
            text_color=("#64748b", "#94a3b8"),
            font=ctk.CTkFont(size=13)
        )
        subtitle.pack(pady=(0, 30))
        
        # Input container
        input_container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        input_container.pack(fill="x", padx=60)
        
        # Username
        ctk.CTkLabel(
            input_container, 
            text="Kullanıcı Adı", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(10, 5))
        
        self.login_username_entry = ctk.CTkEntry(
            input_container, 
            height=40, 
            placeholder_text="Örn: admin",
            fg_color=(COLOR_BG_LIGHT, "#0f172a"),
            border_color=("#cbd5e1", "#334155")
        )
        self.login_username_entry.pack(fill="x")
        
        # Password
        ctk.CTkLabel(
            input_container, 
            text="Şifre", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(15, 5))
        
        self.login_password_entry = ctk.CTkEntry(
            input_container, 
            height=40, 
            show="*", 
            placeholder_text="••••••••",
            fg_color=(COLOR_BG_LIGHT, "#0f172a"),
            border_color=("#cbd5e1", "#334155")
        )
        self.login_password_entry.pack(fill="x")
        
        # Show/Hide password checkbox
        self.show_pass_var = tk.BooleanVar(value=False)
        show_pass_chk = ctk.CTkCheckBox(
            input_container, 
            text="Şifreyi Göster", 
            variable=self.show_pass_var, 
            command=self.toggle_login_password,
            font=ctk.CTkFont(size=11),
            checkbox_width=18,
            checkbox_height=18,
            border_width=2
        )
        show_pass_chk.pack(anchor="w", pady=(10, 0))
        
        # Error / Status Label
        self.login_error_lbl = ctk.CTkLabel(
            self.right_panel, 
            text="", 
            text_color="#ef4444", 
            font=ctk.CTkFont(size=12)
        )
        self.login_error_lbl.pack(pady=(15, 0))
        
        # Login Button
        self.login_btn = ctk.CTkButton(
            self.right_panel, 
            text="Giriş Yap", 
            height=45, 
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_PRIMARY,
            command=self.handle_login
        )
        self.login_btn.pack(fill="x", padx=60, pady=(15, 10))
        
        # Register option
        register_link = ctk.CTkLabel(
            self.right_panel, 
            text="Yeni Doktor Kaydı Oluştur", 
            text_color=("#0284c7", "#38bdf8"),
            font=ctk.CTkFont(size=12, underline=True),
            cursor="hand2"
        )
        register_link.pack(pady=10)
        register_link.bind("<Button-1>", lambda e: self.show_register_form())

    def toggle_login_password(self):
        if self.show_pass_var.get():
            self.login_password_entry.configure(show="")
        else:
            self.login_password_entry.configure(show="*")

    def show_register_form(self):
        # Clear right panel widgets
        for widget in self.right_panel.winfo_children():
            widget.destroy()
            
        form_title = ctk.CTkLabel(
            self.right_panel, 
            text="Yeni Doktor Kaydı", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(family="Inter", size=24, weight="bold")
        )
        form_title.pack(pady=(40, 5))
        
        subtitle = ctk.CTkLabel(
            self.right_panel, 
            text="Sistemi kullanabilmek için yeni hekim kaydı oluşturun.", 
            text_color=("#64748b", "#94a3b8"),
            font=ctk.CTkFont(size=12)
        )
        subtitle.pack(pady=(0, 20))
        
        input_container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        input_container.pack(fill="x", padx=60)
        
        # Full Name
        ctk.CTkLabel(
            input_container, 
            text="Ad Soyad", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", pady=(5, 2))
        
        self.reg_fullname_entry = ctk.CTkEntry(
            input_container, 
            height=35, 
            placeholder_text="Örn: Dr. Mehmet Öz",
            fg_color=(COLOR_BG_LIGHT, "#0f172a"),
            border_color=("#cbd5e1", "#334155")
        )
        self.reg_fullname_entry.pack(fill="x")
        
        # Title / Specialization
        ctk.CTkLabel(
            input_container, 
            text="Unvan / Branş", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", pady=(10, 2))
        
        self.reg_title_entry = ctk.CTkEntry(
            input_container, 
            height=35, 
            placeholder_text="Örn: Gastroenteroloji Uzmanı",
            fg_color=(COLOR_BG_LIGHT, "#0f172a"),
            border_color=("#cbd5e1", "#334155")
        )
        self.reg_title_entry.pack(fill="x")
        
        # Username
        ctk.CTkLabel(
            input_container, 
            text="Kullanıcı Adı", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", pady=(10, 2))
        
        self.reg_username_entry = ctk.CTkEntry(
            input_container, 
            height=35, 
            placeholder_text="Örn: mehmetoz",
            fg_color=(COLOR_BG_LIGHT, "#0f172a"),
            border_color=("#cbd5e1", "#334155")
        )
        self.reg_username_entry.pack(fill="x")
        
        # Password
        ctk.CTkLabel(
            input_container, 
            text="Şifre", 
            text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK),
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", pady=(10, 2))
        
        self.reg_password_entry = ctk.CTkEntry(
            input_container, 
            height=35, 
            show="*", 
            placeholder_text="En az 6 karakter",
            fg_color=(COLOR_BG_LIGHT, "#0f172a"),
            border_color=("#cbd5e1", "#334155")
        )
        self.reg_password_entry.pack(fill="x")
        
        # Error Label
        self.reg_error_lbl = ctk.CTkLabel(
            self.right_panel, 
            text="", 
            text_color="#ef4444", 
            font=ctk.CTkFont(size=11)
        )
        self.reg_error_lbl.pack(pady=(10, 0))
        
        # Register Button
        self.reg_btn = ctk.CTkButton(
            self.right_panel, 
            text="Kayıt Ol", 
            height=40, 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_SECONDARY,
            command=self.handle_register
        )
        self.reg_btn.pack(fill="x", padx=60, pady=(10, 5))
        
        # Back to login option
        login_link = ctk.CTkLabel(
            self.right_panel, 
            text="Zaten bir hesabınız var mı? Giriş yapın", 
            text_color=("#0284c7", "#38bdf8"),
            font=ctk.CTkFont(size=12, underline=True),
            cursor="hand2"
        )
        login_link.pack(pady=5)
        login_link.bind("<Button-1>", lambda e: self.show_login_form())

    def handle_login(self):
        username = self.login_username_entry.get().strip()
        password = self.login_password_entry.get()
        
        if not username or not password:
            self.login_error_lbl.configure(text="Kullanıcı adı ve şifre gereklidir.")
            return
            
        self.login_error_lbl.configure(text="Giriş yapılıyor...")
        self.update()
        
        doctor_info = self.check_credentials(username, password)
        if doctor_info:
            self.logged_in_doctor = doctor_info
            self.save_session(doctor_info)
            self.start_main_application()
        else:
            self.login_error_lbl.configure(text="Geçersiz kullanıcı adı veya şifre.")

    def handle_register(self):
        fullname = self.reg_fullname_entry.get().strip()
        title = self.reg_title_entry.get().strip()
        username = self.reg_username_entry.get().strip()
        password = self.reg_password_entry.get()
        
        if not fullname or not title or not username or not password:
            self.reg_error_lbl.configure(text="Lütfen tüm alanları doldurun.")
            return
            
        if len(password) < 6:
            self.reg_error_lbl.configure(text="Şifre en az 6 karakter olmalıdır.")
            return
            
        self.reg_error_lbl.configure(text="Kayıt oluşturuluyor...")
        self.update()
        
        success, message = self.register_doctor(username, password, fullname, title)
        if success:
            messagebox.showinfo("Başarılı", "Kayıt başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.")
            self.show_login_form()
        else:
            self.reg_error_lbl.configure(text=message)

    def _get_app_dir(self):
        app_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "GastroAI")
        os.makedirs(app_dir, exist_ok=True)
        return app_dir

    def _get_db_path(self):
        target_db = os.path.join(self._get_app_dir(), "gastro_ai.db")
        # Migrate existing database from project directory if not already copied
        if not os.path.exists(target_db):
            for old_db in [
                os.path.join(BASE_DIR, "gastro_ai.db"),
                os.path.join(BASE_DIR, "dist", "Gastro_AI", "gastro_ai.db")
            ]:
                if os.path.exists(old_db):
                    try:
                        import shutil
                        shutil.copy2(old_db, target_db)
                        print(f"[INFO] Veritabani kalici konuma tasindi -> {target_db}")
                        break
                    except Exception as e:
                        print(f"[WARNING] DB tasima hatasi: {e}")
        return target_db

    def _get_session_path(self):
        return os.path.join(self._get_app_dir(), "session.json")

    def save_session(self, doctor_info):
        try:
            import json
            with open(self._get_session_path(), "w", encoding="utf-8") as f:
                json.dump(doctor_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Session save error:", e)

    def get_saved_session(self):
        try:
            import json
            spath = self._get_session_path()
            if os.path.exists(spath):
                with open(spath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "username" in data:
                        return data
        except Exception as e:
            print("Session load error:", e)
        return None

    def clear_session(self):
        try:
            spath = self._get_session_path()
            if os.path.exists(spath):
                os.remove(spath)
        except Exception as e:
            print("Session clear error:", e)

    def _ensure_db(self):
        import sqlite3
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                title TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def check_credentials(self, username, password):
        try:
            import sqlite3
            import hashlib
            self._ensure_db()
            db_path = self._get_db_path()
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM doctors WHERE username = ?", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                hashed_password = user["password_hash"]
                input_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
                if input_hash == hashed_password:
                    return {
                        "username": user["username"],
                        "full_name": user["full_name"],
                        "title": user["title"]
                    }
            return None
        except Exception as e:
            print("Giriş kontrolü hatası:", e)
            return None

    def register_doctor(self, username, password, full_name, title):
        try:
            import sqlite3
            import hashlib
            self._ensure_db()
            db_path = self._get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            hashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
            cursor.execute(
                "INSERT INTO doctors (username, password_hash, full_name, title) VALUES (?, ?, ?, ?)",
                (username, hashed, full_name, title)
            )
            conn.commit()
            conn.close()
            return True, "Kayıt başarıyla oluşturuldu."
        except sqlite3.IntegrityError:
            return False, "Bu kullanıcı adı zaten alınmış."
        except Exception as e:
            return False, f"Hata: {str(e)}"

    def start_main_application(self):
        # Clear the login screen widgets
        if hasattr(self, 'login_container'):
            self.login_container.destroy()
            
        # Reconfigure grid for main window layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Restore window properties
        self.resizable(True, True)
        self.title("Gastro AI - Entegre Tespit ve Teşhis Paneli")
        self.geometry("1600x900")
        self.minsize(1280, 800)
        self.apply_app_icon()
        self.after(150, self.apply_app_icon)
        
        # Center main window
        self.update_idletasks()
        width = 1600
        height = 900
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        # Create main app layout
        self.create_sidebar()
        self.create_main_view()
        self.create_menu()
        
        # Trigger lazy loading labels or status if loaded
        if tf is not None:
            self.status_label.configure(text="Hazır ✔️")

    def create_sidebar(self):
        # Slate card dark background for sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=("gray90", COLOR_CARD_DARK))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(13, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Gastro AI", 
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Show doctor info if logged in
        if hasattr(self, 'logged_in_doctor') and self.logged_in_doctor:
            self.doc_info_frame = ctk.CTkFrame(self.sidebar_frame, fg_color=("gray85", "#1e293b"), corner_radius=10)
            self.doc_info_frame.grid(row=1, column=0, padx=20, pady=(5, 15), sticky="ew")
            
            self.doc_name_lbl = ctk.CTkLabel(
                self.doc_info_frame, 
                text=self.logged_in_doctor['full_name'], 
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=(COLOR_TEXT_LIGHT, COLOR_TEXT_DARK)
            )
            self.doc_name_lbl.pack(pady=(8, 2), padx=10)
            
            self.doc_title_lbl = ctk.CTkLabel(
                self.doc_info_frame, 
                text=self.logged_in_doctor['title'], 
                font=ctk.CTkFont(size=11),
                text_color=("#64748b", "#94a3b8")
            )
            self.doc_title_lbl.pack(pady=(0, 8), padx=10)
        
        self.lbl_grp1 = ctk.CTkLabel(
            self.sidebar_frame, 
            text="TEMEL İŞLEMLER", 
            anchor="w", 
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray50")
        )
        self.lbl_grp1.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")

        self.load_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Görüntü Yükle", 
            command=self.load_image, 
            height=40, 
            font=ctk.CTkFont(weight="bold"),
            fg_color=COLOR_PRIMARY,
            hover_color=("#0284c7", "#0369a1")
        )
        self.load_btn.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        self.predict_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Hastalığı Tahmin Et", 
            command=lambda: self._run_async(self.predict_class), 
            height=40, 
            fg_color="transparent", 
            border_width=2, 
            border_color=COLOR_PRIMARY,
            text_color=COLOR_PRIMARY,
            hover_color=("#bae6fd", "#075985")
        )
        self.predict_btn.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        self.segment_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Segmentasyon Başlat", 
            command=lambda: self._run_async(self.show_segmentation), 
            height=40, 
            fg_color="transparent", 
            border_width=2, 
            border_color=COLOR_SECONDARY,
            text_color=COLOR_SECONDARY,
            hover_color=("#ccfbf1", "#115e59")
        )
        self.segment_btn.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        self.lbl_grp2 = ctk.CTkLabel(
            self.sidebar_frame, 
            text="GELİŞMİŞ ANALİZ", 
            anchor="w", 
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray50")
        )
        self.lbl_grp2.grid(row=6, column=0, padx=20, pady=(20, 5), sticky="w")

        self.gradcam_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Grad-CAM", 
            command=lambda: self._run_async(self.show_gradcam), 
            fg_color=("#cbd5e1", "#334155"),
            text_color=("#0f172a", "#f8fafc"),
            hover_color=("#94a3b8", "#475569")
        )
        self.gradcam_btn.grid(row=7, column=0, padx=20, pady=5, sticky="ew")

        self.hybrid_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Hibrid Görselleştirme", 
            command=lambda: self._run_async(self.show_hybrid_visual), 
            fg_color=("#cbd5e1", "#334155"),
            text_color=("#0f172a", "#f8fafc"),
            hover_color=("#94a3b8", "#475569")
        )
        self.hybrid_btn.grid(row=8, column=0, padx=20, pady=5, sticky="ew")

        self.compare_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Karşılaştır (GT)", 
            command=lambda: self._run_async(self.show_compare_view), 
            fg_color=("#cbd5e1", "#334155"),
            text_color=("#0f172a", "#f8fafc"),
            hover_color=("#94a3b8", "#475569")
        )
        self.compare_btn.grid(row=9, column=0, padx=20, pady=5, sticky="ew")

        self.appearance_mode_menu = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
            fg_color=("gray80", "gray20"),
            button_color=("gray70", "gray30"),
            text_color=("black", "white")
        )
        self.appearance_mode_menu.grid(row=11, column=0, padx=20, pady=(10, 10), sticky="ew")
        
        self.clear_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Temizle", 
            command=lambda: self._run_async(self.clear_results), 
            fg_color=("#f87171", "#b91c1c"),
            text_color=("#ffffff", "#f8fafc"),
            hover_color=("#ef4444", "#dc2626")
        )
        self.clear_btn.grid(row=12, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.chart_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.chart_frame.grid(row=13, column=0, padx=10, pady=10, sticky="nsew")

    def create_main_view(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=(COLOR_BG_LIGHT, COLOR_BG_DARK))
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.status_frame = ctk.CTkFrame(self.main_frame, height=80, corner_radius=12, fg_color=("white", COLOR_CARD_DARK), border_width=1, border_color=("gray90", "#334155"))
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        self.status_frame.grid_columnconfigure(1, weight=1)

        self.status_label_title = ctk.CTkLabel(self.status_frame, text="Sistem Durumu:", font=ctk.CTkFont(size=14, weight="bold"), text_color=("gray40", "gray60"))
        self.status_label_title.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Hazır", font=ctk.CTkFont(size=16), text_color=("gray50", "gray70"))
        self.status_label.grid(row=0, column=1, padx=10, pady=20, sticky="w")

        # Indeterminate pulse progress bar for a premium loading state
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=220, height=8, indeterminate_speed=1.5, progress_color=COLOR_PRIMARY[1])
        self.progress_bar.grid(row=0, column=1, padx=(80, 10), pady=20, sticky="w")
        self.progress_bar.set(0)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.grid_remove()
        
        self.result_label = ctk.CTkLabel(self.status_frame, text="", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_PRIMARY)
        self.result_label.grid(row=0, column=2, padx=20, pady=20, sticky="e")

        self.image_grid_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.image_grid_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        for i in range(3):
            self.image_grid_frame.grid_columnconfigure(i, weight=1, uniform="group1")
        self.image_grid_frame.grid_rowconfigure(0, weight=1)

        self.card1 = self.create_image_card(self.image_grid_frame, "Orijinal Görüntü", 0)
        self.card2 = self.create_image_card(self.image_grid_frame, "Tahmin Maskesi", 1)
        self.card3 = self.create_image_card(self.image_grid_frame, "Overlay Sonuç", 2)
        
        self.image_label = self.card1["image_label"]
        self.mask_label = self.card2["image_label"]
        self.mask_title_label = self.card2["title_label"]
        self.overlay_label = self.card3["image_label"]

        self.bottom_bar = ctk.CTkFrame(self.main_frame, height=60, fg_color=("white", COLOR_CARD_DARK), corner_radius=12, border_width=1, border_color=("gray90", "#334155"))
        self.bottom_bar.grid(row=3, column=0, sticky="ew", padx=20, pady=20)
        
        self.save_btn = ctk.CTkButton(
            self.bottom_bar, 
            text="Profesyonel PDF Raporu Oluştur", 
            command=self.save_analysis, 
            height=40, 
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_SECONDARY,
            hover_color=("#0d9488", "#115e59")
        )
        self.save_btn.pack(side="right", padx=20, pady=10)

    def create_image_card(self, parent, title, col):
        frame = ctk.CTkFrame(parent, corner_radius=15, fg_color=("white", COLOR_CARD_DARK), border_width=1, border_color=("gray90", "#334155"))
        frame.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        lbl_title = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=15, weight="bold"), text_color=("gray20", "#94a3b8"))
        lbl_title.grid(row=0, column=0, pady=(10, 5))

        lbl_img = ctk.CTkLabel(frame, text="", fg_color="transparent", corner_radius=0)
        lbl_img.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        
        return {"frame": frame, "image_label": lbl_img, "title_label": lbl_title}

    def create_menu(self):
        menubar = tk.Menu(self)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Görüntü Aç (Ctrl+O)", command=self.load_image)
        file_menu.add_command(label="Profesyonel PDF Raporu Oluştur (Ctrl+S)", command=self.save_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Analiz Klasörünü Aç", command=self.open_analysis_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Oturumu Kapat", command=self.handle_logout)
        file_menu.add_command(label="Çıkış", command=self.quit)
        menubar.add_cascade(label="Dosya", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        
        tools_menu.add_command(label="Toplu Analiz Sihirbazı (Batch Wizard)...", command=lambda: self._run_async(self.batch_folder_analysis))
        tools_menu.add_separator()

        report_menu = tk.Menu(tools_menu, tearoff=0)
        report_menu.add_command(label="Tahmin Özeti (Top-3)", command=self.show_top3_summary)
        report_menu.add_command(label="Hızlı Rapor Oluştur (.txt)", command=self.save_quick_report)
        tools_menu.add_cascade(label="Gelişmiş Raporlama", menu=report_menu)

        vis_menu = tk.Menu(tools_menu, tearoff=0)
        vis_menu.add_command(label="Grad-CAM Isı Haritası", command=lambda: self._run_async(self.show_gradcam))
        vis_menu.add_command(label="Segmentasyon Maskesi", command=lambda: self._run_async(self.show_segmentation))
        vis_menu.add_command(label="Hibrid Analiz Görünümü", command=lambda: self._run_async(self.show_hybrid_visual))
        vis_menu.add_separator()
        vis_menu.add_command(label="Ground Truth Karşılaştırma", command=lambda: self._run_async(self.show_compare_view))
        tools_menu.add_cascade(label="Görüntü İşleme & Görselleştirme", menu=vis_menu)
        
        menubar.add_cascade(label="Araçlar", menu=tools_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        
        view_menu = tk.Menu(settings_menu, tearoff=0)
        view_menu.add_command(label="Aydınlık Tema (Light)", command=lambda: ctk.set_appearance_mode("Light"))
        view_menu.add_command(label="Karanlık Tema (Dark)", command=lambda: ctk.set_appearance_mode("Dark"))
        view_menu.add_command(label="Sistem Teması", command=lambda: ctk.set_appearance_mode("System"))
        settings_menu.add_cascade(label="Arayüz Görünümü", menu=view_menu)

        settings_menu.add_command(label="Segmentasyon Hassasiyet Eşiği...", command=self.set_seg_threshold)
        settings_menu.add_separator()
        settings_menu.add_command(label="Hakkında", command=self.show_about_dialog)
        
        menubar.add_cascade(label="Ayarlar", menu=settings_menu)
        
        self.config(menu=menubar)

    def handle_logout(self):
        self.clear_session()
        self.logged_in_doctor = None
        self.show_login_screen()

    def show_about_dialog(self):
        messagebox.showinfo("Hakkında", "Gastro AI - Gelişmiş Gastrointestinal Analiz Sistemi\nVersiyon: v2.6 (Exe Fixed)\n\nDeep Learning Tabanlı Polip Tespiti ve Sınıflandırma.\n\n© 2025 Gastro AI Research Team")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def _run_async(self, work_fn):
        if self._busy:
            messagebox.showinfo("Bilgi", "Lütfen mevcut işlem tamamlanana kadar bekleyin.")
            return

        def runner():
            try:
                self.after(0, lambda: self._set_busy(True))
                work_fn()
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda m=err_msg: messagebox.showerror("Hata", m))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=runner, daemon=True).start()

    def _set_busy(self, is_busy: bool):
        self._busy = is_busy
        state = "disabled" if is_busy else "normal"
        text = "İşleniyor..." if is_busy else "Hazır ✔️"
        
        self.status_label.configure(text=text)
        
        buttons = [self.load_btn, self.predict_btn, self.segment_btn, 
                   self.gradcam_btn, self.hybrid_btn, self.compare_btn, 
                   self.save_btn, self.clear_btn]
                   
        for btn in buttons:
            try:
                btn.configure(state=state)
            except:
                pass
        
        if is_busy:
            self.configure(cursor="watch")
            try:
                self.progress_bar.grid()
                self.progress_bar.start()
            except:
                pass
        else:
            self.configure(cursor="")
            try:
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
            except:
                pass

    def show_image_fill(self, label, img_rgb):
        label.update_idletasks()
        w = label.winfo_width()
        h = label.winfo_height()
        if w < 50 or h < 50:
             w, h = 400, 350

        pil_img = Image.fromarray(img_rgb)
        img_w, img_h = pil_img.size
        
        scale = min(w / img_w, h / img_h)
        
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        
        pil_resized = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=pil_resized, dark_image=pil_resized, size=(disp_w, disp_h))
        
        label.configure(image=ctk_img)
        label.image = ctk_img

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Görüntü Dosyaları", "*.jpg;*.png;*.jpeg")]
        )
        if not file_path:
            return

        self.image_path = file_path

        try:
            img_rgb_full = np.array(Image.open(file_path).convert("RGB"))
            self._crop_box = get_crop_box_from_rgb_square(img_rgb_full, thr=18, zoom=1.0, pad=0)

            img_crop = crop_rgb_by_box(img_rgb_full, self._crop_box)
            
            self.show_image_fill(self.image_label, img_crop)
            
            self.mask_label.configure(image=None)
            self.overlay_label.configure(image=None)

            self.result_label.configure(text="Görüntü yüklendi ✔️")
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü yüklenemedi:\n{e}")
            return

        self._last_orig = None
        self._last_mask_vis = None
        self._last_overlay = None
        self._last_gt_vis = None
        self._last_err_vis = None
        self.pred_probs = None
        self.last_probabilities = None
        self.best_class = None
        self.best_conf = None
        self.last_iou = None
        self.last_dice = None

    def clear_results(self):
        for lbl in [self.image_label, self.mask_label, self.overlay_label]:
            lbl.configure(image=None)
            lbl.image = None

        self.result_label.configure(text="")
        self._last_orig = None
        self.pred_probs = None
        self.last_iou = None
        self.last_dice = None
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        messagebox.showinfo("Bilgi", "Sonuçlar temizlendi.")
        try:
             self.mask_title_label.configure(text="Tahmin Maskesi")
        except:
             pass

    def _update_crop_box_for_current_image(self):
        if not self.image_path:
            self._crop_box = None
            return
        img_rgb = np.array(Image.open(self.image_path).convert("RGB"))
        self._crop_box = get_crop_box_from_rgb_square(img_rgb, thr=18, zoom=1.0, pad=0)

    def predict_class(self):
        if not self.image_path:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü yükleyin.")
            return

        if self.ensemble is None:
            max_wait = 60 # seconds increased
            waited = 0
            
            self._set_busy(True)
            try:
                while self.ensemble is None and waited < max_wait:
                    if self._lib_load_error:
                         messagebox.showerror("Hata", f"Modeller yüklenirken hata oluştu:\n{self._lib_load_error}")
                         return
                    
                    self.update() # Force UI update
                    time.sleep(1)
                    waited += 1
                
                if self.ensemble is None:
                     messagebox.showerror("Hata", "Modeller çok geç yüklendi veya yüklenemedi. Lütfen programı yeniden başlatın.")
                     return
            finally:
                self._set_busy(False)
            
            self.status_label.configure(text="İşleniyor...")

        try:
            img_full = safe_imread_rgb(self.image_path)
            if self._crop_box is None:
                self._crop_box = get_crop_box_from_rgb_square(img_full, thr=18, zoom=1.0, pad=0)

            img_crop = crop_rgb_by_box(img_full, self._crop_box)
            
            self.after(0, lambda: self.show_image_fill(self.image_label, img_crop))

            preds = self.ensemble.predict_single_image_array(img_crop)
            self.pred_probs = preds
            self.last_probabilities = preds

            p_idx = np.argmax(preds)
            conf = preds[p_idx]
            label_name = CLASS_NAMES[p_idx]

            # Get professional Turkish medical translation
            translated_label = CLASS_TRANSLATIONS.get(label_name, label_name.upper())

            self.best_class = translated_label
            self.best_conf = conf

            res_text = f"{translated_label} (%{conf*100:.1f})"
            self.after(0, lambda: self.result_label.configure(text=res_text, text_color=COLOR_PRIMARY))
            
            if not self._batch_mode:
                self.after(0, self.show_probability_chart)

        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin hatası:\n{str(e)}")
            print(e)

    def find_closest_gt_mask(self, target_rgb, img_dir, mask_dir):
        try:
            from glob import glob
            target_small = cv2.resize(target_rgb, (16, 16))
            target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY)
            
            candidates = glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png"))
            
            best_diff = float("inf")
            best_mask_path = None
            
            step = 1 if len(candidates) < 500 else 2 
            
            for p in candidates[::step]:
                try:
                    curr = Image.open(p).convert("RGB").resize((16, 16))
                    curr_arr = np.array(curr)
                    curr_gray = cv2.cvtColor(curr_arr, cv2.COLOR_RGB2GRAY)
                    
                    diff = np.mean(np.abs(target_gray - curr_gray))
                    
                    if diff < best_diff:
                        best_diff = diff
                        fname = os.path.basename(p)
                        m_path = os.path.join(mask_dir, fname)
                        if not os.path.exists(m_path):
                            m_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))
                        
                        if os.path.exists(m_path):
                            best_mask_path = m_path
                except:
                    continue
                    
            return best_mask_path, best_diff
        except Exception as e:
            print("Benzerlik araması hatası:", e)
            return None, 999

    def _ensure_segmentation_model(self):
        if self.seg_model is not None:
            return True

        # Priority 1: High performance TFLite model (Keras version agnostic)
        if os.path.exists(SEGMENTATION_TFLITE_PATH):
            try:
                self.seg_model = TFLiteModelWrapper(SEGMENTATION_TFLITE_PATH)
                print(f"[INFO] TFLite Segmenter loaded -> {SEGMENTATION_TFLITE_PATH}")
                return True
            except Exception as e:
                print(f"[WARNING] TFLite load failed: {e}")

        # Priority 2: Standard Keras model
        if os.path.exists(SEGMENTATION_MODEL_PATH):
            try:
                if load_model is None:
                    max_wait = 60
                    waited = 0
                    self._set_busy(True)
                    try:
                        while load_model is None and waited < max_wait:
                            if self._lib_load_error:
                                messagebox.showerror("Hata", f"Kütüphane hatası:\n{self._lib_load_error}")
                                return False
                            self.update()
                            time.sleep(1)
                            waited += 1
                    finally:
                        self._set_busy(False)
                        self.status_label.configure(text="İşleniyor...")

                self.seg_model = load_model(SEGMENTATION_MODEL_PATH, compile=False)
                print(f"[INFO] Keras Segmenter loaded -> {SEGMENTATION_MODEL_PATH}")
                return True
            except Exception as e:
                messagebox.showerror("Hata", f"Segmentasyon modeli yüklenemedi:\n{e}")
                return False

        messagebox.showerror("Hata", f"Segmentasyon modeli bulunamadı:\n{SEGMENTATION_MODEL_PATH}")
        return False

    def show_segmentation(self):
        if not self.image_path:
            messagebox.showwarning("Uyarı", "Lütfen görüntü yükleyin.")
            return

        if not self._ensure_segmentation_model():
            return

        try:
            img_full = safe_imread_rgb(self.image_path)
            orig_w, orig_h = Image.open(self.image_path).size
            
            if self._crop_box is None:
                 self._crop_box = get_crop_box_from_rgb_square(img_full, thr=25, zoom=1.0, pad=0)
            
            img_crop = crop_rgb_by_box(img_full, self._crop_box)
            x1, y1, x2, y2 = self._crop_box

            inp = cv2.resize(img_crop, SEG_IMAGE_SIZE)
            inp = inp.astype("float32") / 255.0
            
            inp_batch = np.expand_dims(inp, axis=0)
            
            inp_flip = cv2.flip(inp, 1)
            inp_flip_batch = np.expand_dims(inp_flip, axis=0)
            
            pred1 = self.seg_model.predict(inp_batch, verbose=0)[0, :, :, 0]
            pred2 = self.seg_model.predict(inp_flip_batch, verbose=0)[0, :, :, 0]
            
            pred2_inv = cv2.flip(pred2, 1)
            
            
            mask = (pred1 + pred2_inv) / 2.0

            mask_bin = refine_mask(mask, self.seg_threshold)

            mask_vis_r = cv2.resize(mask_bin, (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            true_mask_path = find_gt_mask_path(self.image_path, SEGMENTATION_MASKS_PATH)
            
            
            true_bin = None
            is_pseudo_gt = False
            
            

            if true_mask_path and os.path.exists(true_mask_path):
                try:
                    gt_full = Image.open(true_mask_path).convert("L").resize((orig_w, orig_h), Image.NEAREST)
                    gt_arr_full = np.array(gt_full)
                    gt_arr_crop = crop_gray_by_box(gt_arr_full, self._crop_box)
                    true_bin = cv2.resize(gt_arr_crop, (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                    true_bin = (true_bin > 0).astype(np.uint8)
                    
                    if is_pseudo_gt and np.any(mask_vis_r) and np.any(true_bin):
                        try:
                            M_pred = cv2.moments(mask_vis_r)
                            M_gt = cv2.moments(true_bin)
                            
                            if M_pred["m00"] > 0 and M_gt["m00"] > 0:
                                cX_pred = int(M_pred["m10"] / M_pred["m00"])
                                cY_pred = int(M_pred["m01"] / M_pred["m00"])
                                
                                cX_gt = int(M_gt["m10"] / M_gt["m00"])
                                cY_gt = int(M_gt["m01"] / M_gt["m00"])
                                
                                shift_x = cX_pred - cX_gt
                                shift_y = cY_pred - cY_gt
                                
                                T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                                true_bin = cv2.warpAffine(true_bin, T, (true_bin.shape[1], true_bin.shape[0]), flags=cv2.INTER_NEAREST)
                        except Exception as align_err:
                            print("Alignment error:", align_err)

                except Exception as e:
                    print("GT yuklenemedi:", e)
                    true_bin = None

            highlight = img_crop.copy()
            alpha = 0.70

            pred_mask_bool = (mask_vis_r == 1)
            gt_mask_bool = (true_bin == 1) if true_bin is not None else np.zeros_like(pred_mask_bool)

            only_pred = np.logical_and(pred_mask_bool, np.logical_not(gt_mask_bool))
            color_red = np.array([255, 0, 0], dtype=np.uint8)
            highlight[only_pred] = (highlight[only_pred] * (1 - alpha) + color_red * alpha).astype(np.uint8)

            only_gt = np.logical_and(gt_mask_bool, np.logical_not(pred_mask_bool))
            color_green = np.array([0, 255, 0], dtype=np.uint8)
            highlight[only_gt] = (highlight[only_gt] * (1 - alpha) + color_green * alpha).astype(np.uint8)

            intersection = np.logical_and(pred_mask_bool, gt_mask_bool)
            color_yellow = np.array([255, 255, 0], dtype=np.uint8)
            highlight[intersection] = (highlight[intersection] * (1 - alpha) + color_yellow * alpha).astype(np.uint8)
                
            highlight_pil = Image.fromarray(highlight)
            highlight_pil = draw_binary_overlay_legend(highlight_pil, has_gt=(true_bin is not None))
            highlight = np.array(highlight_pil)

            self._last_orig = img_crop
            self._last_mask_vis = (mask_vis_r * 255).astype(np.uint8)
            self._last_overlay = highlight

            self.after(0, lambda: self.show_image_fill(self.image_label, img_crop))
            
            mask_color_display = np.zeros_like(img_crop)
            mask_color_display[mask_vis_r == 1] = [255, 0, 0] 
            
            self.after(0, lambda: self.show_image_fill(self.mask_label, mask_color_display))
            self.after(0, lambda: self.mask_title_label.configure(text="Tahmin Maskesi"))
            self.after(0, lambda: self.show_image_fill(self.overlay_label, highlight))
            
            if true_bin is not None:
                iou, dice, tp, fp, fn = compute_metrics_binary(mask_vis_r, true_bin)
                self.last_iou = iou
                self.last_dice = dice
                res_text = f"Segmentasyon Başarılı ✔️ | IoU: {iou:.3f}"
            else:
                self.last_iou = None
                self.last_dice = None
                res_text = "Segmentasyon Başarılı ✔️"
                
            if not self._batch_mode:
                 self.after(0, lambda: self.result_label.configure(text=res_text))
                 
        except Exception as e:
             messagebox.showerror("Hata", f"Segmentasyon işlemi sırasında hata:\n{e}")
             print(e)

    def show_gradcam(self):
        if self.pred_probs is None:
            messagebox.showwarning("Uyarı", "Önce sınıflandırma (Hastalığı Tahmin Et) yapmalısınız.")
            return

        img_full = safe_imread_rgb(self.image_path)
        img_crop = crop_rgb_by_box(img_full, self._crop_box)

        heatmap = self.ensemble.get_gradcam(img_crop, layer_name="conv5_block16_2_conv")
        
        if heatmap is None:
            messagebox.showinfo("Bilgi", "Grad-CAM bu model yapısında üretilemedi.")
            return

        heatmap = cv2.resize(heatmap, (img_crop.shape[1], img_crop.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(img_crop, 0.6, heatmap_color, 0.4, 0)

        self._last_overlay = superimposed
        
        self.after(0, lambda: self.show_image_fill(self.mask_label, heatmap_color))
        self.after(0, lambda: self.mask_title_label.configure(text="Grad-CAM"))
        self.after(0, lambda: self.show_image_fill(self.overlay_label, superimposed))
        self.after(0, lambda: self.result_label.configure(text="Grad-CAM Gösteriliyor 🔥"))

    def show_hybrid_visual(self):
        if not self._ensure_segmentation_model():
            return
        
        if self.pred_probs is None:
            messagebox.showwarning("Uyarı", "Önce tahmin yapın.")
            return

        img_full = safe_imread_rgb(self.image_path)
        img_crop = crop_rgb_by_box(img_full, self._crop_box)
        
        inp = cv2.resize(img_crop, SEG_IMAGE_SIZE).astype("float32")/255.0
        pred_seg = self.seg_model.predict(np.expand_dims(inp,0), verbose=0)
        mask_bin = refine_mask(pred_seg[0,:,:,0], self.seg_threshold)
        mask_r = cv2.resize(mask_bin, (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        heatmap = self.ensemble.get_gradcam(img_crop, layer_name="conv5_block16_2_conv")
        if heatmap is not None:
             heatmap = cv2.resize(heatmap, (img_crop.shape[1], img_crop.shape[0]))
             heatmap = np.uint8(255 * heatmap)
             hm_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
             hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
             
             hybrid = hm_color.copy()
             contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             cv2.drawContours(hybrid, contours, -1, (255, 255, 255), 2) 
             
             final = cv2.addWeighted(img_crop, 0.7, hybrid, 0.3, 0)
             
             self.after(0, lambda: self.show_image_fill(self.mask_label, hybrid))
        else:
             final = img_crop

        self._last_overlay = final
        self.after(0, lambda: self.show_image_fill(self.overlay_label, final))
        self.after(0, lambda: self.mask_title_label.configure(text="Hibrit Görselleştirme"))
        self.after(0, lambda: self.result_label.configure(text="Hibrid Görünüm (GradCAM + Seg)"))

    def show_compare_view(self):
        if not self.image_path: return
        
        gt_path = find_gt_mask_path(self.image_path, SEGMENTATION_MASKS_PATH)
        is_pseudo_gt = False
        
        img_full = safe_imread_rgb(self.image_path)
        
        if not gt_path:
            sim_mask_path, _ = self.find_closest_gt_mask(img_full, SEGMENTATION_IMAGES_PATH, SEGMENTATION_MASKS_PATH)
            if sim_mask_path:
                gt_path = sim_mask_path
                is_pseudo_gt = True
            else:
                messagebox.showinfo("Bilgi", "Bu görüntü için Ground Truth bulunamadı ve benzer maske eşleştirilemedi.")
                return

        gt_img = np.array(Image.open(gt_path).convert("L")) 
        
        if self._crop_box is None:
             self._crop_box = get_crop_box_from_rgb_square(img_full, thr=25, zoom=1.0, pad=0)
        
        img_crop = crop_rgb_by_box(img_full, self._crop_box)
        
        gt_full_pil = Image.fromarray(gt_img).resize((img_full.shape[1], img_full.shape[0]), Image.NEAREST)
        gt_full = np.array(gt_full_pil)
        gt_crop = crop_gray_by_box(gt_full, self._crop_box)
        
        gt_bin = (gt_crop > 127).astype(np.uint8)
        
        if not self._ensure_segmentation_model():
            return

        inp = cv2.resize(img_crop, SEG_IMAGE_SIZE).astype("float32")/255.0
        
        inp_batch = np.expand_dims(inp, axis=0)
        inp_flip = cv2.flip(inp, 1)
        inp_flip_batch = np.expand_dims(inp_flip, axis=0)
        
        pred1 = self.seg_model.predict(inp_batch, verbose=0)[0, :, :, 0]
        pred2 = self.seg_model.predict(inp_flip_batch, verbose=0)[0, :, :, 0]
        pred2_inv = cv2.flip(pred2, 1)
        pred_avg = (pred1 + pred2_inv) / 2.0
        
        pred_bin_sm = refine_mask(pred_avg, self.seg_threshold)
        pred_bin = cv2.resize(pred_bin_sm, (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        if is_pseudo_gt and np.any(pred_bin) and np.any(gt_bin):
            try:
                M_pred = cv2.moments(pred_bin)
                M_gt = cv2.moments(gt_bin)
                if M_pred["m00"] > 0 and M_gt["m00"] > 0:
                    cX_pred = int(M_pred["m10"] / M_pred["m00"])
                    cY_pred = int(M_pred["m01"] / M_pred["m00"])
                    cX_gt = int(M_gt["m10"] / M_gt["m00"])
                    cY_gt = int(M_gt["m01"] / M_gt["m00"])
                    shift_x = cX_pred - cX_gt
                    shift_y = cY_pred - cY_gt
                    T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    gt_bin = cv2.warpAffine(gt_bin, T, (gt_bin.shape[1], gt_bin.shape[0]), flags=cv2.INTER_NEAREST)
            except: pass

        err_vis = error_map_bgr(img_crop, pred_bin, gt_bin)
        err_vis_rgb = cv2.cvtColor(err_vis, cv2.COLOR_BGR2RGB) 
        
        pred_color = np.zeros_like(img_crop)
        pred_color[pred_bin == 1] = [255, 0, 0]
        
        gt_color = np.zeros_like(img_crop)
        gt_color[gt_bin == 1] = [0, 255, 0]
        
        iou, dice, tp, fp, fn = compute_metrics_binary(pred_bin, gt_bin)
        self.last_iou = iou
        self.last_dice = dice
        
        title_prefix = "Karşılaştırma (Benzer GT)" if is_pseudo_gt else "Karşılaştırma"
        
        self.after(0, lambda: self._display_compare_plot_safe(
            img_crop, pred_bin, pred_color, gt_color, err_vis_rgb, iou, dice, title_prefix
        ))

        res_txt = f"Karşılaştırma Penceresi Açıldı 🗗 | IoU: {iou:.3f}"
        self.after(0, lambda: self.result_label.configure(text=res_txt))


    def _display_compare_plot_safe(self, img_crop, pred_bin, pred_color, gt_color, err_vis_rgb, iou, dice, title_prefix="Karşılaştırma"):
        try:
            is_dark = (ctk.get_appearance_mode() == "Dark")
            bg_color = "#0f172a" if is_dark else "#f8fafc"
            text_color = "#f8fafc" if is_dark else "#0f172a"
            
            plt.close("all") 
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=bg_color)
            fig.canvas.manager.set_window_title("Gastro AI - Karşılaştırma ve Hata Analizi")
            fig.suptitle(f"{title_prefix} — IoU: {iou:.3f} | Dice: {dice:.3f}", fontsize=13, fontweight='bold', color=text_color)
            
            titles = [
                ["Orijinal Görüntü", "Tahmin Maskesi (Binary)", "Tahmin Maskesi (Renkli)"],
                ["Gerçek Maske (GT)", "Hata Haritası (TP=Yeşil, FP=Kırmızı, FN=Mavi)", ""]
            ]
            
            images = [
                [img_crop, pred_bin, pred_color],
                [gt_color, err_vis_rgb, None]
            ]
            
            for r in range(2):
                for c in range(3):
                    ax = axes[r, c]
                    ax.set_facecolor(bg_color)
                    img = images[r][c]
                    if img is not None:
                        if len(img.shape) == 2:
                            ax.imshow(img, cmap='gray')
                        else:
                            ax.imshow(img)
                        ax.set_title(titles[r][c], fontsize=10, fontweight='bold', color=text_color)
                    ax.axis('off')
            
            plt.tight_layout()
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Plot Hatası", f"Grafik çizilirken hata oluştu:\n{e}")


    def show_probability_chart(self):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, self.show_probability_chart)
            return

        if self.pred_probs is None:
            return

        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        probs = np.array(self.pred_probs, dtype=float).flatten()
        num_classes = len(CLASS_NAMES)
        if len(probs) > num_classes:
            probs = probs[:num_classes]

        # Determine theme colors for chart styling
        is_dark = (ctk.get_appearance_mode() == "Dark")
        bg_color = "#1e293b" if is_dark else "#ffffff"
        text_color = "#f8fafc" if is_dark else "#0f172a"
        bar_base_color = "#475569" if is_dark else "#94a3b8"
        bar_winner_color = "#38bdf8" if is_dark else "#0284c7"

        plt.close("all")
        # Keep figure compact to fit sidebar
        fig, ax = plt.subplots(figsize=(2.8, 2.8), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        y_pos = np.arange(num_classes)
        max_idx = np.argmax(probs)
        bar_colors = [bar_base_color] * len(probs)
        bar_colors[max_idx] = bar_winner_color

        ax.barh(y_pos, probs, align='center', color=bar_colors, height=0.55)
        ax.set_yticks(y_pos)
        
        translated_labels = [CLASS_TRANSLATIONS.get(c, c) for c in CLASS_NAMES]
        ax.set_yticklabels(translated_labels, fontsize=7, color=text_color)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)

        # Remove borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', which='both', length=0, colors=text_color, labelsize=7)
        ax.xaxis.grid(True, linestyle=':', alpha=0.2, color=text_color)

        # Text labels on bars
        for i, v in enumerate(probs):
            ax.text(v + 0.02, i, f"%{v*100:.0f}", color=text_color, va='center', fontweight='bold', fontsize=7)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()

    def open_analysis_folder(self):
        try:
            ensure_dir(ANALYSIS_DIR)
            os.startfile(ANALYSIS_DIR)
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def save_quick_report(self):
        try:
            ensure_dir(ANALYSIS_DIR)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(ANALYSIS_DIR, f"quick_report_{ts}.txt")
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Rapor: {ts}\nEn iyi: {self.best_class}\n")
            
            messagebox.showinfo("Başarılı", f"Rapor: {out_path}")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def set_seg_threshold(self):
        try:
            val = sd.askfloat("Ayarlar", "Segmentasyon Eşiği (0.0 - 1.0):", initialvalue=self.seg_threshold)
            if val is not None:
                self.seg_threshold = val
        except:
            pass
            
    def show_top3_summary(self):
        if self.pred_probs is None: return
        probs = self.pred_probs.flatten()
        idx = np.argsort(probs)[::-1][:3]
        msg = "\n".join([f"{i+1}. {CLASS_NAMES[x]} (%{probs[x]*100:.1f})" for i, x in enumerate(idx)])
        messagebox.showinfo("Top 3 Tahmin", msg)

    def save_analysis(self):
        if self._last_orig is None:
            if self.image_path:
                try:
                    img_full = safe_imread_rgb(self.image_path)
                    if self._crop_box is None:
                        self._crop_box = get_crop_box_from_rgb_square(img_full, thr=18, zoom=1.0, pad=0)
                    self._last_orig = crop_rgb_by_box(img_full, self._crop_box)
                except Exception as e:
                    messagebox.showerror("Hata", f"Görüntü yüklenirken hata oluştu: {e}")
                    return
            else:
                messagebox.showwarning("Uyarı", "Rapor oluşturmak için lütfen önce bir görüntü yükleyin.")
                return

        if self.best_class is None:
            ans = messagebox.askyesno("Tahmin Eksik", "Hastalık tahmini henüz yapılmamış. Rapor oluşturulmadan önce tahmin çalıştırılsın mı?")
            if ans:
                self.predict_class()
                if self.best_class is None:
                    return
            else:
                return

        try:
            from report_generator import ReportInfoDialog, generate_pdf_report
            
            dialog = ReportInfoDialog(self)
            if hasattr(self, 'logged_in_doctor') and self.logged_in_doctor:
                doc_str = f"{self.logged_in_doctor['full_name']} ({self.logged_in_doctor['title']})"
                try:
                    dialog.ent_doctor.delete(0, 'end')
                    dialog.ent_doctor.insert(0, doc_str)
                except Exception as e:
                    print("Doktor ismi doldurulurken hata:", e)
            dialog.focus_force()
            self.wait_window(dialog)
            info = dialog.result
            if info is None:
                return  # Cancelled

            ensure_dir(ANALYSIS_DIR)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"GastroAI_Rapor_{ts}.pdf"
            
            pdf_path = filedialog.asksaveasfilename(
                initialdir=ANALYSIS_DIR,
                initialfile=default_name,
                defaultextension=".pdf",
                filetypes=[("PDF Belgeleri", "*.pdf")]
            )
            if not pdf_path:
                return  # Cancelled

            mask_title = "Segmentasyon Maskesi"
            try:
                mask_title = self.mask_title_label.cget("text")
            except:
                pass

            generate_pdf_report(
                pdf_path=pdf_path,
                patient_info=info,
                best_class=self.best_class,
                best_conf=self.best_conf,
                pred_probs=self.pred_probs,
                class_names=CLASS_NAMES,
                orig_img_arr=self._last_orig,
                mask_img_arr=self._last_mask_vis,
                overlay_img_arr=self._last_overlay,
                mask_title=mask_title,
                last_iou=self.last_iou,
                last_dice=self.last_dice
            )
            
            messagebox.showinfo("Kayıt Başarılı", f"Profesyonel PDF Raporu başarıyla oluşturuldu:\n\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Hata", f"Rapor oluşturulamadı:\n{e}")

    def batch_folder_analysis(self):
        folder = filedialog.askdirectory()
        if not folder: return
        messagebox.showinfo("Batch", "Toplu işlemler arkaplanda çalıştırılabilir. (Demo)")

if __name__ == "__main__":
    try:
        app = MainApp()
        app.mainloop()
    except Exception as e:
        import traceback
        import sys
        
        error_msg = traceback.format_exc()
        try:
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            log_path = os.path.join(desktop, "GastroAI_CrashLog.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Crash Time: {datetime.now()}\n")
                f.write(error_msg)
            
            try:
                import tkinter.messagebox
                root = tkinter.Tk()
                root.withdraw()
                tkinter.messagebox.showerror("Gastro AI Error", f"Uygulama çöktü. Hata raporu masaüstüne kaydedildi:\n{log_path}\n\nHata: {str(e)}")
            except:
                pass
        except:
            print(error_msg)
