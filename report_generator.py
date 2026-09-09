"""
Report Generator for Gastro AI
Provides custom patient dialog and PDF reporting capabilities with Turkish character support.
"""

import os
import tempfile
from datetime import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for safety in threads
import matplotlib.pyplot as plt

import customtkinter as ctk
from PIL import Image as PILImage

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Determine standard Windows fonts directory for Turkish support
WINDIR = os.environ.get('WINDIR', 'C:\\Windows')
ARIAL_PATH = os.path.join(WINDIR, 'Fonts', 'arial.ttf')
ARIAL_BOLD_PATH = os.path.join(WINDIR, 'Fonts', 'arialbd.ttf')

# Register Fonts
FONT_FAMILY = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

try:
    if os.path.exists(ARIAL_PATH) and os.path.exists(ARIAL_BOLD_PATH):
        pdfmetrics.registerFont(TTFont('Arial', ARIAL_PATH))
        pdfmetrics.registerFont(TTFont('Arial-Bold', ARIAL_BOLD_PATH))
        FONT_FAMILY = "Arial"
        FONT_BOLD = "Arial-Bold"
        print("🟩 Arial & Arial-Bold fonts registered successfully for PDF generation.")
    else:
        print("⚠️ Arial fonts not found in default Windows path. Falling back to Helvetica (Turkish characters may not render correctly).")
except Exception as e:
    print(f"⚠️ Font registration error: {e}. Falling back to Helvetica.")


class ReportInfoDialog(ctk.CTkToplevel):
    """
    Custom CustomTkinter Dialog to retrieve patient information and medical notes.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Gastro AI - Rapor Bilgileri")
        self.geometry("520x450")
        self.resizable(False, False)
        
        # Style/Theme settings
        self.configure(fg_color=("white", "#1e1e1e"))
        
        # Modality
        self.transient(parent)
        self.grab_set()
        self.result = None
        
        # Header
        lbl_header = ctk.CTkLabel(
            self, 
            text="Hekim ve Hasta Bilgileri Girişi", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        lbl_header.pack(pady=(20, 10))
        
        # Form Container
        form_frame = ctk.CTkFrame(self, fg_color="transparent")
        form_frame.pack(fill="x", padx=30, pady=5)
        
        # Row 0: Patient Name
        ctk.CTkLabel(form_frame, text="Hasta Adı Soyadı:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", pady=6)
        self.ent_name = ctk.CTkEntry(form_frame, width=280)
        self.ent_name.grid(row=0, column=1, pady=6, padx=(15, 0), sticky="e")
        
        # Row 1: Protocol No
        ctk.CTkLabel(form_frame, text="Protokol / ID No:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=0, sticky="w", pady=6)
        self.ent_id = ctk.CTkEntry(form_frame, width=280)
        self.ent_id.grid(row=1, column=1, pady=6, padx=(15, 0), sticky="e")
        
        # Row 2: Attending Doctor
        ctk.CTkLabel(form_frame, text="Sorumlu Hekim:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=2, column=0, sticky="w", pady=6)
        self.ent_doctor = ctk.CTkEntry(form_frame, width=280)
        self.ent_doctor.grid(row=2, column=1, pady=6, padx=(15, 0), sticky="e")
        
        # Set default doctor name if applicable
        self.ent_doctor.insert(0, "Dr. Mehmet Öz")
        
        # Notes
        ctk.CTkLabel(self, text="Klinik Notlar / Değerlendirme:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=30, pady=(10, 5))
        self.txt_notes = ctk.CTkTextbox(self, height=100, width=460)
        self.txt_notes.pack(padx=30, pady=5)
        self.txt_notes.insert("1.0", "Hastada yapılan endoskopik muayenede elde edilen görüntüler yapay zeka destekli Gastro AI sistemi ile analiz edilmiştir.")
        
        # Button Container
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(15, 20))
        
        self.btn_ok = ctk.CTkButton(
            btn_frame, 
            text="Raporu Oluştur (PDF)", 
            command=self.on_ok, 
            width=150,
            font=ctk.CTkFont(weight="bold")
        )
        self.btn_ok.pack(side="right", padx=5)
        
        self.btn_cancel = ctk.CTkButton(
            btn_frame, 
            text="İptal", 
            fg_color="transparent", 
            border_width=1, 
            text_color=("gray10", "#DCE4EE"), 
            command=self.on_cancel, 
            width=100
        )
        self.btn_cancel.pack(side="right", padx=5)
        
        # Bind events
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.bind("<Escape>", lambda event: self.on_cancel())
        
        # Focus on name field
        self.ent_name.focus_set()
        
    def on_ok(self):
        self.result = {
            "patient_name": self.ent_name.get().strip() or "Belirtilmedi",
            "patient_id": self.ent_id.get().strip() or "Belirtilmedi",
            "doctor_name": self.ent_doctor.get().strip() or "Belirtilmedi",
            "notes": self.txt_notes.get("1.0", "end-1c").strip() or "Ek klinik not bulunmamaktadır."
        }
        self.destroy()
        
    def on_cancel(self):
        self.result = None
        self.destroy()


def save_probability_chart_image(probs, class_names, output_path):
    """
    Generates and saves a clean, crisp horizontal bar chart of the predictions.
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(4.5, 2.3))
    
    y_pos = np.arange(len(class_names))
    probs = np.array(probs, dtype=float).flatten()
    
    # Trim to size mismatch if any
    if len(probs) > len(class_names):
        probs = probs[:len(class_names)]
        
    max_idx = np.argmax(probs)
    bar_colors = ['#1f538d'] * len(probs)
    bar_colors[max_idx] = '#d32f2f'  # Winning class colored Red
    
    # Get professional Turkish names for the chart labels
    try:
        from config import CLASS_TRANSLATIONS
        display_labels = [CLASS_TRANSLATIONS.get(c, c) for c in class_names]
    except Exception:
        display_labels = class_names
    
    ax.barh(y_pos, probs, align='center', color=bar_colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=7)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel('Olasılık', fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Hide top and right spines for a clean modern flat look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add values text
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"%{v*100:.1f}", color='black', va='center', fontweight='bold', fontsize=7)
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_pdf_report(
    pdf_path,
    patient_info,
    best_class,
    best_conf,
    pred_probs,
    class_names,
    orig_img_arr,
    mask_img_arr,
    overlay_img_arr,
    mask_title="Tahmin Maskesi",
    last_iou=None,
    last_dice=None
):
    """
    Builds a professional single-page PDF report with Patient Details, AI Predictions Chart,
    Visual Analysis Grid, and Doctor's Remarks.
    """
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=letter, 
        rightMargin=36, 
        leftMargin=36, 
        topMargin=36, 
        bottomMargin=36
    )
    
    story = []
    temp_files = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName=FONT_BOLD,
        fontSize=15,
        leading=18,
        textColor=colors.white,
        alignment=1,  # Center
    )
    
    section_title_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName=FONT_BOLD,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1f538d'),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName=FONT_FAMILY,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#333333')
    )
    
    body_bold = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName=FONT_BOLD
    )
    
    # --- HEADER BANNER ---
    header_data = [[Paragraph("<b>GASTRO AI - ENDOSKOPİ ANALİZ VE TEŞHİS RAPORU</b>", title_style)]]
    header_table = Table(header_data, colWidths=[540])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1f538d')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))
    
    # --- INFO AND DIAGNOSTIC DETAILS (LEFT SIDE) ---
    report_date = datetime.now().strftime("%d.%m.%Y %H:%M")
    
    info_data = [
        [Paragraph("<b>Hasta Adı Soyadı:</b>", body_style), Paragraph(patient_info["patient_name"], body_style)],
        [Paragraph("<b>Protokol / ID No:</b>", body_style), Paragraph(patient_info["patient_id"], body_style)],
        [Paragraph("<b>Sorumlu Hekim:</b>", body_style), Paragraph(patient_info["doctor_name"], body_style)],
        [Paragraph("<b>Teşhis Tarihi:</b>", body_style), Paragraph(report_date, body_style)],
        [Paragraph("<b>Yapay Zeka Teşhisi:</b>", body_bold), Paragraph(f"<font color='#d32f2f'><b>{best_class.upper()} (%{best_conf*100:.1f})</b></font>", body_bold)],
    ]
    
    # Append segmentation metrics if present
    if last_iou is not None and last_dice is not None:
        info_data.append([
            Paragraph("<b>Segmentasyon Başarımı (IoU / Dice):</b>", body_style),
            Paragraph(f"IoU: {last_iou:.4f} | Dice: {last_dice:.4f}", body_style)
        ])
        
    info_table = Table(info_data, colWidths=[120, 180])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    # --- CHART IMAGE (RIGHT SIDE) ---
    chart_temp_path = tempfile.mktemp(suffix=".png")
    save_probability_chart_image(pred_probs, class_names, chart_temp_path)
    temp_files.append(chart_temp_path)
    
    chart_flowable = RLImage(chart_temp_path, width=220, height=110)
    
    # Assemble Top Combined Table (Info on Left, Chart on Right)
    top_combined_data = [[info_table, chart_flowable]]
    top_combined_table = Table(top_combined_data, colWidths=[310, 230])
    top_combined_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (1,0), (1,0), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(top_combined_table)
    
    # --- VISUAL ANALYSIS SECTION ---
    story.append(Paragraph("<b>Görsel Bulgular ve Analiz Çıktıları</b>", section_title_style))
    
    # Save image arrays as temporary files
    orig_temp_path = tempfile.mktemp(suffix=".png")
    PILImage.fromarray(orig_img_arr).save(orig_temp_path)
    temp_files.append(orig_temp_path)
    
    img_width, img_height = 170, 160
    
    if mask_img_arr is not None and overlay_img_arr is not None:
        # Full 3-column view: Original | Mask | Overlay
        mask_temp_path = tempfile.mktemp(suffix=".png")
        if len(mask_img_arr.shape) == 2 or mask_img_arr.shape[2] == 1:
            PILImage.fromarray(mask_img_arr).convert("RGB").save(mask_temp_path)
        else:
            PILImage.fromarray(mask_img_arr).save(mask_temp_path)
        temp_files.append(mask_temp_path)
        
        overlay_temp_path = tempfile.mktemp(suffix=".png")
        PILImage.fromarray(overlay_img_arr).save(overlay_temp_path)
        temp_files.append(overlay_temp_path)
        
        img_table_data = [
            [RLImage(orig_temp_path, width=img_width, height=img_height),
             RLImage(mask_temp_path, width=img_width, height=img_height),
             RLImage(overlay_temp_path, width=img_width, height=img_height)],
            [Paragraph("<b>1. Orijinal Görüntü</b>", ParagraphStyle('C1', parent=body_style, alignment=1)),
             Paragraph(f"<b>2. {mask_title}</b>", ParagraphStyle('C2', parent=body_style, alignment=1)),
             Paragraph("<b>3. Bindirilmiş (Overlay) Çıktı</b>", ParagraphStyle('C3', parent=body_style, alignment=1))]
        ]
        img_table = Table(img_table_data, colWidths=[180, 180, 180])
    else:
        # Single image view
        img_table_data = [
            [RLImage(orig_temp_path, width=240, height=220)],
            [Paragraph("<b>1. Orijinal Görüntü (İşlenmemiş)</b>", ParagraphStyle('C1', parent=body_style, alignment=1))]
        ]
        img_table = Table(img_table_data, colWidths=[540])
        
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ('TOPPADDING', (0,1), (-1,1), 2),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 5))
    
    # --- DOCTOR NOTES SECTION ---
    story.append(Paragraph("<b>Hekim Klinik Değerlendirme Notları</b>", section_title_style))
    notes_para = Paragraph(patient_info["notes"].replace("\n", "<br/>"), body_style)
    notes_box_table = Table([[notes_para]], colWidths=[540])
    notes_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f3f5')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#ced4da')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(notes_box_table)
    story.append(Spacer(1, 15))
    
    # --- FOOTER DISCLAIMER ---
    footer_style = ParagraphStyle(
        'DocFooter',
        parent=body_style,
        fontName=FONT_FAMILY,
        fontSize=7,
        leading=10,
        textColor=colors.HexColor('#6c757d'),
        alignment=1
    )
    footer_para = Paragraph(
        "<b>Bilgilendirme Notu:</b> Bu rapor, yapay zeka destekli Gastro AI klinik karar destek sistemi tarafından üretilmiştir. AI sonuçları "
        "kesin tıbbi tanı yerine geçmez ve kararlar klinik bulgular eşliğinde uzman hekim tarafından doğrulanmalıdır.<br/>"
        "© 2025 Gastro AI Research & Development Team. Tüm Hakları Saklıdır.", 
        footer_style
    )
    story.append(footer_para)
    
    # Build Document
    try:
        doc.build(story)
        print(f"🟩 PDF report generated successfully at: {pdf_path}")
    finally:
        # Clean up temporary files
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Failed to remove temp file {f}: {e}")
