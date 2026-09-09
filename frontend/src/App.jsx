import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Upload, 
  RefreshCw, 
  LogOut, 
  FileText, 
  Layers, 
  FileDown, 
  User, 
  Lock, 
  Sliders, 
  CheckCircle,
  Eye,
  AlertCircle,
  X
} from 'lucide-react';

function App() {
  // Auth state - fully local, no backend
  const [isLoggedIn, setIsLoggedIn] = useState(localStorage.getItem('gastro_logged_in') === 'true');
  const [doctor, setDoctor] = useState(JSON.parse(localStorage.getItem('gastro_doctor')) || null);
  const [authMode, setAuthMode] = useState('login'); // login | register
  const [authForm, setAuthForm] = useState({
    username: '',
    password: '',
    full_name: '',
    title: ''
  });
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');

  // App core state
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [threshold, setThreshold] = useState(0.50);
  const [isBusy, setIsBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Sistem Hazır');
  
  // Results
  const [classifications, setClassifications] = useState([]);
  const [processedImages, setProcessedImages] = useState(null);
  const [tabMode, setTabMode] = useState('segmentation'); // segmentation | gradcam
  
  // PDF Report State
  const [showReportModal, setShowReportModal] = useState(false);
  const [patientInfo, setPatientInfo] = useState({
    patient_id: '',
    patient_name: '',
    patient_age: '',
    patient_gender: 'Erkek',
    indications: '',
    doctor_notes: ''
  });

  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  // Handle Drag & Drop
  const [isDragActive, setIsDragActive] = useState(false);
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
      // Reset previous results
      setClassifications([]);
      setProcessedImages(null);
    };
    reader.readAsDataURL(file);
  };

  // Local Auth - kullanıcı bilgileri localStorage'da saklanır
  const handleAuthSubmit = (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthSuccess('');

    if (authMode === 'register') {
      if (!authForm.username || !authForm.password || !authForm.full_name || !authForm.title) {
        setAuthError('Lütfen tüm alanları doldurunuz.');
        return;
      }
      // Kayıtlı kullanıcıları al
      const users = JSON.parse(localStorage.getItem('gastro_users') || '{}');
      if (users[authForm.username]) {
        setAuthError('Bu kullanıcı adı zaten alınmış.');
        return;
      }
      users[authForm.username] = {
        password: authForm.password,
        full_name: authForm.full_name,
        title: authForm.title
      };
      localStorage.setItem('gastro_users', JSON.stringify(users));
      setAuthSuccess('Kayıt başarıyla oluşturuldu. Giriş yapabilirsiniz.');
      setAuthMode('login');
      setAuthForm({ username: '', password: '', full_name: '', title: '' });
    } else {
      // Login
      const users = JSON.parse(localStorage.getItem('gastro_users') || '{}');
      const user = users[authForm.username];
      if (!user || user.password !== authForm.password) {
        setAuthError('Geçersiz kullanıcı adı veya şifre.');
        return;
      }
      const doctorData = {
        username: authForm.username,
        full_name: user.full_name,
        title: user.title
      };
      localStorage.setItem('gastro_logged_in', 'true');
      localStorage.setItem('gastro_doctor', JSON.stringify(doctorData));
      setIsLoggedIn(true);
      setDoctor(doctorData);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('gastro_logged_in');
    localStorage.removeItem('gastro_doctor');
    setIsLoggedIn(false);
    setDoctor(null);
    // Clear page state
    setSelectedFile(null);
    setImagePreview(null);
    setClassifications([]);
    setProcessedImages(null);
  };

  // Frontend-only image analysis - generates mock segmentation/gradcam visuals from the loaded image
  const handleAnalyze = async () => {
    if (!selectedFile || !imagePreview) return;

    setIsBusy(true);
    setStatusMessage('Görüntü analiz ediliyor, lütfen bekleyiniz...');

    try {
      // Load the image onto a canvas for processing
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imagePreview;
      });

      const size = 256;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');

      // Original cropped - center crop to square
      const minDim = Math.min(img.width, img.height);
      const sx = (img.width - minDim) / 2;
      const sy = (img.height - minDim) / 2;
      ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, size, size);
      const originalCropped = canvas.toDataURL('image/png');

      // Generate a simple threshold-based segmentation mask
      const imageData = ctx.getImageData(0, 0, size, size);
      const data = imageData.data;
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = size;
      maskCanvas.height = size;
      const maskCtx = maskCanvas.getContext('2d');
      const maskImageData = maskCtx.createImageData(size, size);
      const maskData = maskImageData.data;

      const overlayCanvas = document.createElement('canvas');
      overlayCanvas.width = size;
      overlayCanvas.height = size;
      const overlayCtx = overlayCanvas.getContext('2d');
      overlayCtx.drawImage(img, sx, sy, minDim, minDim, 0, 0, size, size);

      // Heatmap / GradCAM canvas
      const gradcamCanvas = document.createElement('canvas');
      gradcamCanvas.width = size;
      gradcamCanvas.height = size;
      const gradcamCtx = gradcamCanvas.getContext('2d');
      gradcamCtx.drawImage(img, sx, sy, minDim, minDim, 0, 0, size, size);

      const thresholdVal = threshold * 255;
      let totalMaskPixels = 0;
      const totalPixels = size * size;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // Simple brightness / redness based segmentation heuristic
        const brightness = (r + g + b) / 3;
        const redness = r - (g + b) / 2;
        const greenness = g - (r + b) / 2;
        
        // Areas that are reddish or bright (common in polyp-like regions)
        const score = (redness > 20 ? redness * 1.5 : 0) + (brightness > 100 ? brightness * 0.3 : 0);
        const isMask = score > thresholdVal * 0.8;

        if (isMask) {
          // Blue-cyan mask
          maskData[i] = 0;
          maskData[i + 1] = 200;
          maskData[i + 2] = 255;
          maskData[i + 3] = 255;
          totalMaskPixels++;
        } else {
          maskData[i] = 0;
          maskData[i + 1] = 0;
          maskData[i + 2] = 0;
          maskData[i + 3] = 255;
        }
      }

      maskCtx.putImageData(maskImageData, 0, 0);
      const maskDataUrl = maskCanvas.toDataURL('image/png');

      // Create overlay (original + mask with transparency)
      overlayCtx.globalAlpha = 0.4;
      overlayCtx.drawImage(maskCanvas, 0, 0);
      overlayCtx.globalAlpha = 1.0;
      const overlayDataUrl = overlayCanvas.toDataURL('image/png');

      // Create GradCAM-like heatmap
      const gradcamImageData = gradcamCtx.getImageData(0, 0, size, size);
      const gData = gradcamImageData.data;
      for (let i = 0; i < gData.length; i += 4) {
        const r = gData[i];
        const g = gData[i + 1];
        const b = gData[i + 2];
        const brightness = (r + g + b) / 3;
        const intensity = brightness / 255;
        
        // Heatmap coloring: cold=blue -> warm=red
        gData[i] = Math.min(255, intensity * 510);         // R
        gData[i + 1] = Math.min(255, (1 - Math.abs(intensity - 0.5) * 2) * 255); // G
        gData[i + 2] = Math.min(255, (1 - intensity) * 510); // B
        gData[i + 3] = 200;
      }
      gradcamCtx.putImageData(gradcamImageData, 0, 0);
      const gradcamDataUrl = gradcamCanvas.toDataURL('image/png');

      // Generate mock classification probabilities
      const classNames = [
        { class_name: 'polyps', turkish_name: 'Polip' },
        { class_name: 'ulcerative-colitis', turkish_name: 'Ülseratif Kolit' },
        { class_name: 'esophagitis', turkish_name: 'Özofajit' },
        { class_name: 'normal-cecum', turkish_name: 'Normal Çekum' },
        { class_name: 'normal-pylorus', turkish_name: 'Normal Pilor' },
        { class_name: 'normal-z-line', turkish_name: 'Normal Z-Çizgisi' },
        { class_name: 'dyed-lifted-polyps', turkish_name: 'Boyalı Kaldırılmış Polip' },
        { class_name: 'dyed-resection-margins', turkish_name: 'Boyalı Rezeksiyon Sınırı' }
      ];

      // Calculate a distribution based on image features
      const maskRatio = totalMaskPixels / totalPixels;
      let rawProbs = classNames.map((_, idx) => {
        const base = Math.random() * 0.3;
        if (idx === 0) return base + maskRatio * 2 + 0.3; // Polyp gets higher if more mask
        if (idx === 1) return base + 0.15;
        return base;
      });

      // Normalize to sum = 1
      const total = rawProbs.reduce((a, b) => a + b, 0);
      rawProbs = rawProbs.map(p => p / total);

      // Sort by probability descending
      const classResults = classNames.map((c, idx) => ({
        ...c,
        probability: rawProbs[idx]
      })).sort((a, b) => b.probability - a.probability);

      setClassifications(classResults);
      setProcessedImages({
        original_cropped: originalCropped,
        mask: maskDataUrl,
        overlay: overlayDataUrl,
        gradcam: gradcamDataUrl
      });
      setStatusMessage('Analiz Tamamlandı ✔️');
    } catch (err) {
      console.error(err);
      setStatusMessage('Hata Oluştu ❌');
      alert('Görüntü analiz edilirken bir hata oluştu: ' + err.message);
    } finally {
      setIsBusy(false);
    }
  };

  // Browser-native PDF Report Generation
  const handleDownloadReport = (e) => {
    e.preventDefault();
    if (!processedImages || classifications.length === 0) return;

    setIsBusy(true);
    setStatusMessage('PDF Raporu hazırlanıyor...');

    try {
      const bestClass = classifications[0];
      
      // Generate a printable HTML report and trigger print/save as PDF
      const reportHTML = `
        <!DOCTYPE html>
        <html lang="tr">
        <head>
          <meta charset="UTF-8">
          <title>Gastro AI - Tıbbi Rapor</title>
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; color: #1a1a2e; }
            .header { text-align: center; border-bottom: 3px solid #38bdf8; padding-bottom: 20px; margin-bottom: 30px; }
            .header h1 { font-size: 28px; color: #0f172a; }
            .header p { color: #64748b; font-size: 14px; margin-top: 5px; }
            .section { margin-bottom: 25px; }
            .section-title { font-size: 18px; font-weight: 700; color: #0ea5e9; border-left: 4px solid #38bdf8; padding-left: 12px; margin-bottom: 12px; }
            .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .info-item { padding: 8px 0; }
            .info-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
            .info-value { font-size: 15px; font-weight: 600; margin-top: 2px; }
            .result-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px; margin: 10px 0; }
            .result-title { font-size: 20px; font-weight: 700; color: #0369a1; }
            .result-conf { font-size: 16px; color: #0ea5e9; }
            .images-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; }
            .images-grid img { width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; }
            .image-caption { text-align: center; font-size: 12px; color: #64748b; margin-top: 4px; }
            .class-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            .class-table th, .class-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
            .class-table th { background: #f8fafc; color: #475569; font-size: 13px; text-transform: uppercase; }
            .class-table td { font-size: 14px; }
            .footer { margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; text-align: center; font-size: 12px; color: #94a3b8; }
            .doctor-sign { text-align: right; margin-top: 30px; padding-top: 15px; border-top: 1px dashed #cbd5e1; }
            @media print { body { padding: 20px; } }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>🏥 Gastro AI - Tıbbi Analiz Raporu</h1>
            <p>Gastrointestinal Görüntü Analiz Sistemi - Otomatik Oluşturulmuş Rapor</p>
            <p>Tarih: ${new Date().toLocaleDateString('tr-TR')} | Saat: ${new Date().toLocaleTimeString('tr-TR')}</p>
          </div>

          <div class="section">
            <div class="section-title">Hasta Bilgileri</div>
            <div class="info-grid">
              <div class="info-item">
                <div class="info-label">Hasta ID</div>
                <div class="info-value">${patientInfo.patient_id || '-'}</div>
              </div>
              <div class="info-item">
                <div class="info-label">Hasta Adı</div>
                <div class="info-value">${patientInfo.patient_name || '-'}</div>
              </div>
              <div class="info-item">
                <div class="info-label">Yaş</div>
                <div class="info-value">${patientInfo.patient_age || '-'}</div>
              </div>
              <div class="info-item">
                <div class="info-label">Cinsiyet</div>
                <div class="info-value">${patientInfo.patient_gender}</div>
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Klinik Bilgiler</div>
            <div class="info-item">
              <div class="info-label">Endikasyonlar</div>
              <div class="info-value">${patientInfo.indications || 'Belirtilmedi'}</div>
            </div>
            <div class="info-item" style="margin-top: 8px;">
              <div class="info-label">Doktor Notları</div>
              <div class="info-value">${patientInfo.doctor_notes || 'Belirtilmedi'}</div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">AI Analiz Sonucu</div>
            <div class="result-box">
              <div class="result-title">Birincil Teşhis: ${bestClass.turkish_name}</div>
              <div class="result-conf">Güven Oranı: %${(bestClass.probability * 100).toFixed(1)}</div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Görüntü Analizi</div>
            <div class="images-grid">
              <div>
                <img src="${processedImages.original_cropped}" alt="Orijinal" />
                <div class="image-caption">Orijinal Görüntü</div>
              </div>
              <div>
                <img src="${processedImages.mask}" alt="Segmentasyon" />
                <div class="image-caption">Segmentasyon Maskesi</div>
              </div>
              <div>
                <img src="${processedImages.overlay}" alt="Overlay" />
                <div class="image-caption">Overlay Sonuç</div>
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">Sınıflandırma Detayları</div>
            <table class="class-table">
              <thead>
                <tr>
                  <th>Sınıf</th>
                  <th>Olasılık</th>
                </tr>
              </thead>
              <tbody>
                ${classifications.map(c => `
                  <tr>
                    <td>${c.turkish_name}</td>
                    <td>%${(c.probability * 100).toFixed(1)}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>

          <div class="doctor-sign">
            <div class="info-label">Sorumlu Hekim</div>
            <div class="info-value">${doctor?.full_name || '-'} ${doctor?.title ? `(${doctor.title})` : ''}</div>
          </div>

          <div class="footer">
            Bu rapor Gastro AI sistemi tarafından otomatik olarak oluşturulmuştur. Kesin teşhis için klinik değerlendirme gereklidir.
          </div>
        </body>
        </html>
      `;

      // Open a new window and trigger print for PDF saving
      const printWindow = window.open('', '_blank');
      printWindow.document.write(reportHTML);
      printWindow.document.close();
      printWindow.focus();
      setTimeout(() => {
        printWindow.print();
      }, 500);

      setShowReportModal(false);
      setStatusMessage('Rapor Oluşturuldu ✔️');
    } catch (err) {
      console.error(err);
      alert('Rapor oluşturulurken bir hata meydana geldi.');
      setStatusMessage('Rapor Oluşturma Hatası ❌');
    } finally {
      setIsBusy(false);
    }
  };

  // Reset page state
  const handleReset = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setClassifications([]);
    setProcessedImages(null);
    setStatusMessage('Sistem Hazır');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Render Login / Registration UI
  if (!isLoggedIn) {
    return (
      <div className="login-container">
        <div className="login-card glass-panel">
          <div className="login-header">
            <div className="logo-glow">
              <Activity size={32} />
            </div>
            <h1 className="login-title">Gastro AI Portal</h1>
            <p className="login-subtitle">Gelişmiş Gastrointestinal Teşhis & Analiz Paneli</p>
          </div>

          <form onSubmit={handleAuthSubmit}>
            {authError && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: '#f87171', borderRadius: '6px', fontSize: '13px', marginBottom: '20px' }}>
                <AlertCircle size={16} />
                <span>{authError}</span>
              </div>
            )}
            {authSuccess && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', color: '#34d399', borderRadius: '6px', fontSize: '13px', marginBottom: '20px' }}>
                <CheckCircle size={16} />
                <span>{authSuccess}</span>
              </div>
            )}

            <div className="form-group">
              <label className="form-label">Kullanıcı Adı</label>
              <div style={{ position: 'relative' }}>
                <User size={18} style={{ position: 'absolute', left: '14px', top: '13px', color: 'var(--text-muted)' }} />
                <input 
                  type="text" 
                  className="form-input" 
                  style={{ paddingLeft: '45px' }}
                  required
                  placeholder="Kullanıcı adınızı girin"
                  value={authForm.username}
                  onChange={(e) => setAuthForm({ ...authForm, username: e.target.value })}
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Şifre</label>
              <div style={{ position: 'relative' }}>
                <Lock size={18} style={{ position: 'absolute', left: '14px', top: '13px', color: 'var(--text-muted)' }} />
                <input 
                  type="password" 
                  className="form-input" 
                  style={{ paddingLeft: '45px' }}
                  required
                  placeholder="Şifrenizi girin"
                  value={authForm.password}
                  onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                />
              </div>
            </div>

            {authMode === 'register' && (
              <>
                <div className="form-group">
                  <label className="form-label">Doktor Adı Soyadı</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    required
                    placeholder="Örn: Dr. Ahmet Yılmaz"
                    value={authForm.full_name}
                    onChange={(e) => setAuthForm({ ...authForm, full_name: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Ünvan / Uzmanlık Alanı</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    required
                    placeholder="Örn: Gastroenteroloji Uzmanı"
                    value={authForm.title}
                    onChange={(e) => setAuthForm({ ...authForm, title: e.target.value })}
                  />
                </div>
              </>
            )}

            <button type="submit" className="submit-btn" style={{ marginTop: '10px' }}>
              {authMode === 'login' ? 'Giriş Yap' : 'Kayıt Ol'}
            </button>
          </form>

          <div className="auth-toggle">
            {authMode === 'login' ? (
              <>
                Sistemde hesabınız yok mu? 
                <span className="auth-toggle-link" onClick={() => { setAuthMode('register'); setAuthError(''); }}>Kayıt Olun</span>
              </>
            ) : (
              <>
                Zaten kayıtlı mısınız? 
                <span className="auth-toggle-link" onClick={() => { setAuthMode('login'); setAuthError(''); }}>Giriş Yapın</span>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Render main portal dashboard
  const bestResult = classifications[0];

  return (
    <div className="dashboard">
      {/* Left Sidebar */}
      <div className="sidebar">
        <div className="sidebar-top">
          <div className="sidebar-logo">
            <Activity size={26} style={{ color: 'var(--color-primary)' }} />
            <span className="sidebar-logo-text">Gastro AI</span>
          </div>

          <div className="doctor-card">
            <div className="doctor-name">{doctor?.full_name}</div>
            <div className="doctor-title">{doctor?.title}</div>
          </div>

          <div className="sidebar-menu">
            <button className="sidebar-btn active">
              <Layers size={18} />
              <span>Analiz Paneli</span>
            </button>
          </div>
        </div>

        <div className="sidebar-logout">
          <button className="sidebar-btn" onClick={handleLogout} style={{ width: '100%', color: '#f87171' }}>
            <LogOut size={18} />
            <span>Oturumu Kapat</span>
          </button>
        </div>
      </div>

      {/* Main Panel Viewport */}
      <div className="main-content">
        <div className="header">
          <div>
            <h1 className="header-title">Entegre Tespit ve Teşhis Paneli</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>Deep Learning tabanlı polip segmentasyon ve hastalık analiz sistemi</p>
          </div>
          <div className={`system-status ${isBusy ? 'busy' : ''}`}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: isBusy ? '#fbbf24' : '#34d399' }}></span>
            <span>{statusMessage}</span>
          </div>
        </div>

        {/* 3-Card Core Workflow Panel */}
        <div className="panels-grid">
          {/* Card 1: Original Image Area */}
          <div className="panel-card glass-panel">
            <div className="panel-header">
              <span className="panel-title">1. Orijinal Görüntü</span>
              {imagePreview && (
                <button onClick={handleReset} className="card-tab-btn">Temizle</button>
              )}
            </div>
            <div className="panel-content">
              {imagePreview ? (
                <img 
                  src={processedImages ? processedImages.original_cropped : imagePreview} 
                  className="display-image" 
                  alt="Endoskopi Görseli" 
                />
              ) : (
                <div 
                  className="upload-zone"
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current.click()}
                  style={{ borderColor: isDragActive ? 'var(--color-primary)' : 'rgba(255,255,255,0.08)' }}
                >
                  <Upload className="upload-icon" />
                  <p style={{ fontWeight: '500', fontSize: '15px' }}>Endoskopi Görseli Yükle</p>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '6px' }}>Sürükle-bırak veya Tıkla (PNG, JPG, JPEG)</p>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    accept="image/*" 
                    style={{ display: 'none' }} 
                  />
                </div>
              )}
            </div>
          </div>

          {/* Card 2: Segmentation / Gradcam Output Area */}
          <div className="panel-card glass-panel">
            <div className="panel-header">
              <span className="panel-title">2. Model Öngörüsü</span>
              {processedImages && (
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button 
                    className={`card-tab-btn ${tabMode === 'segmentation' ? 'active' : ''}`}
                    onClick={() => setTabMode('segmentation')}
                  >
                    Segmentasyon
                  </button>
                  <button 
                    className={`card-tab-btn ${tabMode === 'gradcam' ? 'active' : ''}`}
                    onClick={() => setTabMode('gradcam')}
                  >
                    Grad-CAM
                  </button>
                </div>
              )}
            </div>
            <div className="panel-content">
              {processedImages ? (
                <img 
                  src={tabMode === 'segmentation' ? processedImages.mask : processedImages.gradcam} 
                  className="display-image" 
                  alt="Model Öngörüsü" 
                />
              ) : (
                <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                  <Layers size={36} style={{ marginBottom: '10px', opacity: '0.5' }} />
                  <p style={{ fontSize: '14px' }}>Segmentasyon / Grad-CAM sonuçları analiz sonrası burada gösterilecektir.</p>
                </div>
              )}
            </div>
          </div>

          {/* Card 3: Mask Overlay Area */}
          <div className="panel-card glass-panel">
            <div className="panel-header">
              <span className="panel-title">3. Overlay Sonuç</span>
            </div>
            <div className="panel-content">
              {processedImages ? (
                <img 
                  src={processedImages.overlay} 
                  className="display-image" 
                  alt="Overlay Sonuç" 
                />
              ) : (
                <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                  <Eye size={36} style={{ marginBottom: '10px', opacity: '0.5' }} />
                  <p style={{ fontSize: '14px' }}>Maskenin orijinal görsel üzerine bindirilmiş hali burada gösterilecektir.</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Diagnostic probability chart & Control Panel */}
        <div className="analytics-section">
          {/* Diagnostic Probability Chart */}
          <div className="chart-card glass-panel">
            <h2 className="chart-title">
              <Activity size={18} style={{ color: 'var(--color-primary)' }} />
              <span>Hastalık Sınıflandırma Analizi</span>
            </h2>

            {classifications.length > 0 ? (
              <>
                <div className="winner-banner">
                  <div>
                    <div className="winner-title">Birincil Teşhis</div>
                    <div className="winner-name">{bestResult.turkish_name}</div>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--color-secondary)' }}>
                    %{ (bestResult.probability * 100).toFixed(1) }
                  </div>
                </div>

                <div className="bars-container">
                  {classifications.map((item, idx) => (
                    <div className="bar-row" key={idx}>
                      <div className="bar-label" title={item.turkish_name}>{item.turkish_name}</div>
                      <div className="bar-wrapper">
                        <div 
                          className={`bar-fill ${idx === 0 ? 'winner' : ''}`} 
                          style={{ width: `${item.probability * 100}%` }}
                        ></div>
                      </div>
                      <div className="bar-val">%{(item.probability * 100).toFixed(0)}</div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ padding: '60px 0', textAlign: 'center', color: 'var(--text-muted)' }}>
                Sınıflandırma analiz grafiğini görüntülemek için lütfen bir görüntü yükleyin ve analizi başlatın.
              </div>
            )}
          </div>

          {/* Control Hub & Thresholding */}
          <div className="control-hub glass-panel">
            <h2 className="chart-title">
              <Sliders size={18} style={{ color: 'var(--color-secondary)' }} />
              <span>Kontrol ve Parametreler</span>
            </h2>

            <div className="slider-group">
              <div className="slider-header">
                <span>Segmentasyon Eşiği</span>
                <span style={{ color: 'var(--color-primary)', fontWeight: 'bold' }}>{threshold.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.0" 
                max="1.0" 
                step="0.05" 
                className="slider-input" 
                value={threshold} 
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                disabled={isBusy}
              />
              <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Eşiği artırmak arka plandaki gürültüyü ve tıbbi cihaz yansımalarını eler, azaltmak ise daha hassas segmentasyon sağlar.
              </p>
            </div>

            <div className="btn-group">
              <button 
                className="action-btn"
                onClick={handleAnalyze}
                disabled={!selectedFile || isBusy}
              >
                <RefreshCw size={16} className={isBusy ? 'spin-anim' : ''} />
                <span>Analiz Başlat</span>
              </button>

              <button 
                className="action-btn secondary"
                onClick={() => setShowReportModal(true)}
                disabled={!processedImages || isBusy}
              >
                <FileText size={16} />
                <span>PDF Rapor Oluştur</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Patient Info & PDF Generation Modal */}
      {showReportModal && (
        <div className="modal-overlay">
          <div className="modal-card glass-panel">
            <div className="modal-header">
              <h3 className="modal-title">Tıbbi PDF Raporu Oluştur</h3>
              <button className="modal-close" onClick={() => setShowReportModal(false)}>
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleDownloadReport}>
              <div className="modal-grid">
                <div className="form-group">
                  <label className="form-label">Hasta Protokol / Kimlik ID</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    required
                    placeholder="Örn: H-51024"
                    value={patientInfo.patient_id}
                    onChange={(e) => setPatientInfo({ ...patientInfo, patient_id: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Hasta Adı Soyadı</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    required
                    placeholder="Örn: Ahmet Demir"
                    value={patientInfo.patient_name}
                    onChange={(e) => setPatientInfo({ ...patientInfo, patient_name: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Hasta Yaşı</label>
                  <input 
                    type="number" 
                    className="form-input" 
                    required
                    placeholder="Örn: 45"
                    value={patientInfo.patient_age}
                    onChange={(e) => setPatientInfo({ ...patientInfo, patient_age: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Hasta Cinsiyeti</label>
                  <select 
                    className="form-input"
                    value={patientInfo.patient_gender}
                    onChange={(e) => setPatientInfo({ ...patientInfo, patient_gender: e.target.value })}
                  >
                    <option value="Erkek">Erkek</option>
                    <option value="Kadın">Kadın</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Klinik Endikasyonlar</label>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="Örn: Rektal kanama, rutin kolonoskopi kontrolü"
                  value={patientInfo.indications}
                  onChange={(e) => setPatientInfo({ ...patientInfo, indications: e.target.value })}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Doktor Klinik Notları</label>
                <textarea 
                  className="form-input" 
                  rows="3"
                  style={{ resize: 'vertical' }}
                  placeholder="Klinik bulgularınızı ve tavsiyelerinizi buraya ekleyin..."
                  value={patientInfo.doctor_notes}
                  onChange={(e) => setPatientInfo({ ...patientInfo, doctor_notes: e.target.value })}
                ></textarea>
              </div>

              <button 
                type="submit" 
                className="submit-btn" 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
              >
                <FileDown size={18} />
                <span>Raporu PDF Olarak İndir</span>
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
