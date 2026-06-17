# MotionDetection App (OpenCV + Tkinter)

Aplikasi Computer Vision berbasis **Python + OpenCV + Tkinter** untuk mendeteksi objek bergerak pada **video (.mp4)** maupun **webcam**.

Fitur utama:
- Background subtraction: **MOG2** dan **KNN** (user memilih melalui GUI)
- Perbandingan hasil: tampil **Original Frame**, **Foreground Mask MOG2**, **Foreground Mask KNN**, dan **Hasil Deteksi** sekaligus
- Preprocessing minimal 3 tahap: **resize**, **grayscale**, **Gaussian Blur** (+ optional thresholding)
- Noise cleanup: **morphology open/close** (erode/dilate melalui operasi morfologi)
- Deteksi dengan **contours** + **cv2.boundingRect()** dan **centroid**
- ROI (Region of Interest): user drag mouse untuk memilih area yang dihitung
- Parameter realtime melalui slider: threshold, min_area, learningRate, history, varThreshold
- Output otomatis ke folder `outputs/`:
  - Screenshot `.png` (manual via tombol)
  - Video `.mp4` (selama detection)

---

## Struktur Proyek
- `main.py` : wiring aplikasi + loop pemrosesan
- `gui.py` : antarmuka Tkinter (tombol, slider, panel output)
- `config.py` : konfigurasi parameter
- `preprocessing.py` : resize, grayscale, Gaussian Blur, optional threshold
- `background.py` : implementasi background subtraction (MOG2 & KNN)
- `detector.py` : contour detection, filtering min_area, drawing bbox + label, centroid
- `roi.py` : ROI drawing dan masking
- `utils.py` : helper FPS counter & video recorder
- `requirements.txt` : dependensi

---

## Instalasi

a) Buat environment (disarankan):
```bash

```

b) Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Menjalankan Aplikasi
```bash
python main.py
```

Langkah:
1. Klik **Open Video** untuk memilih file `.mp4` atau **Webcam**
2. (Opsional) Drag mouse pada canvas untuk membuat **ROI**
3. Tekan **Start Detection**
4. Ubah slider untuk melihat perubahan hasil secara realtime
5. Tekan **Save Screenshot** untuk menyimpan frame teranotasi
6. Tekan **Stop Detection** untuk menghentikan proses

Output tersimpan di:
- `outputs/` : `detection-YYYYmmdd-HHMMSS.mp4`
- `outputs/screenshot-YYYYmmdd-HHMMSS.png`

---

## Penjelasan Algoritma Singkat

### Preprocessing
1. **Resize**: menyeragamkan lebar frame agar performa stabil
2. **Grayscale**: konversi ke citra satu kanal
3. **Gaussian Blur**: mengurangi noise frekuensi tinggi
4. (Optional) **Thresholding**: mengubah ke citra biner jika diaktifkan di config

### Background Subtraction
- **MOG2**: memodelkan distribusi intensitas piksel menggunakan mixture model untuk mendeteksi foreground.
- **KNN**: memodelkan background dengan k-nearest neighbors pada fitur intensitas.

Kedua metode menghasilkan **foreground mask** yang kemudian dibersihkan dengan morphology (open + close).

### ROI
ROI dibentuk dengan drag mouse. Mask ROI diterapkan pada foreground mask agar hanya objek di dalam area yang dihitung.

### Deteksi Gerak
- `cv2.findContours()` pada foreground mask
- `cv2.boundingRect()` untuk bounding box
- filter berdasarkan `min_area`
- centroid dihitung dari momen kontur (`cv2.moments`)

---

## Catatan Implementasi
- Bagian perbandingan mask selalu dihitung (MOG2 dan KNN) supaya tampilan komparatif tersedia.
- Status:
  - **Motion Detected** jika jumlah objek > 0
  - **No Motion** jika jumlah objek == 0

---

## Contoh Output
- Video: `outputs/detection-20260101-120000.mp4`
- Screenshot: `outputs/screenshot-20260101-120000.png`

