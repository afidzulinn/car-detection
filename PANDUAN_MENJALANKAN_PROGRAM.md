# Panduan Lengkap Menjalankan PA-SCAN Car Detection Workstation

Dokumentasi ini mencakup langkah-langkah dari awal (persiapan lingkungan & dependensi) hingga akhir (menjalankan backend server, membuka dashboard UI, simulator dataset, serta ekstensi Chrome).

---

## 1. Prasyarat Sistem & Dependensi

Pastikan perangkat Anda sudah terinstal:
- **Python 3.10+** (disarankan menggunakan [uv](https://github.com/astral-sh/uv) untuk manajemen virtual environment yang cepat).
- **GPU NVIDIA** (opsional tapi disarankan untuk akselerasi CUDA; sistem otomatis berjalan pada mode CPU jika GPU tidak tersedia).
- **Google Chrome** (untuk menjalankan ekstensi Chrome Manifest V3 jika ingin streaming langsung dari tab browser 3D Car).

### Struktur File Utama
- [backend/app.py](file:///e:/car-detection/backend/app.py): Backend server FastAPI & WebSocket.
- [frontend/index.html](file:///e:/car-detection/frontend/index.html): Dashboard utama PA-SCAN Telemetry Workstation (Objektif 1).
- [frontend/vlm.html](file:///e:/car-detection/frontend/vlm.html): Antarmuka deskripsi VLM (Objektif 2).
- [frontend/grounding.html](file:///e:/car-detection/frontend/grounding.html): Antarmuka Visual Grounding (Objektif 3).
- [checkpoints/best_model.pt](file:///e:/car-detection/checkpoints/best_model.pt): Bobot model PA-SCAN terlatih (98.02% akurasi).
- [checkpoints/thresholds.json](file:///e:/car-detection/checkpoints/thresholds.json): Ambang batas optimal per-komponen.
- [extension/](file:///e:/car-detection/extension/): Kode sumber Chrome Extension (Manifest V3).

---

## 2. Cara Menjalankan Server Backend

Buka terminal (Command Prompt atau PowerShell) di folder proyek `e:\car-detection`:

### Opsi A: Menggunakan Launcher Cepat (Rekomendasi)
Cukup jalankan salah satu file script peluncur yang sudah disediakan:
```powershell
.\start.bat
```
atau via PowerShell:
```powershell
.\start.ps1
```

### Opsi B: Menjalankan Manual dengan `uv`
Jika ingin menjalankan uvicorn langsung di terminal:
```powershell
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Jika server berhasil berjalan, Anda akan melihat log seperti berikut:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
Model loaded from E:\car-detection\checkpoints\best_model.pt
Loaded per-label thresholds from E:\car-detection\checkpoints\thresholds.json
```

---

## 3. Mengakses Dashboard Web

Setelah server aktif, buka browser Google Chrome dan kunjungi:
👉 **[http://localhost:8000](http://localhost:8000)** (atau `http://localhost:8000/index.html`)

### Fitur-Fitur Utama di Dashboard:
1. **Live 3D Vehicle Inspection Viewport**:
   - Menampilkan kanvas visual kendaraan 3D beresolusi tinggi.
   - **Optical Viewport Zoom**: Dropdown bawaan untuk mengubah skala tampilan:
     - `1.85x Studio Close-Up (Default)`: Memperbesar kendaraan 3D dan memotong margin studio yang kosong.
     - `2.40x Door & Hood Detail`: Zoom ekstra untuk melihat bukaan pintu dan kap secara detail.
     - `1.00x Raw Frame`: Melihat ukuran penuh tangkapan layar mentah (1264x625).
   - **Show PA-SCAN ROI**: Menampilkan kotak bounding box area Region of Interest (`crop_car_roi`) yang digunakan model.
2. **Dataset Frame Simulator**:
   - Klik **Next 3D Sample** atau **Prev Sample** untuk menguji frame 3D langsung dari dataset.
   - **Camera Elevation Filter**: Saring sampel berdasarkan sudut elevasi (`All`, `Mid 15°`, `High 45°`, atau `Top 90°`).
   - **Start Auto-Stream (1.5s)**: Memutar sampel dataset secara otomatis bergantian tiap 1,5 detik (dapat di-pause kapan saja).
3. **Inspect Local Image**:
   - Klik tombol **Inspect Local Image** untuk menguji file gambar apa saja dari komputer Anda (otomatis mencocokkan Ground Truth jika gambar berasal dari `dataset/raw`).
4. **Component State Telemetry & 2D Schematic**:
   - Menampilkan status 5 komponen: `Front Left Door`, `Front Right Door`, `Rear Left Door`, `Rear Right Door`, dan `Hood`.
   - Menampilkan probabilitas prediksi model dan label verifikasi GT (`MATCH` / `MISMATCH`).
   - Diagram sasis kendaraan 2D di atas tabel akan menyala secara real-time mengikuti status pintu.
5. **Theme Switcher**:
   - Tombol **Theme: Dark / Light** di pojok kanan atas untuk beralih mode warna sesuai kenyamanan mata Anda.

---

## 4. (Opsional) Menjalankan Chrome Extension untuk Live Tab Streaming

Jika Anda memiliki halaman simulator mobil 3D yang berputar di browser dan ingin mengalirkan frame-nya langsung ke AI secara real-time:

1. Buka browser Chrome dan ketik pada address bar:
   ```text
   chrome://extensions
   ```
2. Aktifkan **Developer mode** di pojok kanan atas.
3. Klik tombol **Load unpacked** di pojok kiri atas.
4. Pilih folder:
   ```text
   e:\car-detection\extension
   ```
5. Buka tab yang menampilkan mobil 3D.
6. Klik ikon ekstensi **Car Component Streamer** di Chrome toolbar, lalu klik **Start Streaming**.
7. Buka tab dashboard `http://localhost:8000`, status koneksi akan otomatis menerima frame dan menampilkan prediksi secara langsung.

---

## 5. Menjalankan Evaluasi & Re-Training (Jika Diperlukan)

Jika Anda ingin menguji kembali metrik performa atau melatih model:
- **Evaluasi Lengkap**:
  ```powershell
  uv run python model/evaluate.py
  ```
- **Ablation Study (4 Varian Arsitektur)**:
  ```powershell
  uv run python model/ablation_study.py
  ```
- **Benchmark Efisiensi (FPS, VRAM, Latensi)**:
  ```powershell
  uv run python model/benchmark_efficiency.py
  ```
- **Training Ulang**:
  ```powershell
  uv run python model/train.py --epochs 45
  ```

---

## Ringkasan Perintah Cepat

| Kebutuhan | Perintah |
|---|---|
| **Menjalankan Program** | `.\start.bat` atau `uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload` |
| **Membuka Dashboard** | Buka browser: `http://localhost:8000` |
| **Menghentikan Server** | Tekan `Ctrl + C` di jendela terminal |
