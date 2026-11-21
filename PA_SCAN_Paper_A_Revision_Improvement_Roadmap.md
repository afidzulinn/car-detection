# MASTER ROADMAP — Revisi dan Improvement PA-SCAN untuk Paper A

**Working title:** *Visibility-Aware and Component-Specific Pose-Conditioned Learning for Robust Vehicle Component State Recognition from Synthetic 3D Images*  
**Repository:** https://github.com/afidzulinn/car-detection  
**Tanggal audit sumber:** 20 November 2026  
**Status:** Dokumen perencanaan penelitian dan engineering; **belum** membuktikan peningkatan performa PA-SCAN v2.  
**Ruang lingkup:** **Paper A — Computer Vision / Deep Learning**, bukan Paper B (AI-agent workflow).

> **Cara menggunakan:** kerjakan tahap P0 → P1 → P2 → P3, jalankan eksperimen yang ditentukan, dan isi kolom hasil dengan angka nyata. Item dengan `[VERIFIED-CODE]` didukung oleh kode publik yang diperiksa; `[HYPOTHESIS]` perlu dibuktikan dengan eksperimen; `[VERIFY-RUNTIME]` tidak dapat disimpulkan dari source code saja. Tautan sumber kode dan literatur ada di bagian akhir.

---

## 0. Ringkasan eksekutif

PA-SCAN saat ini adalah custom CNN multi-label yang memprediksi lima status biner: `front_left_door`, `front_right_door`, `rear_left_door`, `rear_right_door`, dan `hood`. Arsitekturnya memakai residual CNN, CoordConv, perhatian kanal/spasial, multi-scale fusion, FiLM pose conditioning, dan lima classifier heads. Target koleksi data dasarnya adalah **32 kombinasi status × 8 azimuth × 3 elevasi = 768 render asli**. Perkalian ini adalah **desain koleksi**, bukan verifikasi bahwa seluruh file dataset tersedia dan valid. [S1–S4]

Benchmark yang tertulis di README (bukan hasil re-run audit ini):

| Metrik validasi pada README | Nilai dilaporkan |
|---|---:|
| Mean per-label accuracy @ threshold 0,5 | 94,69% |
| Mean per-label accuracy @ optimal threshold | 96,56% |
| Exact-match @ threshold 0,5 | 73,44% |
| Exact-match @ optimal threshold | 82,81% |
| Mean ROC-AUC | 0,9934 |
| Rear Left Door accuracy @ optimal threshold | 89,06% |
| Hood accuracy @ optimal threshold | 93,75% |

**Jangan gunakan 96,56% sebagai independent-test result.** `model/evaluate.py` mencari threshold F1 terbaik pada validation set, lalu melaporkan hasil threshold tersebut pada set yang sama. Validasi yang memakai slot pose tertentu per kombinasi juga berbeda dari global held-out-angle evaluation. [S1, S5, S6]

### Prioritas akhir

| Level | Fokus | Tujuan / keputusan |
|---|---|---|
| **P0 — wajib** | Perbaikan metadata elevasi, label, split, evaluasi, threshold, dan reproducibility | Menghasilkan baseline ilmiah yang sahih |
| **P1 — kuatkan baseline** | Adaptive ROI, variasi render, augmentasi, baseline pembanding, fair ablation | Mengisolasi sumber kegagalan dan mengetahui peningkatan tanpa modul baru |
| **P2 — kandidat novelty** | Elevation/pose conditioning, component-aware attention, visibility-aware learning, part-mask supervision | Menguji hipotesis PA-SCAN v2 secara terkontrol |
| **P3 — pembuktian robustness** | Unseen viewpoint, unseen mesh, occlusion, corruption, explainability, calibration, efficiency | Mendukung klaim robustness pada paper |
| **P4 — di luar Paper A** | AI agent, VLM, grounding, WebSocket orchestration | Disimpan untuk Paper B atau appendix/deployment |

**Hipotesis utama yang diusulkan:** representasi fitur yang *spesifik terhadap komponen* dan *dikondisikan oleh pose serta visibilitas* dapat menurunkan kesalahan pada viewpoint sulit dibandingkan PA-SCAN standar, tanpa overhead komputasi yang tidak proporsional. Hipotesis ini **belum terbukti**.

---

## 1. Audit temuan terkonfirmasi dan dampaknya

### P0-01 — Metadata elevasi `top` tersaring oleh loader

- **Status:** `[VERIFIED-CODE]`; dampak pada checkpoint tertentu `[VERIFY-RUNTIME]`.
- **File:** `data_collection/collect_data.py`, `model/dataset.py`.
- **Bukti:** collector menyimpan `mid`, `high`, `top`; `_split()` pada dataset memfilter ke `{'mid','high','Slightly High','Top-Down'}` saat `clean_only=True`. Nilai `top` tidak ada. [S2, S3]
- **Risiko:** sampel elevasi `top` keluar dari training/validation default, sementara dokumentasi seolah-olah mengevaluasi tiga elevasi. Kesalahan pada contoh top-down **konsisten** dengan masalah ini tetapi **tidak membuktikan hubungan kausal**.
- **Revisi:** buat enumerasi/normalisasi elevasi kanonis (`mid`, `high`, `top`) dan translasi eksplisit alias legacy. Jangan menyeleksi data dengan daftar string tersembunyi. Keluarkan report sebelum dan sesudah filtering.
- **Uji:** untuk dataset lengkap, hitung frekuensi `(state_combo, azimuth, elevation)`; jika memang koleksi lengkap, harus ada 256 original untuk tiap elevasi (32×8). Unit-test semua alias.
- **Definition of done (DoD):** tidak ada drop tanpa reason code; total per-elevasi di manifest tepat dan dapat diaudit.

### P0-02 — Koreksi label legacy dilakukan dengan heuristik rentang state/view

- **Status:** `[VERIFIED-CODE]`.
- **File:** `model/dataset.py` fungsi `_is_legacy_combo0_fl_open` dan `_repair_item`.
- **Bukti:** komentar menyebut dua file legacy, namun predicate memilih *semua* record dengan seluruh label 0, elevasi `mid`, dan sudut `(225,270)`, bukan ID file spesifik. [S3]
- **Risiko:** data baru yang secara sah *all-closed* pada slot sama dapat diubah menjadi label front-left OPEN.
- **Revisi:** lakukan perbaikan di manifest migrasi berbasis `sample_id`/filename + versi dataset + audit visual; **jangan** melakukan silent repair di `__getitem__`/loader. Simpan original label, corrected label, alasan, dan reviewer.
- **Uji/DoD:** sample baru pada kombinasi all-closed tidak berubah; hanya record legacy teridentifikasi yang dapat diperbaiki; seluruh perubahan terlacak.

### P0-03 — Split train/validation belum menyediakan test set independen

- **Status:** `[VERIFIED-CODE]`.
- **File:** `model/dataset.py`, `model/evaluate.py`.
- **Bukti:** `_split()` memiliki cabang eksplisit `split == 'val'`; selain itu mengembalikan train. `get_dataloaders()` hanya menghasilkan train/val. Jika seseorang memberi `split='test'`, pada implementasi saat ini berisiko mengembalikan data training tanpa error. [S3, S5]
- **Revisi:** gunakan `split_manifest.json` berisi tiga partisi eksplisit; wajib error untuk split tidak dikenal; jangan membuat ulang split dari seed yang tersebar pada masing-masing script.
- **Uji/DoD:** train ∩ val = train ∩ test = val ∩ test = ∅ pada **source scene/render group**; `split='test'` mengembalikan test sungguhan; unknown split menimbulkan exception.

### P0-04 — Validasi threshold pada data yang sama mengandung selection bias

- **Status:** `[VERIFIED-CODE]`.
- **File:** `model/evaluate.py` fungsi `find_optimal_threshold()`.
- **Bukti:** threshold optimal F1 dipilih dari label/probabilitas validation, kemudian skor optimal threshold dihitung kembali pada validation yang sama. [S5]
- **Revisi:** pilih checkpoint dan threshold hanya pada train/validation (atau calibration subset), *freeze* threshold, lalu laporkan test set **sekali**. Tampilkan threshold 0,5 dan tuned secara terpisah.
- **Uji/DoD:** test labels tidak dibaca di tahap tuning; thresholds disimpan bersama dataset split ID dan checkpoint hash; tidak ada penggunaan test untuk memilih hyperparameter.

### P0-05 — Klaim “unseen camera angles” perlu dikoreksi

- **Status:** `[VERIFIED-CODE]`.
- **File:** `model/dataset.py`, `README.md`, docs.
- **Bukti:** split mengacak slot `(elevation, angle)` *per foreground state group*, bukan menahan satu sudut secara global untuk semua kombinasi status. Sudut tertentu bisa muncul di training untuk state lain. [S3, S1]
- **Revisi:** namai evaluasi saat ini `held-out state-view combination` atau `within-grid held-out slots`. Buat protokol `global-held-out azimuth` terpisah (contoh: azimuth 90° dan 270° tidak muncul di training sama sekali).
- **DoD:** setiap tabel memuat definisi split; cek `train_angles ∩ test_angles == ∅` khusus protokol global held-out.

### P0-06 — ROI statis tidak otomatis menjamin seluruh komponen terlihat

- **Status:** fixed crop `[VERIFIED-CODE]`; besarnya cropping loss `[VERIFY-RUNTIME]`.
- **File:** `model/architecture.py` (`ROI_BOX_NORM=(0.17,0.19,0.83,0.62)`), `model/dataset.py`, `backend/app.py`.
- **Risiko:** kap/pintu pada elevasi `top` atau konfigurasi OPEN tertentu dapat terpotong; menambah resolusi model tidak mengembalikan informasi yang sudah hilang. [S4, S7]
- **Revisi:** benchmark `full frame`, `fixed ROI`, `content-aware vehicle ROI` dengan margin, dan `oracle ROI` dari mask renderer khusus upper bound. Pastikan transform crop yang sama di training/evaluation/inference.
- **Uji:** hitung `component visible-pixel retention` sebelum/sesudah crop dan persentase kasus komponen keluar dari crop, per elevasi.
- **DoD:** preprocessing tidak menghilangkan objek target tanpa ada penanda visibilitas; mode ROI terdokumentasi di checkpoint/config.

### P0-07 — Dokumentasi dan hasil historis saling bertentangan

- **Status:** `[VERIFIED-CODE/DOC]`.
- **File:** `README.md`, `docs/PRESENTATION_SLIDE_DECK.md`, `notebook_training_colab.ipynb` (perlu audit metadata run lokal).
- **Bukti:** README mengklaim 96,56% optimal validation, sementara slide deck memuat tabel hasil lama dengan exact-match 0% dan konfigurasi model/training berbeda. [S1, S8]
- **Revisi:** beri label `historical/legacy` dan `current/reproduced`; siapkan satu `results/experiment_registry.csv` dengan commit, dataset version, split, seed, checkpoint, metrik, threshold, dan timestamp.
- **DoD:** setiap angka pada abstrak, tabel, README, dan figure dapat dilacak ke satu run + artefaknya.

### P0-08 — Konvensi decision threshold tidak sepenuhnya konsisten

- **Status:** `[VERIFIED-CODE]`.
- **File:** `model/architecture.py` (`predict`), `model/evaluate.py`, `model/gradcam.py`, `model/change_detector.py`, `backend/app.py`.
- **Bukti:** kode menggunakan campuran `>` dan `>=`; Grad-CAM memakai threshold 0,5, sedangkan change detector menggunakan threshold optimal dan hysteresis. [S4–S7, S9]
- **Revisi:** satu utilitas `decision_policy.py` dengan fungsi `predict_from_prob`, `calibrate_threshold`, dan `hysteresis_update`; dokumentasikan hubungan `prob_open`, `predicted_state`, dan `confidence_predicted_class`.
- **DoD:** unit tests untuk probabilitas tepat 0,5, threshold optimal, dan hysteresis boundaries; hasil statis evaluasi sama dengan inference ketika menggunakan kebijakan yang sama.

### P0-09 — Ground truth collector belum diverifikasi sebagai state mesin 3D aktual

- **Status:** `[HYPOTHESIS/VERIFY-RUNTIME]`, bukan bug yang sudah terbukti.
- **File:** `data_collection/collect_data.py`.
- **Bukti kode:** `apply_state()` melakukan click tombol bagi target OPEN; `save_sample()` menyimpan **target_state** sebagai label. [S2]
- **Risiko:** jika suatu click tidak berhasil, animasi belum selesai, atau state renderer berbeda dari tombol yang dikirim, gambar dan label dapat tidak cocok.
- **Revisi:** expose state dari 3D scene/DOM bila aplikasi sumber memungkinkan, tunggu completion animasi, buat screenshot QA; simpan `requested_state`, `observed_state`, dan flag `verified`.
- **DoD:** random sample audit dengan pelacakan ke scene state atau pemeriksaan manual terstratifikasi; error label tercatat dan diperbaiki di dataset version baru.

### P0-10 — Reproducibility artefak dataset dan model belum cukup kuat

- **Status:** `[VERIFIED-CODE]` sebagian; dataset/checkpoint `[NOT AVAILABLE FOR AUDIT]`.
- **Bukti:** `.gitignore` mengecualikan `dataset/`, `checkpoints/`, dan file bobot. Requirement mayoritas memakai batas minimal `>=`; `train.py` mencatat seed, tetapi `torch.backends.cudnn.benchmark=True` sehingga determinisme lintas run tidak dijamin. [S10, S11, S12]
- **Revisi:** dataset manifest dengan SHA256 per file; environment lock; PyTorch/CUDA/GPU details; checkpoint metadata; optional deterministic run mode; log validation/test split signature.
- **DoD:** dari clone bersih + data yang diotorisasi, seluruh laporan dan tabel dapat dibangun ulang; toleransi variasi numerik dijelaskan.

---

## 2. Checklist P0 — keluaran sebelum model baru

- [ ] `P0-01` Normalisasi metadata `top`/legacy; verifikasi 32×8×3 slot.
- [ ] `P0-02` Pisahkan label correction dari loader; simpan audit trail.
- [ ] `P0-03` Buat train/val/test manifest eksplisit dan bebas leakage.
- [ ] `P0-04` Kalibrasi threshold hanya pada validation/calibration split.
- [ ] `P0-05` Definisikan dan labeli ulang evaluasi *held-out slots* vs *held-out viewpoint*.
- [ ] `P0-06` Ukur kehilangan area komponen akibat fixed ROI.
- [ ] `P0-07` Sinkronkan dokumentasi, benchmark, dan experiment registry.
- [ ] `P0-08` Samakan threshold policy (`>` atau `>=`) dan confidence semantics.
- [ ] `P0-09` QA state UI dan ground-truth status pada renderer.
- [ ] `P0-10` Simpan hash, seed, dependencies, dan metadata checkpoint.

**Gate P0:** dilarang mengklaim PA-SCAN v2 lebih baik sebelum E0 (baseline baru) selesai dengan test independen dan setiap data version terverifikasi.

---

## 3. P1 — Perkuat data, preprocessing, dan baseline

### P1-01 — Dataset coverage dan balance report

**File:** tambah `data_collection/audit_dataset.py`, revisi `model/dataset.py`.

- Hitung jumlah foto asli per `(combo_id, angle, elevation)`, distribusi lima label, missing/corrupt paths, duplicate SHA, ukuran/resolusi, rasio foreground, serta histori revisi label.
- Bedakan **original render** dengan **augmented sample**; jangan menganggap 6.144 gambar augmentasi sebagai 6.144 scene independen.
- Plot matrix coverage 32×(8×3), laporan komponen per pose, dan daftar kasus bermasalah.
- **DoD:** report JSON/CSV reproducible, no silent filtering.

### P1-02 — Split aman terhadap variasi turunan

**File:** tambah `model/splits.py` dan `configs/splits/*.yaml`.

- Split pada **source render ID / scene group** lebih dulu; augmentasi **hanya** pada data train.
- Untuk multi-mesh: group berdasarkan mesh saat mengevaluasi `unseen vehicle`.
- Untuk global-held-out-view: tetapkan sudut/elevasi test sebelum augmentasi.
- Buat report intersections dari `source_id`, `scene_id`, `mesh_id`, angle dan image hash.
- **DoD:** tidak ada variant turunan source scene yang sama masuk ke beberapa split.

### P1-03 — Data augmentasi yang relevan

**File:** `data_collection/augment_data.py`, `model/train.py`.

- Terdapat tujuh augmentasi di kode: brightness high/low, motion blur, Gaussian noise, hue, perspective, JPEG. Komentar header script masih menyebut enam jenis; sesuaikan. [S13]
- `model/train.py` memiliki `--max-aug 0` sebagai default: offline augmentation yang telah dibuat belum tentu dipakai. [S12]
- Audit kompatibilitas parameter Albumentations untuk versi terpasang; script saat ini menangkap exception augmentasi dan melanjutkan, sehingga jumlah keluaran dapat berkurang tanpa gagal keras.
- Jalankan eksperimen `none`, `online photometric`, `offline+online`, `geometry-safe`, `domain randomized rendering` secara terpisah.
- Hindari horizontal flipping tanpa menukar label kiri/kanan dan memverifikasi orientasi kamera.
- **DoD:** jumlah augmented per source terlapor; tidak ada sampel invalid; peningkatan tidak hanya berasal dari duplikasi data.

### P1-04 — Perbandingan kebijakan ROI

**File:** refactor `crop_car_roi()` menjadi `model/preprocess.py`.

- A0: fixed ROI saat ini.
- A1: full-frame resize, untuk menunjukkan risiko informasi terpotong.
- A2: adaptive foreground bounding box (klasik: background subtraction/contour, bila layak).
- A3: renderer/object mask bounding box + margin (oracle; **tidak boleh dipakai sebagai input test biasa** kecuali tersedia di deployment).
- A4: component-specific learned crop (opsional setelah baseline stabil).
- **DoD:** tabel akurasi, retention dan latency per ROI; tidak memilih crop berdasarkan test label.

### P1-05 — Baseline pembanding yang fair

**File:** tambah `model/baselines.py`, `configs/baselines/*.yaml`.

Minimal:

1. Small standard CNN, **from scratch**.
2. Residual CNN dengan input dan split identik, **from scratch**.
3. ResNet-18, MobileNetV3-Small (from scratch **dan**, jika dipilih, pretrained sebagai kategori terpisah).
4. PA-SCAN v1 yang *reproduced* setelah P0.

Jangan membandingkan pretrained vs non-pretrained sebagai bukti bahwa satu attention module lebih unggul. Catat optimizer, budget epoch, early stopping, augmentasi, dan ukuran input yang setara; laporkan parameter serta FLOPs.

### P1-06 — Fair ablation: pisahkan modul yang sekarang saling tergantung

**File:** `model/architecture.py`.

Saat ini `use_coordconv=False` sekaligus mengganti residual blocks ke plain convolution, menghapus multi-scale fusion, dan mengganti pooling. Ini adalah **ablation gabungan**, bukan isolated CoordConv ablation. [S4]

Pisahkan flag:

```python
ModelConfig(
    backbone="residual",     # residual | plain
    use_coordconv=True,
    use_multiscale=True,
    use_channel_attention=True,
    use_spatial_attention=True,
    use_pose_film=True,
    pooling="hybrid_grid",  # global_avg | hybrid_grid
)
```

- Pastikan flag tertentu hanya mengubah satu family modul.
- Dokumentasikan perbedaan parameter/FLOPs. Bila kapasitas berubah banyak, tambahkan *parameter-matched control*.
- **DoD:** tabel ablation dapat mengatribusikan perubahan performa ke modul yang tepat.

### P1-07 — Audit kemampuan pose head

**File:** `model/architecture.py`, `model/train.py`, `model/evaluate.py`.

- Saat ini pose auxiliary head mengeluarkan delapan kelas azimuth saja; `pose_probs` di-`detach()` sebelum masuk generator FiLM. [S4]
- Tambahkan evaluasi pose classification, confusion matrix, dan akurasi per elevasi. Uji FiLM dari true pose (oracle), predicted pose (deployable), dan no-pose.
- Lakukan *ablation detach/non-detach* dengan pelaporan potensi perubahan gradien multi-task; jangan mengasumsikan non-detach selalu lebih baik.
- **DoD:** manfaat pose terhadap state classification terukur, bukan hanya diasumsikan.

---

## 4. P2 — Proposal arsitektur PA-SCAN v2 (kandidat kontribusi paper)

> **Peringatan novelty:** CoordConv, CBAM-inspired attention, FiLM, multi-scale fusion, part attention, maupun visibility-aware learning telah memiliki pendahulu di literatur. *Kontribusi baru* harus berupa desain spesifik yang dibedakan dengan penelitian sebelumnya dan dibuktikan lewat eksperimen, bukan sekadar merangkai modul. [L1–L8]

### P2-01 — Elevation-aware / continuous camera pose conditioning

**Hipotesis:** representasi pose yang mengetahui elevasi dan struktur sudut dapat mengurangi kesalahan pada kamera high/top dan azimuth sulit.

**Implementasi yang diuji:**

- **B0:** original 8-class azimuth softmax + FiLM.
- **B1:** auxiliary heads 8-class azimuth + 3-class elevation.
- **B2:** pose embedding continuous `[sin(azimuth), cos(azimuth), elevation_norm]`, dengan regress/encode saat sesuai.
- **B3:** ground-truth camera pose ke FiLM (oracle upper bound; tidak boleh dibandingkan sebagai deployable bila kamera metadata tidak tersedia di inference).
- **B4:** predicted pose vs externally supplied camera metadata jika memang tersedia saat inference.

**Contoh interface (sketsa, belum kode final):**

```python
state_logits, aux = model(
    images,
    camera_pose=None,   # opsional saat deployment
    return_aux=True,
)
# aux: azimuth_logits, elevation_logits atau pose_regression
```

**Ablation:** cek pengaruh conditioning, supervision, dan penggunaan true/predicted pose secara terpisah.  
**DoD:** worst-elevation F1 serta exact-match tidak memburuk signifikan pada split independen, atau tradeoff jelas dilaporkan.

### P2-02 — Component-specific spatial feature extraction

**Hipotesis:** classifier yang menggunakan fitur terarah untuk setiap komponen dapat mengurangi confounding dari komponen lain/background.

**Sketsa:**

```text
RGB -> residual features F4/F5 -> multi-scale fused map F
                                      |
             +------------------------+-----------------------+
             |                        |                       |
       learned part query 1     part query 2             part query 5
             |                        |                       |
        attention A1              attention A2             attention A5
             |                        |                       |
          pooled F1                 pooled F2                pooled F5
             +------------------- 5 state logits --------------+
```

- Gunakan **part embeddings** berbeda untuk FL, FR, RL, RR, Hood; setiap peta `A_k` berukuran spasial.
- Pertahankan ringkasan konteks global sebagai kontrol eksperimen; pintu tertutup mungkin hanya dapat dikenali dari hubungan dengan bodi.
- Bandingkan shared global features, independent MLP heads (PA-SCAN v1), dan truly part-specific spatial pooling.
- Tambahkan regularisasi anti-collapse hanya jika ablation mendukung.
- **DoD:** tidak hanya accuracy naik; part-mask localization atau causal occlusion test juga mendukung interpretasi.

### P2-03 — Visibility-aware learning

**Hipotesis:** mengetahui apakah komponen secara visual dapat diamati membantu model mengelola prediksi yang ambigu karena occlusion/truncation.

**Label tambahan per komponen:**

- `state`: 0 CLOSED / 1 OPEN (status fisik).
- `visible_mask`: piksel permukaan komponen yang benar-benar terlihat.
- `amodal_mask`: proyeksi keseluruhan komponen, termasuk yang tersembunyi (bila renderer memungkinkan).
- `visibility_ratio`: `area(visible_mask) / max(area(amodal_mask), eps)`; dokumentasikan definisi jika amodal di luar frame.
- `truncation_ratio`: terpotong frame/crop, dibedakan dari occlusion oleh komponen lain.
- `visibility_quality`: visible / partially visible / not observable (threshold ditetapkan pada validation atau menurut protokol anotasi, bukan ditentukan dari test outcome).

**Model:** lima state logits + lima visibility scores atau visibility-head classification. Pertimbangkan *uncertainty-aware selective prediction* dengan opsi `uncertain/need another view` pada kondisi sulit.

**Catatan metodologis:** komponen tidak terlihat **bukan berarti CLOSED**. Jika state tidak dapat disimpulkan dari satu frame, jangan menjanjikan klasifikasi sempurna; evaluasi *coverage versus risk* penting. Mask ground truth digunakan untuk **training/evaluation**, bukan sebagai input inference yang tidak tersedia.

**DoD:** perbaikan diuji per visibility bin dan confidence calibration, dengan coverage dilaporkan; tidak ada peningkatan semu karena mengecualikan kasus sulit tanpa menghitungnya.

### P2-04 — Part-mask auxiliary supervision

Bila akses ke renderer menyediakan object ID/depth/segmentation, gunakan mask sebagai pengawasan training.

```text
L_total = L_state + lambda_pose * L_pose
                  + lambda_vis * L_visibility
                  + lambda_mask * L_mask
```

- `L_state`: BCEWithLogits lima output biner; pos_weight dihitung dari train saja.
- `L_pose`: azimuth CE atau circular regression; elevation CE/regression sesuai desain.
- `L_visibility`: BCE/regression dari label visibility.
- `L_mask`: BCE/Dice/attention alignment untuk mask visible/part target.
- Uji tiap lambda, normalisasi antar-loss, dan kapan loss dipakai jika mask tidak tersedia.
- **Ablation:** `no-mask`, `visible-mask`, `amodal-mask` (jika tersedia), `oracle ROI`, dan `predicted attention`.
- **DoD:** jelas apa yang dipakai pada training vs inference; mask labels dapat direproduksi dan diperiksa kualitasnya.

### P2-05 — Multi-scale component pooling

- Stage 4 memiliki feature map 14×14 sedangkan Stage 5 7×7; detil geometri pintu/engsel mungkin memerlukan fitur lebih halus. [S4]
- Uji pooling dari `F4`, `F5`, fusion, atau neck ringan (FPN-like) dengan part-aware queries.
- Laporkan resolusi input 224 vs 320/384 sebagai eksperimen terpisah dari perubahan neck.
- **DoD:** keuntungan lokasi fitur terhadap biaya komputasi terukur.

### P2-06 — Background-dependence dan spurious correlation mitigation

- Renderer dengan background statis dapat mendorong shortcut learning. Ini **hipotesis**, bukan kesimpulan hanya dari Grad-CAM.
- Uji texture/lighting/background swaps **dengan status fisik tetap**, serta conditional randomization saat train.
- Lakukan occlusion/patch perturbation di foreground dan background, lalu ukur perubahan state logits, bukan sekadar warna heatmap.
- **DoD:** laporan perubahan akurasi pada background OOD vs kondisi asli dan uji faithfulness.

### P2-07 — Confidence calibration dan abstention

- Jika status output OPEN probability `p`, `confidence_predicted_class = p` saat pred OPEN dan `1-p` saat CLOSED. Nilai ini tidak otomatis probabilitas bahwa model benar.
- Bandingkan calibration: none, temperature scaling (yang sesuai multi-label logits), dan threshold calibration per label.
- Metrik: Brier score, ECE beserta reliability diagram, coverage-risk, false-acceptance untuk low-visibility.
- **DoD:** threshold dikalibrasi pada validation/calibration split, tidak di-retune pada test.

### P2-08 — *Scope freeze* untuk paper utama

**Direkomendasikan sebagai kombinasi inti, jika ablation positif:**

1. Elevation-aware pose conditioning.
2. Component-specific attention/pooling.
3. Visibility-aware auxiliary prediction / selective inference.

**Opsional:** part-mask supervision jika masks dapat diperoleh tanpa menciptakan ketergantungan pada informasi oracle saat inference.  
**Jangan jadikan tiga modul wajib sekaligus jika datanya belum tersedia:** paper dengan satu kontribusi yang terukur lebih kuat daripada banyak blok tanpa isolasi efek.

---

## 5. P3 — Rancangan benchmark robustness yang layak dipublikasikan

### 5.1 Protokol dataset / split

Gunakan *disjoint experimental protocols*, bukan menggabungkan seluruh hasil ke satu angka:

| Kode | Protokol | Apa yang dibuktikan | Catatan |
|---|---|---|---|
| **T0** | Held-out state-view samples, source-group disjoint | Generalisasi interpolasi kombinasi render dalam grid yang dikenal | Hanya cocok sebagai within-domain baseline |
| **T1** | Global held-out azimuth, misalnya seluruh 90° dan 270° | Generalisasi sudut horizontal yang tidak dipakai training | Pastikan sudut test benar-benar tidak ada di train |
| **T2** | Leave-one-elevation-out (`top` sebagai test) | Ekstrapolasi perspektif top-down | Untuk run ini, jangan melatih pada elevasi top |
| **T3** | Unseen 3D vehicle meshes | Generalisasi antar-geometri kendaraan | **Tidak bisa diklaim** jika dataset hanya punya satu kendaraan |
| **T4** | Lighting/background/material OOD | Robustness domain visual | Buat scene variasi baru, bukan cuma augment versi test yang sama |
| **T5** | Visibility/occlusion bins | Ketahanan terhadap komponen sebagian tersembunyi | Perlu annot visibility valid |
| **T6** | Blur/JPEG/noise/crop shifts | Corruption robustness | Lapor severity dan penurunan metrik |

**Penting:** `top` boleh masuk training pada T0/T1, tetapi harus dikeluarkan dari training pada T2. Ini protokol eksperimen yang berbeda, bukan kontradiksi.

### 5.2 Metrik primer dan sekunder

**Primer (tentukan sebelum eksperimen):**

- **Exact-match accuracy**: seluruh lima komponen benar pada gambar yang sama.
- **Macro-F1 antar lima komponen**: rata-rata F1 per label, dengan pengelompokan evaluasi yang sama.
- **Worst-view macro-F1**: minimum performa antar viewpoint/elevasi yang memiliki jumlah sampel memadai.

**Sekunder:**

- Per-component precision, recall, F1, AUROC, average precision.
- Balanced accuracy, confusion matrix per komponen dan per view.
- Robustness drop `metric_clean - metric_shifted` dan relative drop bila jelas denominatornya.
- Coverage-risk, ECE, Brier score, confusion visibility.
- Parameter count, MACs/FLOPs (sebut konvensi), peak VRAM, latency p50/p95, FPS dengan hardware dan batch size jelas.
- Grounding/attention: energy-in-mask, pointing game, deletion/insertion atau komponen perturbation.

### 5.3 Statistik dan keadilan perbandingan

- Gunakan **minimal tiga seed** untuk perbandingan utama; lebih banyak bila variance besar.
- Model dibandingkan pada **identical train/val/test manifests**, label repair, dan image preprocessing untuk eksperimen yang memang ingin mengisolasi modul.
- Laporkan mean ± standard deviation antar-seed; confidence interval dapat dihitung dengan bootstrap **berdasarkan source scene / mesh group**, bukan memperlakukan variasi augmentasi sebagai pengamatan independen.
- Hindari klaim "significant" tanpa desain uji yang tepat; untuk pasangan prediksi pada test sama, pertimbangkan paired bootstrap atau McNemar (sesuai metrik).
- Hanya pilih hyperparameter dan architecture variant berdasarkan validation. Bekukan semuanya sebelum tes final.

### 5.4 Diagnostik error yang harus dibuat

Hasilkan tabel kegagalan berikut:

1. Label × elevation × angle → F1, number of positives/negatives.
2. Label × visibility bin → F1, confidence, number of observations.
3. Crop retention × prediction correctness.
4. `P(OPEN)` distributions untuk TP/TN/FP/FN.
5. Sisi dekat vs sisi jauh relatif kamera dan prediksi salah.
6. Kasus input berpasangan: **state 4 komponen sama, hanya 1 pintu ditoggle**, pose/render seed sama. Ini *counterfactual sensitivity* yang sangat berguna untuk memisahkan sinyal komponen dari konteks.
7. Failure gallery dengan ROI asli, prediction/GT, camera pose, part mask, dan attribution target yang tepat.

---

## 6. Revisi explainability / Grad-CAM

**Bukti kode:** `model/gradcam.py` meng-hook `stage5` (7×7) dan membackprop `logits[0,class_idx]` untuk **OPEN**. Warna border panel hijau menandakan prediksi benar, merah menandakan salah; overlay merah bukan status OPEN. Normalisasi min/max dilakukan per heatmap. [S9]

### G0 — Perbaiki semantics visualisasi

- Tampilkan `GT`, `P(OPEN)`, `P(CLOSED)=1-P(OPEN)`, `predicted_state`, `decision_threshold`, dan `match` secara terpisah.
- Border hijau/merah hanya menandakan correct/incorrect; gunakan legenda jelas.
- Jangan melabeli `P(OPEN)=0.01` sebagai `confidence=1%` untuk prediksi CLOSED.
- Pada score target CLOSED, gunakan `-logit_open` sebagai target eksplisit (binary symmetric score) dan jelaskan bahwa ini bukan kelas sigmoid kedua yang dilatih secara terpisah.

### G1 — Bandingkan layer

- Hook `stage1_4[-1]` (Stage 4), `stage5`, serta fitur fusion (instrumentasikan model jika perlu).
- Tampilkan informasi resolusi map (14×14 vs 7×7) dan heatmap overlay pada gambar ROI yang identik.
- Bandingkan atribut OPEN dan CLOSED pada sample salah dan benar.

### G2 — Kuantifikasi spasial

Untuk heatmap positif H dan binary component mask M:

```text
energy_in_mask = sum(H * M) / (sum(H) + eps)
```

- Ukuran ini menunjukkan **proporsi heatmap dalam mask**, bukan bukti bahwa model mengambil keputusan karena lokasi tersebut.
- Tambahkan tests dengan memburamkan/menghapus foreground region dan background region. Ukur perubahan logit target, success/faithfulness.
- Hindari perbandingan absolut intensitas merah antarpanel saat tiap CAM dinormalisasi independen.

### G3 — Pemilihan sampel yang defensible

- Jangan hanya mengambil enam contoh dari seluruh `dataset/raw/annotations.json` lalu menyebutnya test set.
- Pilih stratified test samples dan hard-negative cases dari split final yang independen; simpan `sample_id`, `model_hash`, `threshold`, `layer`, dan `target_score` di figure metadata.
- **DoD:** setiap figure Grad-CAM pada paper dapat direproduksi dengan perintah dan checkpoint yang tersimpan.

---

## 7. Rencana ablation matrix dan eksperimen

| ID | Perubahan relatif baseline | Tujuan isolasi | Data yang wajib sama |
|---|---|---|---|
| E0 | PA-SCAN v1 setelah P0 | Reproduced reference | Fixed manifests |
| E1 | E0 + full-frame/adaptive ROI | Dampak cropping | Same model training budget |
| E2 | E0 dengan CoordConv OFF, backbone/neck tetap | Kontribusi koordinat | Same architecture kecuali 2 input coord channels |
| E3 | E0 dengan attention OFF | Kontribusi attention | Same backbone/fusion |
| E4 | E0 dengan FiLM OFF | Kontribusi conditioning | Same pose supervision, atau nyatakan perbedaannya |
| E5 | E0 + elevation auxiliary conditioning | Robustness elevasi | Same splits and compute budget |
| E6 | E0 + component-specific spatial attention | Per-component localization | Same splits, masks bila dipakai sama |
| E7 | E0 + visibility head | Reliability pada occlusion | Same label/visibility availability |
| E8 | E6 + mask supervision | Pengaruh ground-truth geometry | Mask training only |
| E9 | Kombinasi modul terpilih berdasarkan validation | PA-SCAN v2 final | No test-informed selection |
| E10 | ResNet18 / MobileNetV3 baseline | Komparasi model yang dikenal | Scratch vs pretrained dicatat terpisah |

**Saran eksperimen factorial kecil:** bila E5, E6, dan E7 masing-masing menjanjikan, uji kombinasi 2^3 (delapan konfigurasi) untuk mengecek interaksi modul; jika sumber daya terbatas, gunakan ablation berurutan dan jelaskan keterbatasannya.

**Template tabel hasil akhir (isi dari eksperimen nyata; jangan isi angka prediksi):**

| Model | T0 Exact-match ↑ | T1 worst-view F1 ↑ | T2 top F1 ↑ | T3 unseen-mesh F1 ↑ | ECE ↓ | Latency ms ↓ | Params M ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| PA-SCAN v1 (re-run) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PA-SCAN + pose | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PA-SCAN + part | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PA-SCAN + visibility | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PA-SCAN v2 final | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet-18 baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 8. Pemetaan revisi per file

| File sekarang | Action | Prioritas | Output yang diharapkan |
|---|---|---|---|
| `data_collection/collect_data.py` | Canonical camera metadata; verify actual 3D state; optional mask/depth/object IDs, mesh and render-seed metadata | P0/P2 | Clean, versioned raw dataset |
| `data_collection/augment_data.py` | Pin Albumentations-compatible API; fail/record errors; parent IDs; label-safe transformations | P1 | Verified augmented dataset |
| `model/dataset.py` | Fix `top`, remove heuristic silent label mutation, explicit train/val/test manifests, parent group checks | **P0** | Trusted Dataset class |
| `model/architecture.py` | Separate ablation flags; expose feature maps; pose/elevation and per-component branches | P1/P2 | PA-SCAN v2 model, fair ablations |
| `model/train.py` | Config-driven training; loss schedule; deterministic options; checkpoint metadata; trained model fingerprint | P0/P2 | Reproducible trained checkpoints |
| `model/evaluate.py` | Independent test path; threshold calibration only on val; per-view and calibration metrics | **P0** | Auditable results CSV/JSON |
| `model/gradcam.py` | Correct target logit, stage4/fusion, legends, quantitative localization | P3 | Explainability figures |
| `model/change_detector.py` | Separate framewise scores vs temporal smoothing evaluation | P4 | Optional real-time appendix |
| `backend/app.py` | Match preprocessing/threshold config; disallow random-weight service for evaluation; session isolation | P0/P4 | Reliable inference protocol |
| `frontend/index.html` | Correct confidence vs `P(OPEN)` label, show threshold/uncertainty | P4 | Transparent dashboard |
| `extension/background.js` | Fix retry after STOP, duplicate timer, target tab selection | P4 | Reliable browser capture |
| `vlm/finetune.py` | Keep isolated from Paper A; validate collator and independent VLM metrics if later used | Paper B | Optional language task |
| `visual_grounding/inference.py` | Keep isolated from Paper A; evaluate part localization separately if used as baseline | Paper B | Optional grounding task |
| `README.md` | Baseline vs past versions, real split definition, exact environment, current metrics only | P0 | Reproducible project README |
| `docs/PRESENTATION_SLIDE_DECK.md` | Mark legacy metrics/config or refresh against experiment registry | P0 | No contradictory reported results |
| `backend/requirements.txt` | Environment lock/pin tested versions; ensure GPU variants documented | P0 | Dependency manifest |
| `tests/` | Audit existing tests; add data, split, model, threshold, and smoke tests | P0/P3 | Automated checks |
| `Dockerfile`, `docker-compose.yml` | Reproduce evaluation environment; record GPU and artifact mount paths | P3/P4 | Runnable benchmark container |

### Struktur file baru yang disarankan

```text
car-detection/
├── configs/
│   ├── dataset_v2.yaml
│   ├── splits/
│   │   ├── t0_within_grid.yaml
│   │   ├── t1_unseen_azimuth.yaml
│   │   └── t2_unseen_elevation.yaml
│   └── experiments/
│       ├── e0_baseline.yaml
│       ├── e5_pose.yaml
│       ├── e6_component_attention.yaml
│       ├── e7_visibility.yaml
│       └── e9_full.yaml
├── data_collection/
│   ├── audit_dataset.py
│   └── export_part_masks.py            # jika renderer mendukung
├── model/
│   ├── preprocess.py
│   ├── splits.py
│   ├── decision_policy.py
│   ├── baselines.py
│   ├── component_attention.py
│   ├── visibility_head.py
│   ├── losses.py
│   └── metrics.py
├── experiments/
│   ├── run_experiment.py
│   ├── run_ablation.py
│   ├── evaluate_ood.py
│   └── compare_runs.py
├── tests/
│   ├── test_elevation_metadata.py
│   ├── test_labels.py
│   ├── test_split_leakage.py
│   ├── test_roi_retention.py
│   ├── test_model_shapes.py
│   ├── test_ablation_independence.py
│   ├── test_threshold_policy.py
│   └── test_inference_consistency.py
└── results/
    ├── experiment_registry.csv
    ├── manifests/
    ├── metrics/
    ├── figures/
    └── model_cards/
```

File/file struktur di atas merupakan **usulan**, bukan klaim bahwa file tersebut sudah ada di repository.

---

## 9. Test plan minimal: unit, integration, dan scientific controls

### Data tests

- [ ] `test_all_32_state_combinations` — setiap kombinasi label tepat lima bit.
- [ ] `test_elevation_normalization` — `top`, alias legacy, dan unknown case ditangani benar.
- [ ] `test_no_silent_label_repair` — record baru tidak dimutasi oleh loader.
- [ ] `test_augmented_parent_in_same_split` — semua turunan berada di train bersama originalnya.
- [ ] `test_split_disjointness` — test source IDs tidak pernah muncul di train/val.
- [ ] `test_global_angle_holdout` — sudut T1 absent dari training.
- [ ] `test_roi_retains_target` — retention komponen dihitung dari mask dan dicatat.

### Model/inference tests

- [ ] Shape output `(batch, 5)` untuk seluruh varian.
- [ ] Loss dan gradient finite pada satu batch; mask loss hanya mengonsumsi label yang valid.
- [ ] Flag `use_coordconv=False` **tidak** mengubah backbone/neck tanpa flag eksplisit.
- [ ] `P(OPEN)`, prediksi kelas, dan confidence kelas terprediksi konsisten.
- [ ] Border Grad-CAM menunjukkan correct/incorrect, bukan label OPEN/CLOSED.
- [ ] Fixed seed + fixed config menghasilkan metrik dalam toleransi repeatability yang disepakati.
- [ ] Backend dan offline preprocessing memberi tensor/hasil identik pada input yang sama.

### Experimental controls

- [ ] Log Git commit SHA / branch / dirty working-tree state.
- [ ] Log dataset version, SHA manifest, sample counts, dan train/val/test IDs.
- [ ] Log random seeds, GPU, framework versions, batch, input resolution, optimizer, scheduler, epochs.
- [ ] Log checkpoint hash dan threshold config hash.
- [ ] Freeze evaluation protocol sebelum final test.
- [ ] Simpan prediction-level file (`sample_id`, `y_true`, `prob_open`, `y_pred`, `view`, `visibility`) untuk analisis berpasangan.

---

## 10. Milestone/gates (tanpa target waktu yang dibuat-buat)

### M0 — Dataset & evaluation repair

**Input:** kode repo, dataset lokal, dan metadata renderer.  
**Deliverables:** `audit_dataset` report, `dataset_v2` manifest, corrected split, E0 runner.  
**Exit criteria:** P0 selesai, tidak ada silent drop/correction, independent test aman.

### M1 — Baseline reproducibility

**Deliverables:** PA-SCAN v1 re-run, standard CNN/ResNet baselines, isolated architecture flags.  
**Exit criteria:** skor dan variance antarseed tercatat; ketidaksesuaian angka historis dapat dijelaskan.

### M2 — Pose robustness

**Deliverables:** eksperimen B0–B4 untuk pose/elevation, tabel per-view.  
**Exit criteria:** keputusan mempertahankan/menolak modul pose enhancement berdasar validation dan T1/T2 yang sudah terkunci.

### M3 — Component-aware learning

**Deliverables:** spatial attention per komponen, mask extraction bila mungkin, localization scores.  
**Exit criteria:** ablation menyatakan apakah per-komponen membantu di luar perubahan parameter/FLOPs.

### M4 — Visibility robustness

**Deliverables:** visibility annotations, visibility head, selective prediction evaluation.  
**Exit criteria:** terdapat grafik performance vs occlusion/coverage yang dapat diaudit.

### M5 — Final Paper A

**Deliverables:** final trained model, frozen test evaluation, multi-seed table, failure analysis, Grad-CAM quantitative, reproducibility package, paper draft.  
**Exit criteria:** semua klaim di abstrak/kesimpulan dirujuk ke tabel/figure dan code/data version spesifik.

---

## 11. Research questions dan klaim yang *boleh* diuji

### Research questions

- **RQ1:** Bagaimana distribusi azimuth, elevasi, visibilitas, dan ROI memengaruhi kesalahan klasifikasi multi-label status komponen kendaraan?
- **RQ2:** Apakah elevation-aware pose conditioning meningkatkan generalisasi pada unseen viewpoints dibandingkan azimuth-only PA-SCAN?
- **RQ3:** Apakah component-specific spatial attention mengungguli global shared features dalam akurasi dan lokalisasi fitur?
- **RQ4:** Apakah supervision visibilitas dan/atau mask komponen memperbaiki robustness, calibration, dan selective prediction pada part yang tertutup?
- **RQ5:** Apa tradeoff improvement terhadap parameter, FLOPs, dan latensi?

### Proposed contributions (*hanya bila terverifikasi*)

1. Protokol evaluasi multi-view yang leak-free dan mendefinisikan variasi pose/domain secara terpisah.
2. Komponen model pose/component/visibility-aware yang diuji melalui ablation terisolasi.
3. Analisis atribut perhatian dan reliabilitas prediksi berbasis visibility mask, bukan hanya panel Grad-CAM visual.
4. Reproducible benchmark dan failure taxonomy pada synthetic 3D vehicle states.

### Klaim yang harus dihindari tanpa bukti

- "PA-SCAN adalah state-of-the-art" tanpa pembanding sistematis.
- "Model robust pada unseen vehicles" jika hanya satu model/mesh 3D.
- "Model memiliki attention yang tepat" hanya karena Grad-CAM terlihat merah pada pintu.
- "96,56% test accuracy" ketika nilai didapat dari validation + threshold tuning.
- "Dataset terdiri dari 768 original + 5.376 augmentation aktif selama training" sebelum audit menghitung yang benar-benar diload.
- "Penambahan AI agent meningkatkan vision accuracy" tanpa desain Paper B terpisah.

---

## 12. Scope Paper A vs Paper B

| Masukkan Paper A | Tunda untuk Paper B |
|---|---|
| CNN/state classification, PA-SCAN v2 | Planning/orchestration agent |
| Pose & visibility conditioning | Agent memanggil model vision/tools secara adaptif |
| Per-component attention/mask supervision | Dynamic algorithm selection (OpenCV, model, view request) |
| Robustness datasets dan benchmark | Tool cost, agent failure recovery, stepwise validation |
| Grad-CAM/faithfulness, calibration | LLM reasoning dan tool orchestration evaluation |
| Inference latency model | End-to-end multi-agent system evaluation |

**Alasan:** jika kedua paper memakai dataset yang sama, pisahkan *pertanyaan penelitian*, *kontribusi*, dan *evaluasi*. Hindari membuat Paper B hanya sebagai pengemasan ulang model dan hasil Paper A.

---

## 13. Literatur dan posisi terhadap prior work

| Topik | Referensi yang diperiksa | Bagaimana digunakan |
|---|---|---|
| Synthetic articulated vehicles | Kim et al., *Learning From Synthetic Vehicles*, WACV Workshops 2022 [L1] | Menunjukkan synthetic vehicle parts bukan novelty baru sendirinya |
| Vehicle parsing arbitrary views | Miao et al., *Robust 2D/3D Vehicle Parsing in Arbitrary Camera Views for CVIS*, ICCV 2021 [L2] | Justifikasi generalisasi lintas viewpoint |
| Component segmentation synthetic vehicles | Liu et al., *Learning Part Segmentation Through Unsupervised Domain Adaptation From Synthetic Vehicles*, CVPR 2022 [L3] | Inspirasi mask supervision dan part localization |
| Channel/spatial attention | Woo et al., *CBAM*, ECCV 2018 [L4] | Landasan modul attention; implementasi repo bukan harus CBAM identik |
| CoordConv | Liu et al., *An Intriguing Failing of CNNs and the CoordConv Solution*, NeurIPS 2018 [L5] | Landasan koordinat spasial; ablation harus terisolasi |
| FiLM | Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018 [L6] | Landasan pose-conditioned feature modulation |
| Amodal occlusion | Tai et al., *Segment Anything, Even Occluded*, CVPR 2025 [L7] | Inspirasi beda visible mask dan amodal mask |
| Grad-CAM | Selvaraju et al., ICCV 2017 [L8] | Dasar explainability dan target-score analysis |
| Saliency sanity checks | Adebayo et al., NeurIPS 2018 [L9] | Wajib menghindari interpretasi saliency yang terlalu kuat |
| Car-parts detection (terkini) | *Optimized car parts detection with advanced feature fusion and attention modules*, Scientific Reports 2025 [L10] | Pembanding related work, tetapi task part detection ≠ state classification |

### Audit novelty yang masih diperlukan

- Cari systematic keywords: `articulated vehicle state recognition`, `car door open closed classification`, `pose-conditioned multi-label classification`, `visibility-aware part recognition`, `component-specific spatial attention`, `synthetic-to-real articulated parts`.
- Bandingkan data input, supervised labels, tasks, metrics, compute, cross-domain, serta kelemahan tiap paper.
- Buat tabel *related work* yang membedakan part detection, part segmentation, state classification, dan multi-view recognition.
- **Tidak mengklaim nama PA-SCAN sudah/ belum resmi dipublikasikan** tanpa pencarian database bibliografis yang lengkap.

---

## 14. Tautan sumber kode yang diperiksa

- **[S1]** `README.md` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/README.md
- **[S2]** `data_collection/collect_data.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/data_collection/collect_data.py
- **[S3]** `model/dataset.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/model/dataset.py
- **[S4]** `model/architecture.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/model/architecture.py
- **[S5]** `model/evaluate.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/model/evaluate.py
- **[S6]** `model/train.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/model/train.py
- **[S7]** `backend/app.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/backend/app.py
- **[S8]** `docs/PRESENTATION_SLIDE_DECK.md` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/docs/PRESENTATION_SLIDE_DECK.md
- **[S9]** `model/gradcam.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/model/gradcam.py
- **[S10]** `.gitignore` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/.gitignore
- **[S11]** `backend/requirements.txt` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/backend/requirements.txt
- **[S12]** `model/train.py` (seed, `--max-aug`) — sama seperti [S6]
- **[S13]** `data_collection/augment_data.py` — https://raw.githubusercontent.com/afidzulinn/car-detection/main/data_collection/augment_data.py
- **Tambahan sistem:** `model/change_detector.py`, `extension/background.js`, `frontend/index.html`, `vlm/finetune.py`, `visual_grounding/inference.py`, `backend/vlm_module.py`, `docker-compose.yml` melalui pola URL yang sama pada repo `main`.

## 15. Referensi literatur ilmiah

- **[L1]** Kim et al. (2022), *Learning From Synthetic Vehicles*, WACV Workshops — https://openaccess.thecvf.com/WACV2022_workshops/RWS
- **[L2]** Miao et al. (2021), *Robust 2D/3D Vehicle Parsing in Arbitrary Camera Views for CVIS*, ICCV — https://openaccess.thecvf.com/ICCV2021?day=2021-10-13
- **[L3]** Liu et al. (2022), *Learning Part Segmentation Through Unsupervised Domain Adaptation From Synthetic Vehicles*, CVPR — https://openaccess.thecvf.com/CVPR2022?day=2022-06-24
- **[L4]** Woo et al. (2018), *CBAM: Convolutional Block Attention Module*, ECCV — https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html
- **[L5]** Liu et al. (2018), *An Intriguing Failing of CNNs and the CoordConv Solution*, NeurIPS — https://papers.nips.cc/paper_files/paper/2018/hash/60106888f8977b71e1f15db7bc9a88d1-Abstract.html
- **[L6]** Perez et al. (2018), *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI — https://ojs.aaai.org/index.php/AAAI/article/view/11671
- **[L7]** Tai et al. (2025), *Segment Anything, Even Occluded*, CVPR — https://openaccess.thecvf.com/content/CVPR2025/html/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.html
- **[L8]** Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV — https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- **[L9]** Adebayo et al. (2018), *Sanity Checks for Saliency Maps*, NeurIPS — https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- **[L10]** *Optimized car parts detection with advanced feature fusion and attention modules* (2025), Scientific Reports — https://www.nature.com/articles/s41598-025-29855-w

---

## 16. Batas audit, asumsi, dan catatan kejujuran penelitian

1. **Ini adalah static review kode publik dan penelusuran referensi**, bukan pelaksanaan training, pengujian unit, evaluasi checkpoint, ataupun inspeksi lengkap binary dataset.
2. Dataset asli, bobot model, serta log run tidak tersedia pada URL repository publik karena sejumlah path diabaikan `.gitignore`. Karena itu efek filter `top` pada checkpoint tertentu harus dikonfirmasi pada dataset/manifest yang benar-benar dipakai.
3. Struktur `tests/` dan sebagian berkas notebook tidak diaudit sampai seluruh isinya; pekerjaan penambahan tests adalah rekomendasi, bukan pernyataan bahwa tests saat ini tidak ada.
4. Target peningkatan dan robustness **tidak diberi angka fiktif**. Seluruh metrik PA-SCAN v2 ditulis `TBD` sampai ada eksperimen.
5. Bila web interface 3D sumber tidak mengekspos part IDs, mask/depth/amodal annotation adalah *new data engineering work*, bukan kemampuan yang telah ada dalam repo.
6. Protokol `unseen mesh` hanya valid bila menambah kendaraan berbeda dan split tidak bocor pada mesh target. Ini krusial bila paper ingin membahas generalisasi di luar satu model mobil sintetis.
7. Semua source link menunjuk `main` saat audit. Sebelum menjalankan eksperimen akhir, **pin commit SHA** agar hasil tidak berubah akibat update repository.

---

## 17. Ringkasan keputusan untuk dieksekusi

**Urutan awal yang disetujui untuk perencanaan:**

1. Perbaiki `model/dataset.py` elevasi/label/split dan buat dataset audit.
2. Buat evaluasi independent test + threshold calibration; rerun PA-SCAN v1 sebagai E0.
3. Audit `crop_car_roi()` dengan per-elevation/part visibility retention.
4. Refactor `model/architecture.py` agar ablation benar-benar independen dan jalankan model pembanding.
5. Uji elevation-aware pose conditioning dan component-specific attention **secara terpisah**.
6. Tambahkan visibility-aware branch/mask supervision **hanya setelah annotation valid tersedia**.
7. Bekukan arsitektur final lewat validation, lalu lakukan independent held-out-view/domain tests dan quantitative Grad-CAM.
8. Tulis Paper A dengan klaim terbatas pada apa yang benar-benar ditunjukkan tabel, grafik, dan run yang direproduksi.

**Definisi sukses Paper A:** bukan hanya angka mean accuracy lebih tinggi, melainkan **exact-match, macro-F1, dan worst-view performance** yang lebih baik atau tradeoff yang secara kuantitatif bermanfaat pada **test independen**, beserta bukti ablation yang mengidentifikasi sumber perbaikannya.
