# LungSphere Program Final

Folder ini adalah versi program yang dirapikan untuk laporan Tugas Akhir.
Versi ini mengikuti dokumen PROPTA: fitur utama adalah gabungan DWT-MFCC,
model yang dibandingkan adalah SVM-RBF dan Random Forest, lalu aplikasi web
memakai model dengan performa patient-level terbaik.

## Struktur

- `app.py` - aplikasi web Flask untuk upload audio dan inferensi.
- `config.py` - konfigurasi path, sample rate, threshold, dan parameter fitur.
- `features.py` - fungsi load audio, segmentasi 4 detik, dan ekstraksi DWT-MFCC.
- `train_models.py` - training ulang SVM-RBF dan Random Forest dari CSV fitur.
- `data/fraiwan_dwt_segments_best.csv` - data fitur penelitian.
- `models/` - lokasi model kandidat dan model terbaik setelah training.
- `reports/` - lokasi laporan metrik evaluasi JSON.
- `templates/index.html` - tampilan web LungSphere.

## Setup

```bash
cd program_final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training model final

```bash
python train_models.py
```

Output:

- `models/svm_dwt_mfcc_lungsphere.joblib`
- `models/rf_dwt_mfcc_lungsphere.joblib`
- `models/best_lungsphere_model.joblib`
- `models/best_lungsphere_model_metadata.json`
- `reports/model_evaluation.json`

## Menjalankan aplikasi web

```bash
python app.py
```

Buka:

```text
http://127.0.0.1:5001
```

Format audio yang didukung: `.wav`, `.mp3`, dan `.flac`.

## Deploy ke Vercel

Repository ini sudah menyiapkan `vercel.json`, sehingga bisa dideploy sebagai
Flask application di Vercel.

Konfigurasi yang digunakan:

- Install Command: `pip install -r requirements.txt`
- Entry point: `app.py`
- Health Check Path: `/health`

Saat berjalan di Vercel, file upload sementara disimpan di `/tmp/lungsphere_uploads`.

## Deploy ke Render

Repository ini sudah menyiapkan `render.yaml` dan `Procfile`, sehingga bisa
dideploy sebagai Python web service.

Konfigurasi manual jika diminta:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`
- Health Check Path: `/health`

## Catatan akademik

Sistem ini adalah prototipe skrining awal dan bukan alat diagnosis medis final.
Hasil berisiko tinggi tetap harus ditindaklanjuti oleh tenaga kesehatan.
