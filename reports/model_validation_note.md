# LungSphere Model Validation Note

Tanggal validasi: 2026-06-24

## Dataset dan fitur

- Dataset fitur: `data/fraiwan_dwt_segments_best.csv`
- Jumlah pasien: 112
- Jumlah segmen: 429
- Distribusi pasien: 35 sehat dan 77 tidak sehat
- Fitur: DWT-MFCC, 70 fitur per segmen
- Komposisi fitur: 30 fitur statistik DWT dan 40 fitur MFCC mean/std

## Skema evaluasi

- Split: patient-wise hold-out
- Data latih: 89 pasien, 331 segmen
- Data uji: 23 pasien, 98 segmen
- Random state: 42
- Threshold prediksi segmen: 0.5
- Agregasi pasien: majority voting dari prediksi segmen

## Ringkasan hasil training ulang

| Model | Segment Acc | Patient Acc | Sens. | Spec. | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM-RBF DWT-MFCC | 74.49% | 78.26% | 93.75% | 42.86% | 85.71% | 0.7679 |
| Random Forest DWT-MFCC | 68.37% | 60.87% | 62.50% | 57.14% | 68.97% | 0.7589 |

Model terpilih untuk aplikasi web: **SVM-RBF DWT-MFCC**.

## Validasi teknis

- Syntax check Python berhasil untuk seluruh file `.py` di `program_final`.
- Ekstraksi fitur aplikasi dibandingkan dengan CSV training pada sampel pertama menghasilkan selisih maksimum `0.000102`, sehingga pipeline fitur aplikasi sudah konsisten dengan dataset fitur.
- Smoke test inferensi pada `BP85_N,N,A R U,33,M.wav` menghasilkan label `Healthy`, risk score `30.38%`, dan 3 segmen sehat.
- Server Flask berhasil berjalan pada `http://127.0.0.1:5001`.

## Catatan akademik

Artefak model lama `program/rf_dwt_fraiwan_best.joblib` dapat menghasilkan akurasi patient-level 82.61% pada split uji yang sama, tetapi proses training lengkapnya tidak dapat direproduksi persis pada environment saat ini. Untuk laporan TA, hasil yang paling aman digunakan adalah hasil training ulang dari `program_final/train_models.py`, karena pipeline, data, model, dan metriknya terdokumentasi penuh.
