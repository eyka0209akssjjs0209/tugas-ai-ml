# Klasifikasi Ponsel Premium dengan Logistic Regression

Mini project mata kuliah *Machine Learning dan Dasar Deep Learning*. Model memprediksi apakah sebuah ponsel termasuk **premium** (harga di atas ₹25.000) berdasarkan spesifikasinya, menggunakan **Logistic Regression**.

---

## 1. Ringkasan

| Item | Keterangan |
|---|---|
| Tugas | Klasifikasi biner (premium vs non-premium) |
| Model | Logistic Regression (`class_weight="balanced"`) |
| Dataset | `ndtv_data_final.csv`, 1.359 ponsel, 22 kolom, harga dalam rupee India (INR) |
| Target | `premium = 1` jika `Price > 25000`, selain itu `0` |
| Distribusi kelas | Non-premium 1.232 (90,7%) dan premium 127 (9,3%) |
| Evaluasi | Precision, recall, F1-score, ROC-AUC, confusion matrix, 5-fold cross-validation |

**Catatan penyesuaian tugas.** Dataset tidak memiliki kolom waktu atau riwayat harga, sehingga prediksi "harga naik/turun" tidak dapat dilakukan secara jujur. Target diubah menjadi **segmen harga**, yang tetap berupa klasifikasi biner dan memenuhi seluruh tahapan tugas (analisis data, pemodelan, evaluasi).

---

## 2. Isi Folder

| File | Fungsi |
|---|---|
| `logreg_ponsel_premium.py` | Skrip utama: cleaning, EDA, training, evaluasi, dan grafik |
| `ndtv_clean.csv` | Data hasil pembersihan dan feature engineering |
| `hasil_logreg.png` | Grafik korelasi, koefisien model, dan confusion matrix |
| `README.md` | Dokumen ini |

---

## 3. Cara Menjalankan

Kebutuhan: Python 3.9+, `pandas`, `numpy`, `scikit-learn` (versi 1.1 ke atas, diuji pada 1.8.0), dan `matplotlib`.

```bash
pip install pandas numpy scikit-learn matplotlib
python logreg_ponsel_premium.py
```

Ubah path pada bagian atas skrip sesuai lokasi file Anda:

```python
SRC     = "ndtv_data_final.csv"   # lokasi dataset asli
OUT_CSV = "ndtv_clean.csv"
OUT_FIG = "hasil_logreg.png"
```

Dua pengaturan penting:

- `THRESHOLD_PRICE = 25_000` menentukan batas segmen premium.
- `USE_DECOYS = True` mempertahankan dua fitur pengecoh untuk bahan penjelasan. Ubah ke `False` untuk melatih model tanpa keduanya.

---

## 4. Alur Proses Training

```
Load data
   -> Cleaning
      -> Feature engineering & target
         -> EDA & korelasi
            -> Train-test split (stratified)
               -> Pipeline: log + scaling + one-hot
                  -> Cross-validation (skenario A-D)
                     -> Training model final
                        -> Evaluasi data uji
                           -> Tuning ambang probabilitas
                              -> Analisis koefisien & fitur pengecoh
```

### Tahap 1: Load data
Dataset dibaca dan diperiksa: 1.359 baris, tanpa missing value, dan tanpa duplikat persis.

### Tahap 2: Cleaning
Dataset sudah rapi secara format, tetapi tetap memerlukan pembersihan agar cocok untuk Logistic Regression.

| Temuan | Tindakan |
|---|---|
| `Unnamed: 0`, `Name`, `Model` adalah identitas (hampir unik per baris) | Dibuang dari fitur |
| `Price` adalah sumber target | Hanya dipakai membuat target, tidak menjadi fitur |
| 39 baris memiliki spesifikasi identik dengan baris lain | Dipertahankan, dicatat sebagai risiko hasil terlalu optimis |
| 12 ponsel storage di bawah 1 GB, 10 ponsel RAM ≤ 256 MB, 18 ponsel tanpa kamera depan | Kemungkinan *feature phone*, nilainya sah, tidak diimputasi |
| 17 baris dengan `Resolution x > y` | Diatasi dengan memakai PPI (berbasis diagonal) |
| 245 ponsel "4G=No tetapi 3G=Yes" dan 43 ponsel "4G=Yes tetapi 3G=No" | Dicatat sebagai inkonsistensi data |
| Kolom Yes/No | Diubah menjadi 1/0 |
| Outlier harga tinggi (hingga ₹174.990) | **Tidak dibuang**, karena itulah kelas yang ingin ditangkap |

### Tahap 3: Feature engineering dan target

| Fitur baru | Cara pembuatan | Alasan |
|---|---|---|
| `ram_gb` | `RAM (MB) / 1024` | Satuan lebih mudah dibaca |
| `ppi` | `sqrt(res_x² + res_y²) / screen_size` | Menggabungkan resolusi dan ukuran layar, menghindari fitur kembar |
| `log_storage` | `log2(storage_gb)` | Data miring tajam (skew 4,07) |
| `log_rear` | `log(1 + rear_camera)` | Data miring (skew 3,66) |
| `premium` | `Price > 25000` | Target klasifikasi |

### Tahap 4: EDA dan korelasi

Korelasi fitur numerik dengan target premium:

| Fitur | Korelasi | AUC tunggal | Jenis |
|---|---|---|---|
| `ppi` | 0,457 | 0,876 | Inti |
| `ram_gb` | 0,457 | 0,844 | Inti |
| `log_storage` | 0,380 | 0,848 | Inti |
| `log_rear` | 0,306 | 0,741 | Inti |
| `screen_in` | 0,277 | 0,775 | Inti |
| `front_cam` | 0,206 | 0,704 | Inti |
| `battery` | 0,199 | 0,712 | Inti |
| `has_4g` | 0,124 | 0,593 | **Pengecoh** |
| `n_sims` | -0,168 | 0,392 | **Pengecoh** |

Temuan lain:

- `ppi` berkorelasi 0,94-0,95 dengan Resolution x/y, dan `ram_gb` berkorelasi 0,86 dengan `log_storage` (multikolinearitas).
- Proporsi premium pada iOS mencapai 70,6% (12 dari 17), jauh di atas Android (8,5%).

### Tahap 5: Train-test split
Data dibagi 80% training (1.087 baris) dan 20% testing (272 baris) dengan `stratify=y` dan `random_state=42`. Proporsi premium terjaga: 9,4% pada training dan 9,2% pada testing (25 ponsel premium).

### Tahap 6: Pipeline preprocessing dan model
Semua transformasi berada di dalam `Pipeline`, sehingga *scaler* dan *encoder* hanya belajar dari data training (mencegah data leakage).

- Fitur numerik: `StandardScaler` (wajib untuk Logistic Regression).
- Fitur kategorikal (`os`, `brand`): `OneHotEncoder` dengan `min_frequency=10`, sehingga kategori langka digabung otomatis.
- Model: `LogisticRegression(class_weight="balanced", max_iter=3000)` untuk menangani imbalance 9,3%.

### Tahap 7: Cross-validation (5-fold, data training)
Empat skenario fitur dibandingkan:

| Skenario | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| A. Spesifikasi inti | 0,313 | 0,823 | 0,454 | 0,909 |
| B. A + Operating System | 0,351 | 0,862 | 0,499 | 0,926 |
| C. B + Brand | 0,443 | 0,883 | 0,589 | 0,952 |
| D. C + fitur pengecoh (SIM, 4G) | 0,449 | 0,843 | 0,586 | 0,953 |

Operating System dan Brand memberi peningkatan nyata, sedangkan fitur pengecoh tidak menambah nilai (F1 D ≈ C).

### Tahap 8: Evaluasi pada data uji (ambang 0,5)

Confusion matrix:

|  | Prediksi non-premium | Prediksi premium |
|---|---|---|
| **Aktual non-premium** | 216 (TN) | 31 (FP) |
| **Aktual premium** | 4 (FN) | 21 (TP) |

| Metrik | Nilai |
|---|---|
| Precision (premium) | 0,404 |
| Recall (premium) | 0,840 |
| F1 (premium) | 0,545 |
| ROC-AUC | 0,934 |
| Accuracy | 0,871 |
| Accuracy jika selalu menebak "non-premium" | **0,907** |

### Tahap 9: Tuning ambang probabilitas
Ambang dipilih dari prediksi *out-of-fold* data training (bukan dari data uji) agar tidak bocor. Ambang dengan F1 terbaik adalah **0,70**. Pada data uji, hasilnya:

| Ambang | Precision | Recall | F1 |
|---|---|---|---|
| 0,50 | 0,404 | 0,840 | 0,545 |
| 0,70 | 0,415 | 0,680 | 0,515 |

Tuning ambang **tidak memperbaiki** hasil pada data uji. Dengan hanya 25 sampel premium, selisih ini mudah berubah, sehingga tidak ada klaim bahwa satu ambang pasti lebih baik.

### Tahap 10: Analisis koefisien dan fitur pengecoh

| Fitur | Korelasi dengan target | Koefisien LR | Tanda sama? |
|---|---|---|---|
| `ram_gb` | 0,457 | 0,987 | Ya |
| `ppi` | 0,457 | 0,913 | Ya |
| `log_storage` | 0,380 | 0,607 | Ya |
| `battery` | 0,199 | 0,422 | Ya |
| `has_4g` (pengecoh) | 0,124 | 0,353 | Ya |
| `log_rear` | 0,306 | 0,174 | Ya |
| `screen_in` | 0,277 | 0,096 | Ya |
| `n_sims` (pengecoh) | -0,168 | -0,407 | Ya |
| `front_cam` | 0,206 | -0,442 | **Tidak** |

Koefisien kategori terbesar: HTC (3,86), Sony (3,31), Huawei (1,50), iOS (1,41), Apple (1,41).

---

## 5. Fitur Pengecoh

Dua fitur sengaja dipertahankan untuk menunjukkan bahwa **korelasi atau koefisien yang tampak masuk akal belum tentu bermakna**.

**`n_sims` (Number of SIMs).** Ponsel 1 SIM premium 20,3%, sedangkan 2 SIM hanya 7,2%. Ini bukan sebab-akibat: banyak flagship di data ini (Samsung, Sony, Google, Apple) memang 1 SIM. Fitur ini hanya penanda gaya produk. Menambahkannya tidak meningkatkan F1 maupun ROC-AUC.

**`has_4g` (4G/LTE).** Premium 11,5% pada ponsel 4G vs 3,2% tanpa 4G, tetapi hubungan ini lebih mencerminkan tahun rilis ponsel. Datanya juga tidak konsisten dengan kolom 3G.

**Fitur lain yang perlu diwaspadai:**

- `front_cam` berkorelasi +0,21 tetapi berkoefisien -0,44. Efeknya sudah diwakili fitur lain, sehingga tidak boleh ditafsirkan sendiri-sendiri.
- Wi-Fi, Bluetooth, Touchscreen, dan 3G hampir konstan (AUC sekitar 0,50), sehingga tidak dimasukkan ke model.

---

## 6. Keterbatasan

- **Tidak ada dimensi waktu.** Model memprediksi segmen harga, bukan arah pergerakan harga.
- **Data uji kecil untuk kelas premium** (25 sampel). Satu atau dua prediksi saja dapat mengubah precision dan recall cukup besar, sehingga hasil cross-validation lebih layak dipercaya.
- **Model condong ke identitas brand.** Koefisien HTC dan Sony lebih besar daripada iOS, yang menunjukkan model sebagian belajar "brand ini mahal", bukan murni dari spesifikasi.
- **Near-duplicate belum ditangani.** 39 baris dengan spesifikasi identik dapat muncul di data training dan testing sekaligus, sehingga hasil bisa sedikit terlalu optimis.
- **Data berusia sekitar 2015-2019** dengan harga rupee. Ambang ₹25.000 adalah asumsi, bukan standar baku, dan hasilnya tidak otomatis berlaku untuk pasar atau tahun lain.
- **Dominasi brand lokal murah** (Intex, Micromax, Lava) dan sedikitnya data iOS (17 baris) dapat membiaskan model.

---

## 7. Kesimpulan

1. **Spesifikasi cukup untuk mengenali ponsel premium.** Resolusi layar (PPI), RAM, dan storage adalah penanda terkuat. Model mencapai ROC-AUC 0,934 pada data uji dan recall premium 0,84, artinya sebagian besar ponsel premium berhasil ditangkap.

2. **Accuracy menyesatkan pada data tidak seimbang.** Accuracy model (0,871) lebih rendah daripada tebakan "selalu non-premium" (0,907), padahal model jauh lebih berguna karena menemukan 21 dari 25 ponsel premium. Ini menjawab pertanyaan reflektif tugas: kenaikan akurasi tidak selalu berarti model lebih berguna.

3. **Ada trade-off precision dan recall.** Dengan `class_weight="balanced"`, recall tinggi dicapai dengan mengorbankan precision (0,40): sekitar 31 ponsel biasa salah dilabeli premium. Model ini cocok bila kesalahan *melewatkan* ponsel premium lebih merugikan daripada salah melabeli ponsel biasa. Bila sebaliknya, ambang perlu dinaikkan dan precision diprioritaskan.

4. **Operating System dan Brand meningkatkan kinerja**, dengan F1 naik dari 0,454 (spesifikasi saja) menjadi 0,589 pada cross-validation. Namun peningkatan ini datang dengan risiko ketergantungan pada identitas brand.

5. **Fitur pengecoh tidak membantu.** Menambahkan Number of SIMs dan 4G tidak memperbaiki F1 maupun ROC-AUC, meskipun tanda koefisiennya tampak masuk akal. Pemilihan fitur sebaiknya berdasarkan pemahaman domain dan pengujian, bukan hanya korelasi atau koefisien.

6. **Tuning ambang perlu hati-hati.** Ambang yang optimal pada cross-validation (0,70) tidak memberi perbaikan pada data uji yang kecil.

**Saran pengembangan:** tangani near-duplicate dengan *group split*, bandingkan dengan Random Forest atau Gradient Boosting, coba SMOTE di dalam pipeline, bandingkan model dengan dan tanpa Brand, dan gunakan dataset berderet waktu bila ingin memprediksi arah harga.

---

## 8. Referensi

- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
- Materi kuliah *Machine Learning dan Dasar Deep Learning*, Slide 19: Latihan Implementasi untuk Mahasiswa.
