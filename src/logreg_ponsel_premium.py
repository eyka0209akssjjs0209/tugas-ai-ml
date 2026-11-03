"""
Mini Project: Klasifikasi Ponsel Premium (Price > 25.000 INR) dengan Logistic Regression
Dataset : ndtv_data_final.csv (1.359 baris, harga dalam rupee India)

Alur    : Load -> Cleaning -> Feature engineering -> EDA & korelasi -> Split ->
          Pipeline (log + scaler + one-hot) -> Cross-validation (skenario A-D) ->
          Evaluasi data uji -> Tuning ambang -> Analisis fitur pengecoh

FITUR PENGECOH sengaja DIPERTAHANKAN (USE_DECOYS = True) untuk bahan penjelasan:
  1. Number of SIMs : korelasi negatif, tetapi bukan sebab-akibat (penanda gaya produk)
  2. 4G/ LTE        : lemah, lebih mencerminkan tahun rilis, dan datanya inkonsisten dengan 3G
Ubah USE_DECOYS ke False untuk membandingkan model tanpa fitur pengecoh.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

SRC = "/mnt/user-data/uploads/ndtv_data_final.csv"
OUT_CSV = "/mnt/user-data/outputs/ndtv_clean.csv"
OUT_FIG = "/mnt/user-data/outputs/hasil_logreg.png"
SEED = 42
THRESHOLD_PRICE = 25_000      # premium jika Price > 25.000 INR
USE_DECOYS = True             # pertahankan fitur pengecoh untuk penjelasan


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ============================================================ 1. LOAD
df = pd.read_csv(SRC)
header("1. LOAD DATA")
print("Bentuk data :", df.shape)
print("Missing value total :", int(df.isna().sum().sum()))
print("Duplikat baris persis :", int(df.duplicated().sum()))

# ============================================================ 2. CLEANING
header("2. CLEANING")
# 2a. Buang kolom identitas (bukan fitur). Price tetap dipegang hanya untuk membuat target.
df = df.drop(columns=["Unnamed: 0", "Name", "Model"])

# 2b. Cek near-duplicate: spesifikasi identik (tanpa Price) pada brand yang sama
spec_cols = [c for c in df.columns if c != "Price"]
n_near = int(df.duplicated(subset=spec_cols).sum())
print(f"Baris dengan spesifikasi identik dengan baris lain : {n_near}")
print("  -> dipertahankan, tetapi berpotensi membuat hasil terlalu optimis "
      "(near-duplicate di train & test)")

# 2c. Cek nilai ekstrem yang sah (feature phone), jangan dihapus/diimputasi
print("Storage < 1 GB :", int((df["Internal storage (GB)"] < 1).sum()),
      "| RAM <= 256 MB :", int((df["RAM (MB)"] <= 256).sum()),
      "| Kamera belakang = 0 :", int((df["Rear camera"] == 0).sum()),
      "| Kamera depan = 0 :", int((df["Front camera"] == 0).sum()))
print("Resolution x > y (orientasi terbalik) :",
      int((df["Resolution x"] > df["Resolution y"]).sum()), "-> diatasi dengan PPI")

# 2d. Cek konsistensi logis 3G vs 4G
print("4G=No tetapi 3G=Yes :",
      int(((df["4G/ LTE"] == "No") & (df["3G"] == "Yes")).sum()),
      "| 4G=Yes tetapi 3G=No :",
      int(((df["4G/ LTE"] == "Yes") & (df["3G"] == "No")).sum()),
      "-> inkonsisten, salah satu alasan 4G dianggap pengecoh")

# 2e. Encode Yes/No -> 1/0
for c in ["Touchscreen", "Wi-Fi", "Bluetooth", "GPS", "3G", "4G/ LTE"]:
    df[c] = (df[c] == "Yes").astype(int)

# ============================================================ 3. FEATURE ENGINEERING
header("3. FEATURE ENGINEERING")
df["ram_gb"] = df["RAM (MB)"] / 1024
df["ppi"] = (np.sqrt(df["Resolution x"] ** 2 + df["Resolution y"] ** 2)
             / df["Screen size (inches)"])
df["log_storage"] = np.log2(df["Internal storage (GB)"])   # skew 4,07 -> log
df["log_rear"] = np.log1p(df["Rear camera"])                # skew 3,66 -> log
df = df.rename(columns={"Battery capacity (mAh)": "battery",
                        "Screen size (inches)": "screen_in",
                        "Front camera": "front_cam",
                        "Number of SIMs": "n_sims",
                        "4G/ LTE": "has_4g",
                        "Operating system": "os",
                        "Brand": "brand"})

# Target
df["premium"] = (df["Price"] > THRESHOLD_PRICE).astype(int)
df.to_csv(OUT_CSV, index=False)
print("Fitur turunan: ram_gb, ppi, log_storage, log_rear | target: premium")

# ============================================================ 4. EDA
header("4. EDA: DISTRIBUSI KELAS & KORELASI")
vc = df["premium"].value_counts()
print(f"Non-premium: {vc[0]} ({vc[0]/len(df):.1%}) | Premium: {vc[1]} ({vc[1]/len(df):.1%})")
print(f"Accuracy jika selalu menebak 'non-premium': {vc[0]/len(df):.3f}  <- baseline bodoh")

CORE_NUM = ["ppi", "ram_gb", "log_storage", "log_rear", "front_cam", "battery", "screen_in"]
DECOY_NUM = ["n_sims", "has_4g"]
ALL_NUM = CORE_NUM + DECOY_NUM

eda = pd.DataFrame({
    "korelasi": df[ALL_NUM].corrwith(df["premium"]),
    "AUC_tunggal": [roc_auc_score(df["premium"], df[c]) for c in ALL_NUM],
}).round(3)
eda["jenis"] = ["inti"] * len(CORE_NUM) + ["PENGECOH"] * len(DECOY_NUM)
print("\nKorelasi tiap fitur dengan target:")
print(eda.sort_values("korelasi", ascending=False))

print("\nMultikolinearitas (|korelasi| antar fitur > 0,7):")
cm = df[ALL_NUM + ["Resolution x", "Resolution y"]].corr().abs()
cols = cm.columns.tolist()
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        if cm.loc[a, b] > 0.7:
            print(f"  {a:12s} <-> {b:12s}: {cm.loc[a, b]:.2f}")

print("\nProporsi premium per kelompok (bukti fitur pengecoh):")
print(df.groupby("n_sims")["premium"].agg(jumlah="count", premium="sum",
                                          proporsi="mean").round(3))
print(df.groupby("has_4g")["premium"].agg(jumlah="count", premium="sum",
                                          proporsi="mean").round(3))
print(df.groupby("os")["premium"].agg(jumlah="count", premium="sum",
                                      proporsi="mean").round(3))

# ============================================================ 5. SPLIT
header("5. TRAIN-TEST SPLIT (stratified)")
FEATURES_NUM = CORE_NUM + (DECOY_NUM if USE_DECOYS else [])
FEATURES_CAT = ["os", "brand"]
X = df[FEATURES_NUM + FEATURES_CAT]
y = df["premium"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train: {X_tr.shape} (premium {y_tr.mean():.1%}) | "
      f"Test: {X_te.shape} (premium {y_te.mean():.1%})")


# ============================================================ 6. PIPELINE
def build(num_cols, cat_cols):
    """Scaler & encoder dipelajari HANYA dari data training (anti data-leakage)."""
    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        # brand/os yang muncul < 10 kali digabung otomatis ke kategori 'infrequent'
        transformers.append(("cat", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=10), cat_cols))
    return Pipeline([
        ("pre", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                   random_state=SEED)),
    ])


# ============================================================ 7. CROSS-VALIDATION
header("6. CROSS-VALIDATION 5-FOLD (data training) - perbandingan skenario")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scenarios = {
    "A. Spesifikasi inti":               (CORE_NUM, []),
    "B. A + Operating System":           (CORE_NUM, ["os"]),
    "C. B + Brand":                      (CORE_NUM, ["os", "brand"]),
    "D. C + fitur pengecoh (SIM, 4G)":   (CORE_NUM + DECOY_NUM, ["os", "brand"]),
}
rows = []
for name, (nc, cc) in scenarios.items():
    cv = cross_validate(build(nc, cc), X_tr[nc + cc], y_tr, cv=skf,
                        scoring=["precision", "recall", "f1", "roc_auc"])
    rows.append({"Skenario": name,
                 "Precision": cv["test_precision"].mean(),
                 "Recall": cv["test_recall"].mean(),
                 "F1": cv["test_f1"].mean(),
                 "ROC-AUC": cv["test_roc_auc"].mean()})
print(pd.DataFrame(rows).set_index("Skenario").round(3))
print("-> Jika D tidak lebih baik dari C, fitur pengecoh tidak menambah nilai.")

# ============================================================ 8. MODEL FINAL
header("7. MODEL FINAL & EVALUASI DATA UJI (ambang 0,5)")
model = build(FEATURES_NUM, FEATURES_CAT)
model.fit(X_tr, y_tr)
proba_te = model.predict_proba(X_te)[:, 1]
pred_te = (proba_te >= 0.5).astype(int)
cm_te = confusion_matrix(y_te, pred_te)
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm_te)
print(classification_report(y_te, pred_te,
                            target_names=["non-premium", "premium"], digits=3))
print(f"ROC-AUC : {roc_auc_score(y_te, proba_te):.3f}")
print(f"Accuracy: {accuracy_score(y_te, pred_te):.3f} "
      f"(bandingkan dengan tebakan 'selalu non-premium' = {(y_te == 0).mean():.3f})")

# ============================================================ 9. TUNING AMBANG
header("8. TUNING AMBANG PROBABILITAS (dipilih dari CV training, bukan data uji)")
oof = cross_val_predict(build(FEATURES_NUM, FEATURES_CAT), X_tr, y_tr, cv=skf,
                        method="predict_proba")[:, 1]
grid = np.arange(0.30, 0.91, 0.05)
tab = pd.DataFrame({
    "ambang": grid.round(2),
    "precision": [precision_score(y_tr, oof >= t) for t in grid],
    "recall": [recall_score(y_tr, oof >= t) for t in grid],
    "F1": [f1_score(y_tr, oof >= t) for t in grid],
}).round(3)
print(tab.to_string(index=False))
best_t = float(grid[np.argmax([f1_score(y_tr, oof >= t) for t in grid])])
print(f"\nAmbang dengan F1 terbaik pada CV training: {best_t:.2f}")

pred_tuned = (proba_te >= best_t).astype(int)
print(f"\nData uji pada ambang {best_t:.2f}:")
print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_te, pred_tuned))
print(f"Precision={precision_score(y_te, pred_tuned):.3f}  "
      f"Recall={recall_score(y_te, pred_tuned):.3f}  F1={f1_score(y_te, pred_tuned):.3f}")

# ============================================================ 10. KOEFISIEN & PENGECOH
header("9. KOEFISIEN MODEL vs KORELASI (mengungkap fitur yang mengecoh)")
names = model.named_steps["pre"].get_feature_names_out()
coef = pd.Series(model.named_steps["clf"].coef_[0], index=names)
num_coef = coef[[f"num__{c}" for c in FEATURES_NUM]]
num_coef.index = FEATURES_NUM
cmp = pd.DataFrame({
    "korelasi_target": df[FEATURES_NUM].corrwith(df["premium"]),
    "koefisien_LR": num_coef,
}).round(3)
cmp["tanda_sama"] = np.sign(cmp["korelasi_target"]) == np.sign(cmp["koefisien_LR"])
cmp["jenis"] = ["PENGECOH" if c in DECOY_NUM else "inti" for c in cmp.index]
print(cmp.sort_values("koefisien_LR", ascending=False))
print("\nKoefisien kategori (OS & brand), 8 terbesar:")
print(coef[[n for n in names if n.startswith("cat__")]]
      .sort_values(ascending=False).head(8).round(2).to_string())

print("""
PENJELASAN FITUR PENGECOH
- n_sims (Number of SIMs): korelasi NEGATIF dengan premium karena banyak flagship
  (Samsung, Sony, Google, Apple) berjenis 1 SIM di data ini. Itu penanda gaya
  produk/pasar, bukan penyebab harga tinggi. Jangan ditafsirkan "SIM sedikit = mahal".
- has_4g (4G/LTE): hubungan lemah dan lebih mencerminkan tahun rilis ponsel; datanya
  juga tidak konsisten dengan kolom 3G (lihat bagian cleaning).
- Cek juga baris 'tanda_sama' = False di tabel atas: itu fitur yang korelasinya searah
  tetapi koefisien modelnya berbalik karena efeknya sudah diwakili fitur lain
  (multikolinearitas), sehingga tidak boleh ditafsirkan sendiri-sendiri.
""")

# ============================================================ 11. GRAFIK
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

order = cmp.sort_values("korelasi_target").index
colors = ["#d95f02" if c in DECOY_NUM else "#1b9e77" for c in order]
ax[0].barh(order, cmp.loc[order, "korelasi_target"], color=colors)
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_title("Korelasi dengan target premium\n(oranye = fitur pengecoh)")

ax[1].barh(order, cmp.loc[order, "koefisien_LR"], color=colors)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_title("Koefisien Logistic Regression\n(fitur terstandardisasi)")

im = ax[2].imshow(cm_te, cmap="Blues")
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["Pred non-premium", "Pred premium"])
ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["Aktual non-premium", "Aktual premium"])
for i in range(2):
    for j in range(2):
        ax[2].text(j, i, cm_te[i, j], ha="center", va="center",
                   color="white" if cm_te[i, j] > cm_te.max() / 2 else "black",
                   fontsize=14)
ax[2].set_title("Confusion matrix data uji (ambang 0,5)")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=130)
print(f"Grafik disimpan: {OUT_FIG}")
print(f"Data bersih disimpan: {OUT_CSV}")
