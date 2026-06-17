# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings("ignore")

# =========================
# 1) Baca data
# =========================
excel_path = "/Users/arifrahmanhakim/Desktop/PEMINATAN/s1si-ml/src/data/data_latih_dummy.xlsx"  # <-- ganti sesuai file Anda
df = pd.read_excel(excel_path)

# =========================
# 2) Definisi kolom (sesuai master)
# =========================
SUBJECT_COLS = [
    "Pemrograman Berorientasi Objek",
    "Analisis dan Perancangan Sistem Informasi",
    "Algoritma dan Pemrograman",
    "Pengembangan Aplikasi Website",
    "Arsitektur Enterprise",
    "Tata Kelola dan Manajemen Teknologi Informasi",
    "Penambangan Data",
    "Probabilitas dan Statistik",
    "Sistem Basis Data",
    "Statistika Industri",
    "Data Warehouse dan Business Intelligence",
    "Matematika Diskrit",
]

ACTIVITY_COLS = [
    "Asisten Praktikum",
    "MBKM",
    "KP",
    "Lomba",
    "Penelitian",
    "Abdimas",
    "Sertifikasi",
]

MINAT_COLS = ["Minat SAGE", "Minat DELTA"]

# Anti data leakage:
# minat dipakai hanya jika Anda benar-benar butuh sebagai feature (debug).
USE_MINAT_FEATURES = False

feature_columns = SUBJECT_COLS + ACTIVITY_COLS
if USE_MINAT_FEATURES:
    feature_columns += MINAT_COLS

# =========================
# 3) Input feature & label
# =========================
X = df[feature_columns].copy()
y = df["Target"].copy()

# Encoding target: DELTA = 1, SAGE = 0
y_encoded = (y == "DELTA").astype(int)

# Opsional: pastikan numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# =========================
# 4) Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

# =========================
# 5) Standardisasi
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6) Training model
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=12,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# =========================
# 7) Evaluasi
# =========================
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

# =========================
# Simpan informasi model
# =========================
model_info = {
    "accuracy": float(accuracy),
    "algorithm": "Random Forest",
    "n_features": len(feature_columns),
    "target": "SAGE / DELTA"
}

with open("model_info.pkl", "wb") as f:
    pickle.dump(model_info, f)

# =========================
# 8) Simpan model dan scaler
# =========================
with open("model_peminatan.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler_peminatan.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model dan Scaler berhasil disimpan!")
