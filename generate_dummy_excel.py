import numpy as np
import pandas as pd

# Kolom master (sesuai data Anda)[^1]
COLUMNS = [
    "NIM",
    "Nama Lengkap",
    "Kelas",
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
    "Asisten Praktikum",
    "MBKM",
    "KP",
    "Lomba",
    "Penelitian",
    "Abdimas",
    "Sertifikasi",
    "Minat SAGE",
    "Minat DELTA",
    "Target",
]

SUBJECTS = [
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

ACTIVITIES = ["Asisten Praktikum", "MBKM", "KP", "Lomba", "Penelitian", "Abdimas", "Sertifikasi"]


def generate_dummy(n_rows=120, seed=42):
    rng = np.random.default_rng(seed)

    kelas_options = [
        "S1SI-KJ-23-01",
        "S1SI-KJ-23-02",
        "S1SI-KJ-23-03",
        "S1SI-KJ-23-04",
        "S1SI-KJ-23-05",
    ]

    rows = []
    for i in range(n_rows):
        nim = int(rng.integers(311300000, 311300999))
        nama = f"Mahasiswa_{i+1}"

        kelas = rng.choice(kelas_options)

        # nilai mata kuliah 50-100 (agar realistis)
        subj = rng.integers(50, 101, size=len(SUBJECTS))

        # aktivitas biner 0/1 (bisa juga dibuat bernilai 0-5 kalau perlu)
        act = rng.integers(0, 2, size=len(ACTIVITIES))

        # minat (dibuat mirip "range skor" di data Anda: contoh ada 0-100)
        minat_sage = int(rng.integers(20, 101))
        minat_delta = int(rng.integers(20, 101))

        # rule dummy untuk menentukan Target (biar ada pola)
        # skor akademik ~ rata-rata nilai
        akademik = subj.mean()

        # skor aktivitas = jumlah aktivitas
        skor_aktivitas = act.sum() * 3  # scaling

        # minat (lebih "dominan" ke sisi yang lebih besar)
        bias_minat = (minat_sage - minat_delta) * 0.2

        overall = akademik + skor_aktivitas + bias_minat

        target = "SAGE" if overall >= 80 else "DELTA"

        row = {
            "NIM": nim,
            "Nama Lengkap": nama,
            "Kelas": kelas,
            **{SUBJECTS[j]: int(subj[j]) for j in range(len(SUBJECTS))},
            **{ACTIVITIES[j]: int(act[j]) for j in range(len(ACTIVITIES))},
            "Minat SAGE": minat_sage,
            "Minat DELTA": minat_delta,
            "Target": target,
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Pastikan tiap baris unik: NIM & nilai sudah random -> otomatis berbeda
    return df


if __name__ == "__main__":
    df = generate_dummy(n_rows=200, seed=7)

    out_path = "s1si-ml/data/data_latih_dummy.xlsx"
    import os
    os.makedirs("data", exist_ok=True)

    df.to_excel(out_path, index=False)
    print("Selesai generate:", out_path)
    print(df.head())
