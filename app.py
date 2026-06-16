# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

import plotly.express as px

# =========================
# Page config + CSS (mirip contoh)
# =========================
st.set_page_config(
    page_title="Dashboard Peminatan Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(180deg, #f6f8fb 0%, #ffffff 40%);
    }
    .header {
        padding: 10px 0;
        text-align: left;
    }
    .title {
        font-size:28px;
        font-weight:600;
        color:#0b3d91;
        margin-bottom: 0;
    }
    .subtitle {
        color:#6b7280;
        margin-top: 0;
        font-size:14px;
    }
    .kpi {
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0.9),
            rgba(255,255,255,0.9)
        );
        border-radius: 12px;
        padding: 14px;
        border: 1px solid rgba(16,24,40,0.06);
        box-shadow: 0 1px 6px rgba(16,24,40,0.06);

        min-height: 120px;

        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .small { color:#6b7280; font-size:12px; }
    .big { font-size:22px; font-weight:700; color:#111827; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Header
# =========================
col_h1, col_h2 = st.columns([8, 2])
with col_h1:
    st.markdown(
        """
        <div class="header">
            <div class="title">📊 Dashboard Analisis Peminatan S1SI</div>
            <div class="subtitle">Prediksi Target SAGE/DELTA dari Nilai + Aktivitas (anti-leakage Minat)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f"<div style='text-align:right'><small>Versi: 1.0 • {pd.Timestamp.now().strftime('%Y-%m-%d')}</small></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================
# Kolom master (sesuaikan dengan file Anda)
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

# Anti-leakage: Minat tidak dipakai sebagai feature
FEATURE_COLS = SUBJECT_COLS + ACTIVITY_COLS

ID_COLS = ["NIM", "Nama Lengkap", "Kelas"]
MINAT_COLS = ["Minat SAGE", "Minat DELTA"]
LABEL_COL = "Target"

# =========================
# Load model (+ optional scaler)
# =========================
@st.cache_resource
def load_model_scaler():
    model_path = "model_peminatan.pkl"
    scaler_path = "scaler_peminatan.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        scaler = None
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_model_scaler()


model_info = {}

try:
    with open("model_info.pkl", "rb") as f:
        model_info = pickle.load(f)
except:
    model_info = {}

if load_error:
    st.warning(f"Model/scaler tidak ditemukan: {load_error}")
    st.info("Jalankan training terlebih dahulu (buat model_peminatan.pkl & scaler_peminatan.pkl jika pakai scaler).")
# =========================
# Sidebar
# =========================
import os

st.sidebar.markdown("## 📁 Upload & Tools")

uploaded_file = st.sidebar.file_uploader(
    "⬆️ Unggah file Excel (.xlsx)",
    type=["xlsx"]
)

# Simpan file yang diupload
if uploaded_file is not None:
    with open("dataset_aktif.xlsx", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("✅ Dataset aktif berhasil disimpan")

# Tampilkan info dataset aktif
if os.path.exists("dataset_aktif.xlsx"):
    df_info = pd.read_excel("dataset_aktif.xlsx")

    st.sidebar.metric(
        "📂 Dataset Aktif",
        len(df_info)
    )

st.sidebar.markdown("### Template / Utilities")


if st.sidebar.button("⬇️ Download Template Excel"):
    # Template minimal: isi semua kolom master agar sesuai format
    template = {
        "NIM": ["311300001"],
        "Nama Lengkap": ["Budi Santoso"],
        "Kelas": ["S1SI-KJ-23-01"],
    }

    # Subjek: pakai contoh angka
    for c in SUBJECT_COLS:
        template[c] = [75]

    # Aktivitas: contoh biner (0/1) seperti dataset Anda
    # (Kalau di dataset Anda kadang 0-5, tetap boleh; model akan menyesuaikan.)
    for c in ACTIVITY_COLS:
        template[c] = [1]

    # Minat & target (boleh diisi; untuk prediksi anti-leakage, Minat tidak dipakai fitur)
    template["Minat SAGE"] = [50]
    template["Minat DELTA"] = [60]
    template["Target"] = ["SAGE"]

    buffer = BytesIO()
    pd.DataFrame(template).to_excel(buffer, index=False)
    buffer.seek(0)
    st.sidebar.download_button(
        "Klik untuk download template",
        buffer,
        "template_peminatan_s1si.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Jika nama kolom berbeda, sesuaikan daftar kolom di kode.")


# =========================
# Informasi Model
# =========================
with st.sidebar.expander("🤖 Informasi Model", expanded=False):

    st.markdown(
        f"""
**Algoritma**  
{model_info.get("algorithm", "Random Forest")}

**Akurasi Model**  
{model_info.get("accuracy", 0):.2%}

**Jumlah Feature**  
{model_info.get("n_features", len(FEATURE_COLS))}

**Target**  
{model_info.get("target", "SAGE / DELTA")}
"""
    )


# =========================
# Jika belum upload
# =========================
if not os.path.exists("dataset_aktif.xlsx"):
    st.info("👆 Upload file Excel di sidebar untuk melihat prediksi dan dashboard.")
    st.stop()

# =========================
# Load Excel
# =========================
try:
    df = pd.read_excel(
    "dataset_aktif.xlsx",
    dtype={
        "NIM": str,
        "Nama Lengkap": str,
        "Kelas": str
        }
    )

    # Pastikan NIM tetap text
    df["NIM"] = df["NIM"].astype(str).str.strip()

    st.success(f"📥 File berhasil dimuat: {len(df)} baris")

except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

# =========================
# Validasi kolom
# =========================
required_for_pred = ID_COLS[:2] + FEATURE_COLS  # NIM, Nama Lengkap + fitur
missing = [c for c in required_for_pred if c not in df.columns]
if missing:
    st.error("Kolom yang hilang di file: " + ", ".join(missing))
    st.stop()

if model is None:
    st.error("Model belum tersedia. Jalankan training terlebih dahulu.")
    st.stop()

# =========================
# Preview
# =========================
with st.expander("Lihat preview data", expanded=False):
    st.dataframe(df.head(25), use_container_width=True)

# =========================
# Prediksi (anti-leakage: pakai nilai+aktivitas saja)
# =========================
X = df[FEATURE_COLS].copy()
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

if scaler is not None:
    X_scaled = scaler.transform(X)
else:
    X_scaled = X

pred = model.predict(X_scaled)

# Mapping sesuai contoh Anda: DELTA=1, SAGE=0
# artinya: 1 => DELTA, 0 => SAGE
pred_labels = pd.Series(pred).map({0: "SAGE", 1: "DELTA"}).astype(str)

df_out = df.copy()

# Paksa NIM tetap text
df_out["NIM"] = df_out["NIM"].astype(str)

df_out["Prediksi_Lab"] = pred_labels

# Paksa seluruh fitur menjadi numerik
for col in FEATURE_COLS:
    if col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

# Probabilitas + confidence
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_scaled)
    # Pastikan urutan kelas sesuai training (umumnya [0,1])
    # Kita amankan dengan asumsi model class_ = [0,1]
    # Jika beda, Anda bisa sesuaikan mapping.
    df_out["Probabilitas_DELTA"] = probs[:, 1] * 100
    df_out["Probabilitas_SAGE"] = probs[:, 0] * 100
    df_out["Confidence"] = df_out[["Probabilitas_SAGE", "Probabilitas_DELTA"]].max(axis=1)
else:
    df_out["Probabilitas_SAGE"] = np.nan
    df_out["Probabilitas_DELTA"] = np.nan
    df_out["Confidence"] = np.nan


# =========================
# Ringkasan Prediksi
# =========================

pred_counts = df_out["Prediksi_Lab"].value_counts()

# =========================
# KPI
# =========================

total = len(df_out)

count_sage = int(
    (df_out["Prediksi_Lab"] == "SAGE").sum()
)

count_delta = int(
    (df_out["Prediksi_Lab"] == "DELTA").sum()
)

avg_conf = (
    float(df_out["Confidence"].mean())
    if df_out["Confidence"].notna().any()
    else 0.0
)

# =========================
# Tabs
# =========================
tab_summary, tab_table, tab_vis, tab_export = st.tabs(
    ["📈 Ringkasan", "👥 Detail Data", "📊 Visualisasi", "⬇️ Export"]
)

# =========================
# TAB: Ringkasan
# =========================
with tab_summary:

    st.subheader("Ringkasan Hasil Prediksi")

    # ==========================================
    # Insight Otomatis
    # ==========================================

    persen_sage = count_sage / total * 100
    persen_delta = count_delta / total * 100

    st.info(
        f"""
📌 Dari {total} mahasiswa yang dianalisis,
sebanyak {count_sage} mahasiswa ({persen_sage:.1f}%)
direkomendasikan ke Laboratorium SAGE dan
{count_delta} mahasiswa ({persen_delta:.1f}%)
direkomendasikan ke Laboratorium DELTA.
"""
    )

    # ==========================================
    # Top Feature Importance
    # ==========================================

    st.markdown("### 🔝 Faktor Paling Berpengaruh")

    if hasattr(model, "feature_importances_"):

        fi_df = (
            pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": model.feature_importances_
            })
            .sort_values("Importance", ascending=False)
        )

        st.dataframe(
            fi_df.head(5),
            hide_index=True,
            use_container_width=True
        )

    # ==========================================
    # Kesimpulan
    # ==========================================

    lab_dominan = pred_counts.idxmax()

    st.success(
        f"""
🏆 Mayoritas mahasiswa lebih sesuai dengan Laboratorium {lab_dominan} berdasarkan hasil prediksi model Machine Learning.
"""
    )

    # ==========================================
    # Statistik Confidence
    # ==========================================

    if df_out["Confidence"].notna().any():

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Confidence Minimum",
                f"{df_out['Confidence'].min():.1f}%"
            )

        with col2:
            st.metric(
                "Confidence Rata-rata",
                f"{df_out['Confidence'].mean():.1f}%"
            )

        with col3:
            st.metric(
                "Confidence Maksimum",
                f"{df_out['Confidence'].max():.1f}%"
            )

# =========================
# TAB: Detail Data
# =========================
with tab_table:
    st.subheader("Tabel Prediksi & Filter")

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        search = st.text_input("Cari NIM atau Nama (substring)", "")
    with c2:
        lab_filter = st.multiselect(
            "Filter Peminatan", options=["SAGE", "DELTA"], default=["SAGE", "DELTA"]
        )
    with c3:
        min_conf = st.slider("Min Confidence (%)", 0, 100, 0)

    df_filter = df_out[df_out["Prediksi_Lab"].isin(lab_filter)].copy()

    if search.strip():
        mask = df_filter["NIM"].astype(str).str.contains(search, case=False, na=False) | \
               df_filter["Nama Lengkap"].astype(str).str.contains(search, case=False, na=False)
        df_filter = df_filter[mask]

    if df_filter["Confidence"].notna().any():
        df_filter = df_filter[df_filter["Confidence"] >= min_conf]

    df_filter = df_filter.sort_values("Confidence", ascending=False) if df_filter["Confidence"].notna().any() else df_filter
    st.dataframe(
        df_filter.reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 🔍 Profil Mahasiswa")

    choose_nim = st.selectbox("Pilih NIM untuk lihat detail", options=df_out["NIM"].astype(str).tolist())
    student = df_out[df_out["NIM"].astype(str) == str(choose_nim)].iloc[0]

    st.markdown(
        f"**{student['Nama Lengkap']}** — Prediksi: **{student['Prediksi_Lab']}**"
        f" — Confidence: **{student['Confidence']:.1f}%**"
        if pd.notna(student.get("Confidence", np.nan))
        else f"**{student['Nama Lengkap']}** — Prediksi: **{student['Prediksi_Lab']}**"
    )

    # Radar/bar untuk beberapa fitur (ambil 6 mapel biar ringkas)
    radar_features = [
        SUBJECT_COLS[0],
        SUBJECT_COLS[1],
        SUBJECT_COLS[2],
        SUBJECT_COLS[6],
        SUBJECT_COLS[9],
        SUBJECT_COLS[10],
    ]
    vals = [student[f] if f in student.index else 0 for f in radar_features]
    radar_df = pd.DataFrame({"feature": radar_features, "value": vals})

    fig_radar = px.line_polar(
        radar_df,
        r="value",
        theta="feature",
        line_close=True,
        title="Profil Nilai (sample fitur)",
        markers=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# =========================
# TAB: Visualisasi
# =========================
with tab_vis:

    # =====================================================
    # Distribusi Hasil Prediksi
    # =====================================================

    st.markdown("### Distribusi Hasil Prediksi")

    sage_count = pred_counts.get("SAGE", 0)
    delta_count = pred_counts.get("DELTA", 0)
    total = len(df_out)

    # ==========================================
    # KPI
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="kpi kpi-total">
                <div class="small">👥 Total Mahasiswa</div>
                <div class="big">{total}</div>
                <div class="small">Data dianalisis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi kpi-sage">
                <div class="small">🟦 Prediksi SAGE</div>
                <div class="big">{sage_count}</div>
                <div class="small">{sage_count/total*100:.1f}% dari total mahasiswa</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi kpi-delta">
                <div class="small">🟩 Prediksi DELTA</div>
                <div class="big">{delta_count}</div>
                <div class="small">{delta_count/total*100:.1f}% dari total mahasiswa</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================
    # Statistik Distribusi
    # ==========================================

    summary_df = pd.DataFrame({
        "Laboratorium": pred_counts.index,
        "Jumlah": pred_counts.values
    })

    summary_df["Persentase"] = (
        summary_df["Jumlah"]
        / summary_df["Jumlah"].sum()
        * 100
    ).map(lambda x: f"{x:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_info = st.columns([2, 1])

    with col_chart:

        fig = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            hole=0.55,
            title="Distribusi Peminatan Laboratorium"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend_title="Laboratorium",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col_info:

        st.markdown("#### 📋 Statistik")

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True
        )

        st.markdown("#### 🎯 Insight")

        st.success(
            f"""
Mayoritas mahasiswa direkomendasikan ke
Laboratorium {pred_counts.idxmax()}.
"""
        )

    # =====================================================
    # Perbandingan Mata Kuliah
    # =====================================================

    st.markdown("---")
    st.markdown("### Perbandingan Nilai Rata-rata Mata Kuliah")

    compare_subjects = [
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

    temp = df_out.copy()

    for col in compare_subjects:
        temp[col] = pd.to_numeric(
            temp[col],
            errors="coerce"
        )

    pivot_long = (
        temp.groupby("Prediksi_Lab")[compare_subjects]
        .mean(numeric_only=True)
        .reset_index()
        .melt(
            id_vars="Prediksi_Lab",
            value_vars=compare_subjects,
            var_name="Mata Kuliah",
            value_name="Rata-rata"
        )
    )

    fig_compare = px.bar(
        pivot_long,
        x="Mata Kuliah",
        y="Rata-rata",
        color="Prediksi_Lab",
        barmode="group",
        title="Perbandingan Nilai Rata-rata Mata Kuliah",
        height=600
    )

    fig_compare.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    # =====================================================
    # Feature Importance
    # =====================================================

    st.markdown("---")
    st.markdown("### Feature Importance")

    if hasattr(model, "feature_importances_"):

        fi_df = (
            pd.DataFrame({
                "feature": FEATURE_COLS,
                "importance": model.feature_importances_
            })
            .sort_values(
                "importance",
                ascending=False
            )
        )

        col_fi1, col_fi2 = st.columns([2, 1])

        with col_fi1:

            fig_fi = px.bar(
                fi_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance",
                height=500
            )

            st.plotly_chart(
                fig_fi,
                use_container_width=True
            )

        with col_fi2:

            st.markdown("#### 🔝 Top 10 Feature")

            st.dataframe(
                fi_df.head(10),
                hide_index=True,
                use_container_width=True
            )

    # =====================================================
    # Heatmap Korelasi
    # =====================================================

    st.markdown("---")
    st.markdown("### Korelasi Antar Variabel")

    corr = df_out[FEATURE_COLS].corr(
        numeric_only=True
    )

    fig_corr = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        title="Heatmap Korelasi Feature"
    )

    fig_corr.update_layout(
        height=700
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

# =========================
# TAB: Export
# =========================
with tab_export:
    st.subheader("Export Hasil")

    out_cols = ["NIM", "Nama Lengkap", "Kelas", "Prediksi_Lab"]
    if "Probabilitas_SAGE" in df_out.columns and "Probabilitas_DELTA" in df_out.columns:
        out_cols += ["Probabilitas_SAGE", "Probabilitas_DELTA"]
    if "Confidence" in df_out.columns:
        out_cols += ["Confidence"]

    out_df = df_out[out_cols].copy()

    csv = out_df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "hasil_peminatan.csv", "text/csv")

    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="DataAll", index=False)
        out_df.to_excel(writer, sheet_name="Prediksi", index=False)
    towrite.seek(0)

    st.download_button(
        "⬇️ Download Excel (Full)",
        towrite,
        "hasil_peminatan_full.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.markdown(f"© {pd.Timestamp.now().year} Dashboard Peminatan Lab")
