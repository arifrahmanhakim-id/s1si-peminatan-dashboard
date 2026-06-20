# =========================
# app_streamlit.py - FINAL VERSION (FIXED)
# Dashboard Peminatan Laboratorium S1SI
# Menggunakan joblib untuk load model & confidence score
# PERBAIKAN: Confidence Distribution Chart
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from io import BytesIO
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =========================
# CONSTANTS & COLUMNS
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

FEATURE_COLS = SUBJECT_COLS + ACTIVITY_COLS
ID_COLS = ["NIM", "Nama Lengkap", "Kelas"]
MINAT_COLS = ["Minat SAGE", "Minat DELTA"]
LABEL_COL = "Target"

# =========================
# COLOR PALETTE
# =========================
COLORS = {
    "maroon": "#6B0F1A",
    "maroon_light": "#8B1325",
    "maroon_dark": "#4A0A13",
    "gray": "#6B7280",
    "gray_light": "#9CA3AF",
    "gray_dark": "#374151",
    "neutral": "#F3F4F6",
    "border": "#E5E7EB",
    "text_primary": "#111827",
    "text_muted": "#6B7280",
    "white": "#FFFFFF",
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Peminatan Lab",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS STYLING
# =========================
st.markdown(
    f"""
    <link rel="stylesheet" 
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">

    <style>
    /* ===== GLOBAL ===== */
    * {{
        font-family: 'Segoe UI', 'Helvetica Neue', -apple-system, sans-serif;
    }}

    .stApp {{
        background-color: {COLORS['neutral']};
    }}

    /* ===== HEADER ===== */
    .header-container {{
        padding: 28px 0;
        border-bottom: 2px solid {COLORS['maroon']};
        margin-bottom: 28px;
        background: linear-gradient(135deg, rgba(107, 15, 26, 0.03) 0%, rgba(107, 15, 26, 0.01) 100%);
    }}

    .header-title {{
        font-size: 32px;
        font-weight: 700;
        color: {COLORS['maroon']};
        margin: 0;
        display: flex;
        align-items: center;
        gap: 16px;
    }}

    .header-title i {{
        font-size: 36px;
        color: {COLORS['maroon']};
    }}

    .header-subtitle {{
        font-size: 14px;
        color: {COLORS['text_muted']};
        margin: 12px 0 0 52px;
        line-height: 1.6;
    }}

    /* ===== KPI CARD ===== */
    .kpi-card {{
        background: {COLORS['white']};
        border-radius: 14px;
        padding: 24px;
        border: 2px solid var(--accent-color);
        box-shadow: 0 4px 16px rgba(107, 15, 26, 0.1);
        display: grid;
        grid-template-columns: 1fr auto;
        grid-template-rows: auto auto auto;
        gap: 8px 16px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        min-height: 160px;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--accent-color);
    }}

    .kpi-card:hover {{
        box-shadow: 0 12px 32px rgba(107, 15, 26, 0.2);
        transform: translateY(-6px);
    }}

    .kpi-label {{
        font-size: 11px;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        grid-column: 1;
        grid-row: 1;
        margin: 0;
    }}

    .kpi-value {{
        font-size: 48px;
        font-weight: 900;
        color: var(--accent-color);
        line-height: 1;
        grid-column: 1;
        grid-row: 2;
        margin: 0;
        letter-spacing: -1px;
    }}

    .kpi-sub {{
        font-size: 12px;
        color: {COLORS['gray']};
        grid-column: 1;
        grid-row: 3;
        margin: 0;
        line-height: 1.4;
    }}

    .kpi-icon {{
        font-size: 56px;
        color: var(--accent-color);
        grid-column: 2;
        grid-row: 1 / 3;
        display: flex;
        align-items: flex-start;
        justify-content: flex-end;
        opacity: 0.12;
        margin-top: -8px;
    }}

    .accent-maroon {{ --accent-color: {COLORS['maroon']}; }}
    .accent-gray {{ --accent-color: {COLORS['gray']}; }}
    .accent-dark {{ --accent-color: {COLORS['maroon_dark']}; }}

    .card {{
        background: {COLORS['white']};
        border-radius: 10px;
        padding: 20px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}

    .insight-box {{
        background: linear-gradient(135deg, rgba(107, 15, 26, 0.08) 0%, rgba(107, 15, 26, 0.04) 100%);
        border-left: 4px solid {COLORS['maroon']};
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
    }}

    .insight-title {{
        font-weight: 700;
        font-size: 14px;
        color: {COLORS['maroon']};
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .insight-text {{
        font-size: 13px;
        line-height: 1.7;
        color: {COLORS['text_muted']};
    }}

    .narasi-box {{
        background: {COLORS['white']};
        border-left: 4px solid {COLORS['gray']};
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        font-size: 13px;
        line-height: 1.8;
        color: {COLORS['text_muted']};
        border: 1px solid {COLORS['border']};
    }}

    .stat-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}

    .stat-table th {{
        background-color: {COLORS['maroon']};
        color: {COLORS['white']};
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }}

    .stat-table td {{
        padding: 12px;
        border-bottom: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
    }}

    .stat-table tr:hover {{
        background-color: {COLORS['neutral']};
    }}

    .divider {{
        height: 2px;
        background: linear-gradient(90deg, {COLORS['maroon']} 0%, transparent 100%);
        margin: 28px 0;
        border: none;
    }}

    .stButton > button {{
        border-radius: 8px;
        background-color: {COLORS['maroon']};
        color: {COLORS['white']};
        font-weight: 600;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        background-color: {COLORS['maroon_dark']};
        box-shadow: 0 6px 16px rgba(107, 15, 26, 0.3);
    }}

    [data-testid="metric-container"] {{
        display: flex;
        flex-direction: column;
    }}

    [data-testid="metric-container"] > div:first-child {{
        font-size: 14px !important;
        font-weight: 600 !important;
        color: {COLORS['text_primary']} !important;
    }}

    [data-testid="metric-container"] > div:last-child {{
        font-size: 10px !important;
        font-weight: 500 !important;
        color: {COLORS['text_muted']} !important;
    }}

    .text-maroon {{ color: {COLORS['maroon']}; }}
    .text-gray {{ color: {COLORS['gray']}; }}
    .text-sm {{ font-size: 12px; }}
    .text-lg {{ font-size: 16px; font-weight: 700; }}

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPER FUNCTIONS - FIXED
# =========================
@st.cache_resource
def load_model_scaler_info():
    """Load model, scaler, dan model info menggunakan joblib."""
    try:
        # Load model menggunakan joblib (lebih reliable untuk sklearn)
        model = joblib.load("model_peminatan.pkl")

        # Load scaler menggunakan joblib
        scaler = joblib.load("scaler_peminatan.pkl")

        # Load model info menggunakan pickle
        with open("model_info.pkl", "rb") as f:
            model_info = pickle.load(f)

        return model, scaler, model_info, None
    except Exception as e:
        return None, None, {}, str(e)


@st.cache_data
def safe_read_excel(path):
    """Read Excel file dengan type handling."""
    return pd.read_excel(
        path,
        dtype={"NIM": str, "Nama Lengkap": str, "Kelas": str}
    )


# =========================
# LOAD MODEL - PERBAIKAN UTAMA
# =========================
model, scaler, model_info, load_error = load_model_scaler_info()

# Debug info
if load_error:
    st.error(f"⚠️ Error loading model: {load_error}")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
            <i class='fas fa-file-excel' style='font-size: 20px; color: #6B0F1A;'></i>
            <div style='font-size: 16px; font-weight: 700; color: #111827;'>Upload Dataset</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Pilih file Excel (.xlsx)",
        type=["xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with open("dataset_aktif.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✓ Dataset berhasil disimpan")

    # Dataset info
    if os.path.exists("dataset_aktif.xlsx"):
        try:
            df_info = safe_read_excel("dataset_aktif.xlsx")
            # Dataset info
            st.markdown(
                """
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px;'>
                    <i class='fas fa-folder-open' style='font-size: 16px; color: #6B0F1A;'></i>
                    <div style='font-size: 14px; font-weight: 700; color: #111827;'>Dataset Information</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if os.path.exists("dataset_aktif.xlsx"):
                try:
                    df_info = safe_read_excel("dataset_aktif.xlsx")

                    st.caption(f"**Rows:** {len(df_info)}")
                    st.caption(f"**Columns:** {len(df_info.columns)}")
                    st.caption(f"**File:** dataset_aktif.xlsx")

                except Exception:
                    st.warning("⚠️ Dataset tidak dapat dibaca")
            else:
                st.warning("⚠️ Dataset belum diunggah")

            st.divider()
        except Exception:
            pass

    # Template
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px;'>
            <i class='fas fa-download' style='font-size: 16px; color: #6B0F1A;'></i>
            <div style='font-size: 14px; font-weight: 700; color: #111827;'>Template</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Download Template Excel", use_container_width=True, key="download_template"):
        template = {
            "NIM": ["311300001"],
            "Nama Lengkap": ["Budi Santoso"],
            "Kelas": ["S1SI-KJ-23-01"],
        }
        for c in SUBJECT_COLS:
            template[c] = [75]
        for c in ACTIVITY_COLS:
            template[c] = [1]
        template["Minat SAGE"] = [50]
        template["Minat DELTA"] = [60]
        template["Target"] = ["SAGE"]

        buffer = BytesIO()
        pd.DataFrame(template).to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "Klik untuk download",
            buffer,
            "template_peminatan_s1si.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="template_download"
        )

    st.divider()

    # Model info
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px;'>
            <i class='fas fa-brain' style='font-size: 16px; color: #6B0F1A;'></i>
            <div style='font-size: 14px; font-weight: 700; color: #111827;'>Model Information</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model_info:
        st.caption(f"**Algoritma:** {model_info.get('algorithm', 'Random Forest')}")
        st.caption(f"**Akurasi:** {model_info.get('accuracy', 0):.2%}")
        st.caption(f"**Features:** {model_info.get('n_features', len(FEATURE_COLS))}")
    else:
        st.warning("⚠️ Model info tidak tersedia")

    st.divider()

# =========================
# MAIN HEADER
# =========================
st.markdown(
    f"""
    <div class='header-container'>
        <div class='header-title'>
            <i class='fas fa-chart-bar'></i>
            Dashboard Peminatan Laboratorium
        </div>
        <div class='header-subtitle'>
            Sistem prediksi minat mahasiswa S1 Sistem Informasi ke Laboratorium SAGE (Software & Development) 
            atau DELTA (Data & Analytics) berbasis nilai akademik dan aktivitas tambahan.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# REQUIRE DATASET
# =========================
if not os.path.exists("dataset_aktif.xlsx"):
    st.info("👈 **Langkah pertama:** Upload file Excel di sidebar untuk memulai analisis")
    st.stop()

try:
    df = safe_read_excel("dataset_aktif.xlsx")
    df["NIM"] = df["NIM"].astype(str).str.strip()

    if "file_loaded_message" not in st.session_state:
        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            st.markdown(
                f"""
                <div class='insight-box'>
                    <div class='insight-title'>
                        <i class='fas fa-check-circle'></i> File Berhasil Dimuat
                    </div>
                    <div class='insight-text'>
                        ✓ File dimuat: <strong>{len(df)} mahasiswa</strong>. 
                        Dataset siap untuk analisis prediksi peminatan laboratorium.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("✕", key="close_message", help="Tutup pesan"):
                st.session_state.file_loaded_message = True
                st.rerun()

        if "file_loaded_message" not in st.session_state:
            st.session_state.file_loaded_message = False

except Exception as e:
    st.error(f"❌ Gagal membaca file: {e}")
    st.stop()

# Validasi kolom
required_for_pred = ID_COLS[:2] + FEATURE_COLS
missing = [c for c in required_for_pred if c not in df.columns]
if missing:
    st.error(f"❌ Kolom yang hilang: {', '.join(missing)}")
    st.stop()

if model is None:
    st.error("❌ Model belum tersedia. Jalankan training terlebih dahulu dengan menjalankan train_model.py")
    st.stop()

# =========================
# PREDIKSI - PERBAIKAN CONFIDENCE
# =========================
try:
    # Prepare features
    X = df[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Standardisasi dengan scaler
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"❌ Error saat standardisasi: {e}")
            st.stop()
    else:
        X_scaled = X

    # Prediksi
    pred = model.predict(X_scaled)
    pred_labels = pd.Series(pred).map({0: "SAGE", 1: "DELTA"}).astype(str)

    # ✅ PENTING: Get probabilities untuk confidence score
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_scaled)
            prob_sage = probs[:, 0] * 100
            prob_delta = probs[:, 1] * 100
            confidence = np.max(probs, axis=1) * 100

            # Validasi confidence terisi
            if np.isnan(confidence).all():
                st.warning("⚠️ Semua confidence NaN! Menggunakan default 80%")
                confidence = np.full(len(X), 80.0)

        except Exception as e:
            st.error(f"⚠️ Error predict_proba: {e}")
            confidence = np.full(len(X), 50.0)
            prob_sage = np.full(len(X), 50.0)
            prob_delta = np.full(len(X), 50.0)
    else:
        st.warning("⚠️ Model tidak mendukung predict_proba()")
        confidence = np.full(len(X), 50.0)
        prob_sage = np.full(len(X), 50.0)
        prob_delta = np.full(len(X), 50.0)

except Exception as e:
    st.error(f"❌ Error saat prediksi: {e}")
    st.stop()

# Buat dataframe output
df_out = df.copy()
df_out["NIM"] = df_out["NIM"].astype(str)
df_out["Prediksi Lab"] = pred_labels
df_out["Probabilitas SAGE"] = prob_sage
df_out["Probabilitas DELTA"] = prob_delta
df_out["Confidence"] = confidence

# Numeric columns
for col in FEATURE_COLS:
    if col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

# =========================
# SUMMARY METRICS
# =========================
pred_counts = df_out["Prediksi Lab"].value_counts()
total = len(df_out)
count_sage = int((df_out["Prediksi Lab"] == "SAGE").sum())
count_delta = int((df_out["Prediksi Lab"] == "DELTA").sum())

# =========================
# TABS
# =========================
tab_summary, tab_table, tab_vis, tab_export = st.tabs(
    ["⊞ Ringkasan",
     "⊟ Data",
     "◈ Visualisasi",
     "↓ Export"]
)

# =========================
# TAB 1: RINGKASAN
# =========================
with tab_summary:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-chart-pie' style='margin-right: 10px;'></i>Ringkasan Hasil Prediksi
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class='kpi-card accent-maroon'>
                <div class='kpi-label'>Total Mahasiswa</div>
                <div class='kpi-value'>{total}</div>
                <div class='kpi-sub'>Data dianalisis</div>
                <i class='fas fa-users kpi-icon'></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='kpi-card accent-maroon'>
                <div class='kpi-label'>Prediksi SAGE</div>
                <div class='kpi-value'>{count_sage}</div>
                <div class='kpi-sub'>{count_sage / total * 100:.1f}% dari total</div>
                <i class='fas fa-code kpi-icon'></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='kpi-card accent-gray'>
                <div class='kpi-label'>Prediksi DELTA</div>
                <div class='kpi-value'>{count_delta}</div>
                <div class='kpi-sub'>{count_delta / total * 100:.1f}% dari total</div>
                <i class='fas fa-database kpi-icon'></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    mayoritas = pred_counts.idxmax() if len(pred_counts) > 0 else "N/A"
    persen_mayoritas = (pred_counts.max() / total * 100) if total > 0 else 0

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-lightbulb'></i> Insight Utama
            </div>
            <div class='insight-text'>
                Berdasarkan analisis machine learning terhadap {total} mahasiswa, 
                sebagian besar (<strong>{persen_mayoritas:.1f}%</strong>) memiliki profil yang sesuai 
                dengan <strong>Laboratorium {mayoritas}</strong>. 
                {"Lab SAGE fokus pada Software Development & Architecture." if mayoritas == "SAGE" else "Lab DELTA fokus pada Data Analytics & Business Intelligence."}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-star'></i> Faktor Paling Berpengaruh
        </div>
        """,
        unsafe_allow_html=True
    )

    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(5)

        st.dataframe(
            fi_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """  
        <div class='insight-box'>  
            <div class='insight-title'>  
                <i class='fas fa-info-circle'></i> Penjelasan 
            </div>  
            <div class='insight-text'>Tabel di atas menunjukkan 5 faktor (mata kuliah atau aktivitas) 
            yang paling mempengaruhi prediksi peminatan laboratorium. Nilai importance yang lebih tinggi 
            berarti faktor tersebut memiliki kontribusi lebih besar dalam keputusan model. 
            </div>  
        </div>  
            """,
            unsafe_allow_html=True
        )

# =========================
# TAB 2: DATA DETAIL
# =========================
with tab_table:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-table'></i> Data Prediksi Lengkap
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        search = st.text_input("Cari NIM atau Nama", "", label_visibility="collapsed",
                               placeholder="Ketik NIM atau nama mahasiswa...")
    with c2:
        lab_filter = st.multiselect(
            "Filter Lab",
            options=["SAGE", "DELTA"],
            default=["SAGE", "DELTA"],
            label_visibility="collapsed"
        )
    with c3:
        min_conf = st.slider("Min Confidence", 0, 100, 0, label_visibility="collapsed")

    df_filter = df_out[df_out["Prediksi Lab"].isin(lab_filter)].copy()

    if search.strip():
        mask = (
                df_filter["NIM"].astype(str).str.contains(search, case=False, na=False) |
                df_filter["Nama Lengkap"].astype(str).str.contains(search, case=False, na=False)
        )
        df_filter = df_filter[mask]

    if df_filter["Confidence"].notna().any():
        df_filter = df_filter[df_filter["Confidence"] >= min_conf]
        df_filter = df_filter.sort_values("Confidence", ascending=False)

    st.dataframe(
        df_filter[["NIM", "Nama Lengkap", "Kelas", "Prediksi Lab", "Confidence", "Probabilitas SAGE",
                   "Probabilitas DELTA"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-user-circle'></i> Profil Mahasiswa Individual
        </div>
        """,
        unsafe_allow_html=True
    )

    choose_nim = st.selectbox(
        "Pilih NIM",
        options=df_out["NIM"].astype(str).tolist(),
        label_visibility="collapsed"
    )

    student = df_out[df_out["NIM"].astype(str) == str(choose_nim)].iloc[0]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Nama:** {student['Nama Lengkap']}")
        st.markdown(f"**Kelas:** {student['Kelas']}")
    with col2:
        st.metric("Prediksi", student['Prediksi Lab'])
    with col3:
        conf_val = student.get('Confidence', np.nan)
        if pd.notna(conf_val):
            st.metric("Confidence", f"{conf_val:.1f}%")

    radar_features = [
        SUBJECT_COLS[0], SUBJECT_COLS[1], SUBJECT_COLS[2],
        SUBJECT_COLS[6], SUBJECT_COLS[9], SUBJECT_COLS[10],
    ]
    vals = [student.get(f, 0) for f in radar_features]
    radar_df = pd.DataFrame({"Feature": radar_features, "Value": vals})

    fig_radar = px.line_polar(
        radar_df,
        r="Value",
        theta="Feature",
        line_close=True,
        markers=True,
    )

    fig_radar.update_traces(
        fill='toself',
        line_color='#6B0F1A'
    )

    fig_radar.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    fig_radar.update_polars(
        bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(
            showline=False,
            gridcolor='lightgray',
        ),
        angularaxis=dict(
            gridcolor='lightgray',
        )
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        """  
    <div class='insight-box'>  
        <div class='insight-title'>  
            <i class='fas fa-info-circle'></i> Penjelasan 
        </div>  
        <div class='insight-text'>Grafik di atas menampilkan profil nilai mahasiswa 
    dalam 6 mata kuliah utama. Semakin jauh titik dari pusat, semakin tinggi nilainya. 
    Pola ini membantu visualisasi kekuatan akademik di berbagai bidang.
        </div>  
    </div>  
        """,
        unsafe_allow_html=True
    )

# =========================
# TAB 3: VISUALISASI - FIXED CONFIDENCE CHART
# =========================
with tab_vis:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-chart-line'></i> Analisis Visual & Insights
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========== KPI CARDS ==========
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class='kpi-card accent-maroon'>
                <div class='kpi-label'>Total Dataset</div>
                <div class='kpi-value'>{total}</div>
                <div class='kpi-sub'>Mahasiswa dianalisis</div>
                <i class='fas fa-calculator kpi-icon'></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='kpi-card accent-maroon'>
                <div class='kpi-label'>Lab SAGE</div>
                <div class='kpi-value'>{count_sage}</div>
                <div class='kpi-sub'>Software Development</div>
                <i class='fas fa-code kpi-icon'></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='kpi-card accent-gray'>
                <div class='kpi-label'>Lab DELTA</div>
                <div class='kpi-value'>{count_delta}</div>
                <div class='kpi-sub'>Data & Analytics</div>
                <i class='fas fa-database kpi-icon'></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== LAYOUT 2 KOLOM: DONUT & PENJELASAN ==========
    col_chart, col_explanation = st.columns([1.3, 1], gap="large")

    # ========== KOLOM KIRI: DONUT CHART ==========
    with col_chart:
        st.markdown(
            """
            <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>
                <i class='fas fa-pie-chart'></i> Distribusi Peminatan Laboratorium
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            pred_counts_safe = pred_counts.reindex(["SAGE", "DELTA"]).fillna(0).astype(int)

            fig_donut = go.Figure(
                data=[go.Pie(
                    labels=["SAGE", "DELTA"],
                    values=pred_counts_safe.values,
                    hole=0.5,
                    marker=dict(colors=["#6B0F1A", "#6B7280"]),
                    textinfo="label+percent+value",
                    hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>",
                )]
            )
            fig_donut.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                font=dict(size=12),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error rendering donut chart: {str(e)}")

    # ========== KOLOM KANAN: PENJELASAN DONUT CHART ==========
    with col_explanation:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """  
        <div class='insight-box'>  
            <div class='insight-title'>  
                <i class='fas fa-info-circle'></i> Penjelasan 
            </div>  
            <div class='insight-text'>
            Visualisasi ini menampilkan proporsi distribusi mahasiswa antara dua laboratorium. Warna maroon merepresentasikan SAGE (Software & Development), sedangkan warna abu-abu merepresentasikan DELTA (Data & Analytics). Ukuran setiap slice menunjukkan persentase mahasiswa yang termasuk dalam setiap laboratorium.
            </div>  
        </div>  
            """,
            unsafe_allow_html=True
        )

    # ========== FEATURE IMPORTANCE SECTION ==========
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """  
        <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>  
            <i class='fas fa-star'></i> Faktor Paling Berpengaruh  
        </div>  
        """,
        unsafe_allow_html=True
    )

    try:
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": model.feature_importances_
            })
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(10)

            fig_fi = go.Figure(
                data=[go.Bar(
                    x=fi_df["Importance"].values,
                    y=fi_df["Feature"].values,
                    orientation="h",
                    marker=dict(
                        color="#6B0F1A",
                        line=dict(color="#4A0A13", width=1),
                        opacity=0.85
                    ),
                    text=fi_df["Importance"].apply(lambda x: f"{x:.4f}"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                )]
            )

            fig_fi.update_layout(
                height=350,
                margin=dict(l=250, r=20, t=20, b=50),
                xaxis_title="Importance Score",
                showlegend=False,
                font=dict(size=11),
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB", zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )

            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown(
                """  
            <div class='insight-box'>  
                <div class='insight-title'>  
                    <i class='fas fa-info-circle'></i> Penjelasan 
                </div>  
                <div class='insight-text'>  
                    Grafik di atas menunjukkan 10 faktor (mata kuliah atau aktivitas) yang paling berpengaruh dalam prediksi peminatan laboratorium. Nilai importance yang lebih tinggi berarti faktor tersebut memiliki kontribusi lebih besar dalam keputusan model. Semakin panjang bar, semakin penting fitur tersebut.  
                </div>  
            </div>  
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ Model tidak memiliki informasi feature importance")

    except Exception as e:
        st.error(f"❌ Error menampilkan feature importance: {str(e)}")

    # ========== BAR CHART: DISTRIBUSI CONFIDENCE LEVEL - FULLY FIXED ==========
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """  
        <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>  
            <i class='fas fa-chart-bar'></i> Distribusi Confidence Level Prediksi  
        </div>  
        """,
        unsafe_allow_html=True
    )

    col_bar_chart, col_bar_explanation = st.columns([1.3, 1], gap="large")

    # ========== KOLOM KIRI: BAR CHART - FULLY FIXED ==========
    with col_bar_chart:
        try:
            # ==================== STEP 1: VALIDASI DATA ====================
            if "Confidence" not in df_out.columns:
                st.error("❌ Kolom Confidence tidak ada!")
                st.write(f"Kolom tersedia: {df_out.columns.tolist()}")
            else:
                confidence_series = df_out["Confidence"].copy()

                # Cek apakah semua NaN
                if confidence_series.isna().all():
                    st.error("❌ SEMUA nilai Confidence adalah NaN!")
                    st.write("Penyebab: Model tidak menghasilkan probabilitas")

                # Cek nilai valid
                valid_confidence = confidence_series.dropna()

                if len(valid_confidence) == 0:
                    st.warning("⚠️ Tidak ada nilai Confidence yang valid!")

                else:
                    # ==================== STEP 2: ANALISIS RANGE ====================
                    min_conf_val = valid_confidence.min()
                    max_conf_val = valid_confidence.max()
                    mean_conf = valid_confidence.mean()

                    st.info(
                        f"✓ Data valid: {len(valid_confidence)} mahasiswa | Min: {min_conf_val:.1f}% | Max: {max_conf_val:.1f}% | Avg: {mean_conf:.1f}%")

                    # ==================== STEP 3: BINNING DENGAN NUMPY ====================
                    bin_edges = [0, 25, 50, 75, 100.1]
                    bin_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]

                    # Gunakan numpy.digitize untuk binning yang akurat
                    bin_indices = np.digitize(valid_confidence.values, bin_edges)
                    bin_indices = np.clip(bin_indices - 1, 0, 3)

                    # Hitung frekuensi setiap bin
                    bin_counts = np.bincount(bin_indices, minlength=4)

                    # ==================== STEP 4: BUAT BAR CHART ====================
                    fig_bar = go.Figure(
                        data=[go.Bar(
                            x=bin_labels,
                            y=bin_counts,
                            marker=dict(
                                color="#6B0F1A",
                                line=dict(color="#4A0A13", width=2),
                                opacity=0.85
                            ),
                            text=bin_counts,
                            textposition="outside",
                            textfont=dict(size=12, color="#6B0F1A"),
                            hovertemplate="<b>%{x}</b><br>Jumlah Mahasiswa: %{y}<extra></extra>",
                            name="Mahasiswa"
                        )]
                    )

                    # ==================== STEP 5: STYLING ====================
                    fig_bar.update_layout(
                        title=None,
                        height=400,
                        margin=dict(l=60, r=20, t=40, b=50),
                        xaxis=dict(
                            title="<b>Confidence Range (%)</b>",
                            titlefont=dict(size=12, color="#111827"),
                            tickfont=dict(size=11, color="#6B7280"),
                            showgrid=False,
                            zeroline=False,
                            showline=True,
                            linewidth=2,
                            linecolor="#E5E7EB"
                        ),
                        yaxis=dict(
                            title="<b>Jumlah Mahasiswa</b>",
                            titlefont=dict(size=12, color="#111827"),
                            tickfont=dict(size=11, color="#6B7280"),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="#E5E7EB",
                            showline=True,
                            linewidth=2,
                            linecolor="#E5E7EB",
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor="#E5E7EB"
                        ),
                        showlegend=False,
                        font=dict(family="Segoe UI, sans-serif", size=11, color="#111827"),
                        plot_bgcolor="rgba(255, 255, 255, 0.5)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        hovermode="x unified",
                        bargap=0.3
                    )

                    st.plotly_chart(fig_bar, use_container_width=True, key="conf_distribution_chart")

                    # ==================== STEP 6: SUMMARY STATS ====================
                    st.markdown("---")
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                    with col_stat1:
                        count_0_25 = (valid_confidence < 25).sum()
                        pct = count_0_25 / len(valid_confidence) * 100 if len(valid_confidence) > 0 else 0
                        st.metric("0-25%", int(count_0_25), f"{pct:.1f}%")

                    with col_stat2:
                        count_25_50 = ((valid_confidence >= 25) & (valid_confidence < 50)).sum()
                        pct = count_25_50 / len(valid_confidence) * 100 if len(valid_confidence) > 0 else 0
                        st.metric("25-50%", int(count_25_50), f"{pct:.1f}%")

                    with col_stat3:
                        count_50_75 = ((valid_confidence >= 50) & (valid_confidence < 75)).sum()
                        pct = count_50_75 / len(valid_confidence) * 100 if len(valid_confidence) > 0 else 0
                        st.metric("50-75%", int(count_50_75), f"{pct:.1f}%")

                    with col_stat4:
                        count_75_100 = (valid_confidence >= 75).sum()
                        pct = count_75_100 / len(valid_confidence) * 100 if len(valid_confidence) > 0 else 0
                        st.metric("75-100%", int(count_75_100), f"{pct:.1f}%")

        except Exception as e:
            st.error(f"❌ Error saat membuat chart: {str(e)}")
            import traceback

            st.write("**Traceback:**")
            st.write(traceback.format_exc())

    # ========== KOLOM KANAN: PENJELASAN ===========
    with col_bar_explanation:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='insight-box'>
                <div class='insight-title'>
                    <i class='fas fa-lightbulb'></i> Penjelasan Confidence Level
                </div>
                <div class='insight-text'>
                    <strong>Confidence Level</strong> menunjukkan seberapa yakin model dalam memprediksi peminatan laboratorium setiap mahasiswa.
                    <br><br>
                    📊 <strong>Interpretasi Range:</strong><br>
                    • <strong>75-100%:</strong> Prediksi sangat akurat & reliabel ✅<br>
                    • <strong>50-75%:</strong> Prediksi cukup baik ⚠️<br>
                    • <strong>25-50%:</strong> Prediksi kurang yakin ⚠️⚠️<br>
                    • <strong>0-25%:</strong> Model tidak yakin ❌<br><br>
                    Mayoritas mahasiswa seharusnya berada di range <strong>75-100%</strong> untuk hasil prediksi yang berkualitas tinggi.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== HISTOGRAM PROBABILITAS SAGE ==========
    st.markdown(
        """  
        <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>  
            <i class='fas fa-chart-area'></i> Analisis Lanjutan  
        </div>  
        """,
        unsafe_allow_html=True
    )

    col_analysis1, col_analysis2 = st.columns(2, gap="large")

    with col_analysis1:
        st.markdown(
            """  
            <div style='font-size: 16px; font-weight: 600; color: #6B0F1A; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'>  
                <i class='fas fa-chart-column'></i> Distribusi Probabilitas SAGE  
            </div>  
            """,
            unsafe_allow_html=True
        )

        try:
            if "Probabilitas SAGE" in df_out.columns:
                prob_sage_data = df_out["Probabilitas SAGE"].dropna()

                if len(prob_sage_data) > 0:
                    fig_hist_sage = go.Figure(
                        data=[go.Histogram(
                            x=prob_sage_data.values,
                            nbinsx=20,
                            marker=dict(
                                color="#6B0F1A",
                                line=dict(color="#4A0A13", width=1),
                                opacity=0.85
                            ),
                            hovertemplate="<b>Probabilitas:</b> %{x:.1f}%<br><b>Frekuensi:</b> %{y}<extra></extra>",
                            name="SAGE"
                        )]
                    )
                    fig_hist_sage.update_layout(
                        height=350,
                        margin=dict(l=60, r=20, t=20, b=50),
                        xaxis_title="Probabilitas SAGE (%)",
                        yaxis_title="Frekuensi",
                        showlegend=False,
                        font=dict(size=11),
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor="#E5E7EB"),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB", showline=True, linewidth=1,
                                   linecolor="#E5E7EB"),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist_sage, use_container_width=True)
                else:
                    st.warning("⚠️ Tidak ada data Probabilitas SAGE")
            else:
                st.warning("⚠️ Kolom Probabilitas SAGE tidak ditemukan")

        except Exception as e:
            st.error(f"❌ Gagal menampilkan histogram SAGE: {str(e)}")

    with col_analysis2:
        st.markdown(
            """  
            <div style='font-size: 16px; font-weight: 600; color: #6B0F1A; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'>  
                <i class='fas fa-chart-column'></i> Distribusi Probabilitas DELTA  
            </div>  
            """,
            unsafe_allow_html=True
        )

        try:
            if "Probabilitas DELTA" in df_out.columns:
                prob_delta_data = df_out["Probabilitas DELTA"].dropna()

                if len(prob_delta_data) > 0:
                    fig_hist_delta = go.Figure(
                        data=[go.Histogram(
                            x=prob_delta_data.values,
                            nbinsx=20,
                            marker=dict(
                                color="#6B7280",
                                line=dict(color="#374151", width=1),
                                opacity=0.85
                            ),
                            hovertemplate="<b>Probabilitas:</b> %{x:.1f}%<br><b>Frekuensi:</b> %{y}<extra></extra>",
                            name="DELTA"
                        )]
                    )
                    fig_hist_delta.update_layout(
                        height=350,
                        margin=dict(l=60, r=20, t=20, b=50),
                        xaxis_title="Probabilitas DELTA (%)",
                        yaxis_title="Frekuensi",
                        showlegend=False,
                        font=dict(size=11),
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor="#E5E7EB"),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB", showline=True, linewidth=1,
                                   linecolor="#E5E7EB"),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist_delta, use_container_width=True)
                else:
                    st.warning("⚠️ Tidak ada data Probabilitas DELTA")
            else:
                st.warning("⚠️ Kolom Probabilitas DELTA tidak ditemukan")

        except Exception as e:
            st.error(f"❌ Gagal menampilkan histogram DELTA: {str(e)}")

# =========================
# TAB 4: EXPORT
# =========================
with tab_export:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-download'></i> Export Data Hasil Prediksi
        </div>
        """,
        unsafe_allow_html=True
    )

    out_cols = ["NIM", "Nama Lengkap", "Kelas", "Prediksi Lab", "Probabilitas SAGE", "Probabilitas DELTA", "Confidence"]
    out_df = df_out[out_cols].copy()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        csv = out_df.to_csv(index=False)
        st.download_button(
            "Unduh sebagai CSV",
            csv,
            "hasil_peminatan.csv",
            "text/csv",
            use_container_width=True,
            key="csv_export"
        )
        st.caption("Format: CSV (kompatibel dengan Excel, Google Sheets, dll)")

    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Data Lengkap", index=False)
            out_df.to_excel(writer, sheet_name="Prediksi", index=False)
        buffer.seek(0)

        st.download_button(
            "Unduh sebagai Excel",
            buffer,
            "hasil_peminatan_full.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="excel_export"
        )
        st.caption("Format: Excel (.xlsx) - 2 sheet (Data Lengkap & Prediksi)")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Panduan Export
            </div>
            <div class='insight-text'>
                <strong>CSV:</strong> Gunakan untuk import ke aplikasi lain atau analisis lebih lanjut di Python/R.<br>
                <strong>Excel:</strong> Gunakan untuk presentasi atau sharing dengan stakeholder yang lebih familiar dengan Excel.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# FOOTER
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style='text-align: center; padding: 20px 0; color: {COLORS['text_muted']}; font-size: 12px;'>
        <i class='fas fa-copyright'></i> {datetime.now().year} Data Driven Decision Version 2.0 <br>
        Data Exploration, Learning & Translational Analytics Lab |
        <span style='color: {COLORS['maroon']};'>
            δ Delta Lab
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
