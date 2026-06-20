# =========================
# app_streamlit.py - Version 2.0
# Dashboard Peminatan Laboratorium S1SI
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
# 1. CONSTANTS & COLUMNS
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
# 2. COLOR PALETTE
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
# 3. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Peminatan Lab",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# 4. CSS STYLING
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

    /* ===== MODAL DIALOG STYLES ===== */
    .modal-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }}

    .modal-content {{
        background: white;
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 500px;
        width: 90%;
        animation: slideDown 0.3s ease-out;
    }}

    @keyframes slideDown {{
        from {{
            opacity: 0;
            transform: translateY(-50px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .modal-title {{
        font-size: 20px;
        font-weight: 700;
        color: {COLORS['maroon']};
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .modal-message {{
        font-size: 14px;
        color: {COLORS['text_muted']};
        line-height: 1.6;
        margin-bottom: 24px;
    }}

    .modal-warning {{
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #EF4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 24px;
        font-size: 13px;
        color: #7F1D1D;
    }}

    .modal-buttons {{
        display: flex;
        gap: 12px;
        justify-content: flex-end;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 5. INITIALIZE SESSION STATE
# =========================
if "show_reset_dataset_modal" not in st.session_state:
    st.session_state.show_reset_dataset_modal = False

if "show_reset_cache_modal" not in st.session_state:
    st.session_state.show_reset_cache_modal = False

if "show_reset_all_modal" not in st.session_state:
    st.session_state.show_reset_all_modal = False

if "file_loaded_message" not in st.session_state:
    st.session_state.file_loaded_message = False

# =========================
# 6. MODAL TRIGGER FUNCTIONS
# =========================
def show_modal_reset_dataset():
    """Tampilkan modal konfirmasi reset dataset"""
    st.session_state.show_reset_dataset_modal = True

def show_modal_reset_cache():
    """Tampilkan modal konfirmasi reset cache"""
    st.session_state.show_reset_cache_modal = True

def show_modal_reset_all():
    """Tampilkan modal konfirmasi reset semua"""
    st.session_state.show_reset_all_modal = True

def close_modal():
    """Tutup semua modal"""
    st.session_state.show_reset_dataset_modal = False
    st.session_state.show_reset_cache_modal = False
    st.session_state.show_reset_all_modal = False

# =========================
# 7. MODAL ACTION FUNCTIONS
# =========================
def confirm_reset_dataset():
    """Konfirmasi dan jalankan reset dataset"""
    try:
        if os.path.exists("dataset_aktif.xlsx"):
            os.remove("dataset_aktif.xlsx")
        st.cache_data.clear()
        st.session_state.show_reset_dataset_modal = False
        st.success("✓ Dataset berhasil direset!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal reset dataset: {str(e)}")

def confirm_reset_cache():
    """Konfirmasi dan jalankan reset cache"""
    try:
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.show_reset_cache_modal = False
        st.success("✓ Cache berhasil direset!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal reset cache: {str(e)}")

def confirm_reset_all():
    """Konfirmasi dan jalankan reset semua"""
    try:
        if os.path.exists("dataset_aktif.xlsx"):
            os.remove("dataset_aktif.xlsx")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.show_reset_all_modal = False
        st.success("✓ Semua berhasil direset!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal reset semua: {str(e)}")


# =========================
# 8. RENDER MODAL DIALOGS (BEFORE SIDEBAR)
# =========================

# Modal Reset Dataset
if st.session_state.show_reset_dataset_modal:
    st.markdown(
        """
        <div class='modal-overlay'>
            <div class='modal-content'>
                <div class='modal-title'>
                    <i class='fas fa-exclamation-circle'></i> Hapus Dataset?
                </div>
                <div class='modal-message'>
                    Anda akan menghapus file <strong>dataset_aktif.xlsx</strong> yang sudah diupload.
                    <br><br>
                    Setelah ini, Anda harus upload file baru untuk melanjutkan analisis.
                </div>
                <div class='modal-warning'>
                    <i class='fas fa-info-circle'></i> <strong>PERINGATAN:</strong> Tindakan ini tidak dapat dibatalkan!
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("❌ Batal", use_container_width=True, key="cancel_reset_dataset"):
            close_modal()
            st.rerun()
    with col2:
        if st.button("✓ Hapus Dataset", use_container_width=True, key="confirm_reset_dataset"):
            confirm_reset_dataset()

# Modal Reset Cache
if st.session_state.show_reset_cache_modal:
    st.markdown(
        """
        <div class='modal-overlay'>
            <div class='modal-content'>
                <div class='modal-title'>
                    <i class='fas fa-exclamation-circle'></i> Reset Cache?
                </div>
                <div class='modal-message'>
                    Cache model dan data akan direset. Aplikasi akan memuat ulang data dari awal.
                    <br><br>
                    Ini berguna jika aplikasi terasa lambat atau ada bug yang perlu diatasi.
                </div>
                <div class='modal-warning'>
                    <i class='fas fa-info-circle'></i> Proses ini mungkin memakan waktu lebih lama saat load ulang.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("❌ Batal", use_container_width=True, key="cancel_reset_cache"):
            close_modal()
            st.rerun()
    with col2:
        if st.button("✓ Reset Cache", use_container_width=True, key="confirm_reset_cache"):
            confirm_reset_cache()

# Modal Reset All
if st.session_state.show_reset_all_modal:
    st.markdown(
        """
        <div class='modal-overlay'>
            <div class='modal-content'>
                <div class='modal-title'>
                    <i class='fas fa-exclamation-triangle'></i> Reset Semua Data?
                </div>
                <div class='modal-message'>
                    Anda akan menghapus:
                    <br>• File <strong>dataset_aktif.xlsx</strong>
                    <br>• Cache model dan data
                    <br><br>
                    Aplikasi akan kembali ke state awal (startup).
                </div>
                <div class='modal-warning'>
                    <i class='fas fa-exclamation-circle'></i> <strong>PERINGATAN PENTING:</strong> Tindakan ini tidak dapat dibatalkan!
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("❌ Batal", use_container_width=True, key="cancel_reset_all"):
            close_modal()
            st.rerun()
    with col2:
        if st.button("✓ Reset Semua", use_container_width=True, key="confirm_reset_all"):
            confirm_reset_all()


# =========================
# 9. HELPER FUNCTIONS
# =========================
@st.cache_resource
def load_model_scaler_info():
    """Load model, scaler, dan model info menggunakan joblib."""
    try:
        model = joblib.load("model_peminatan.pkl")
        scaler = joblib.load("scaler_peminatan.pkl")
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
# 10. LOAD MODEL & SCALER
# =========================
model, scaler, model_info, load_error = load_model_scaler_info()
if load_error:
    st.error(f"⚠️ Error loading model: {load_error}")

# =========================
# 11. SIDEBAR
# =========================
with st.sidebar:
    # ===== UPLOAD DATASET SECTION =====
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

    # ===== DATASET INFO SECTION =====
    if os.path.exists("dataset_aktif.xlsx"):
        try:
            df_info = safe_read_excel("dataset_aktif.xlsx")
            st.markdown(
                """
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px;'>
                    <i class='fas fa-folder-open' style='font-size: 16px; color: #6B0F1A;'></i>
                    <div style='font-size: 14px; font-weight: 700; color: #111827;'>Dataset Information</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"**Rows:** {len(df_info)}")
            st.caption(f"**Columns:** {len(df_info.columns)}")
            st.caption(f"**File:** dataset_aktif.xlsx")

            st.divider()
        except Exception:
            pass

    # ===== TEMPLATE DOWNLOAD SECTION =====
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

    # ===== MODEL INFORMATION SECTION =====
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

    # ===== RESET & CACHE SECTION =====
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px;'>
            <i class='fas fa-sync' style='font-size: 16px; color: #6B0F1A;'></i>
            <div style='font-size: 14px; font-weight: 700; color: #111827;'>Reset & Cache</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_reset1, col_reset2 = st.columns(2, gap="small")

    with col_reset1:
        if st.button("Reset Cache", use_container_width=True, key="btn_reset_cache"):
            show_modal_reset_cache()

    with col_reset2:
        if st.button("Reset Dataset", use_container_width=True, key="btn_reset_dataset"):
            show_modal_reset_dataset()

    if st.button("Reset Semua", use_container_width=True, key="btn_reset_all"):
        show_modal_reset_all()

    st.markdown(
        """
        <div style='font-size: 11px; color: #9CA3AF; margin-top: 12px; text-align: center;'>
            <i class='fas fa-lock'></i> Konfirmasi akan diminta sebelum reset
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 12. MAIN HEADER
# =========================
st.markdown(
    """
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
# 13. REQUIRE DATASET VALIDATION
# =========================
if not os.path.exists("dataset_aktif.xlsx"):
    st.info("👈 **Langkah pertama:** Upload file Excel di sidebar untuk memulai analisis")
    st.stop()

try:
    df = safe_read_excel("dataset_aktif.xlsx")
    df["NIM"] = df["NIM"].astype(str).str.strip()

    if not st.session_state.file_loaded_message:
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
            if st.button("✕", key="close_message"):
                st.session_state.file_loaded_message = True
                st.rerun()

except Exception as e:
    st.error(f"❌ Gagal membaca file: {e}")
    st.stop()

# ===== CHECK REQUIRED COLUMNS =====
required_for_pred = ID_COLS[:2] + FEATURE_COLS
missing = [c for c in required_for_pred if c not in df.columns]
if missing:
    st.error(f"❌ Kolom yang hilang: {', '.join(missing)}")
    st.stop()

# ===== CHECK MODEL AVAILABILITY =====
if model is None:
    st.error("❌ Model belum tersedia. Jalankan training terlebih dahulu dengan menjalankan train_model.py")
    st.stop()


# =========================
# 14. PREDICTION
# =========================
try:
    X = df[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"❌ Error saat standardisasi: {e}")
            st.stop()
    else:
        X_scaled = X

    # Get predictions
    pred = model.predict(X_scaled)
    pred_labels = pd.Series(pred).map({0: "SAGE", 1: "DELTA"}).astype(str)

    # Get probabilities untuk confidence score
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_scaled)
            prob_sage = probs[:, 0] * 100
            prob_delta = probs[:, 1] * 100
            confidence = np.max(probs, axis=1) * 100
        except Exception as e:
            st.warning(f"⚠️ Error predict_proba: {e}")
            confidence = np.full(len(X), 75.0)
            prob_sage = np.full(len(X), 50.0)
            prob_delta = np.full(len(X), 50.0)
    else:
        st.warning("⚠️ Model tidak mendukung predict_proba()")
        confidence = np.full(len(X), 75.0)
        prob_sage = np.full(len(X), 50.0)
        prob_delta = np.full(len(X), 50.0)

except Exception as e:
    st.error(f"❌ Error saat prediksi: {e}")
    st.stop()

# =========================
# 15. CREATE OUTPUT DATAFRAME
# =========================
df_out = df.copy()
df_out["NIM"] = df_out["NIM"].astype(str)
df_out["Prediksi Lab"] = pred_labels
df_out["Probabilitas SAGE"] = prob_sage
df_out["Probabilitas DELTA"] = prob_delta
df_out["Confidence"] = confidence

# Convert feature columns to numeric
for col in FEATURE_COLS:
    if col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

# =========================
# 16. SUMMARY METRICS
# =========================
pred_counts = df_out["Prediksi Lab"].value_counts()
total = len(df_out)
count_sage = int((df_out["Prediksi Lab"] == "SAGE").sum())
count_delta = int((df_out["Prediksi Lab"] == "DELTA").sum())


# =========================
# 17. CREATE TABS
# =========================
tab_summary, tab_table, tab_vis, tab_export = st.tabs(
    ["⊞ Ringkasan", "⊟ Data", "◈ Visualisasi", "↓ Export"]
)

# =========================
# 18. TAB 1: RINGKASAN (SUMMARY)
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

    # ===== KPI CARDS =====
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

    # ===== MAIN INSIGHT =====
    mayoritas = pred_counts.idxmax() if len(pred_counts) > 0 else "N/A"
    persen_mayoritas = (pred_counts.max() / total * 100) if total > 0 else 0

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-lightbulb'></i> Insight Utama
            </div>
            <div class='insight-text'>
                Berdasarkan analisis machine learning terhadap <strong>{total} mahasiswa</strong>, 
                sebagian besar (<strong>{persen_mayoritas:.1f}%</strong>) memiliki profil yang sesuai 
                dengan <strong>Laboratorium {mayoritas}</strong>. 
                {"Lab SAGE fokus pada Software Development & Architecture." if mayoritas == "SAGE" else "Lab DELTA fokus pada Data Analytics & Business Intelligence."}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== FEATURE IMPORTANCE =====
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
                <div class='insight-text'>
                    Tabel di atas menunjukkan 5 faktor (mata kuliah atau aktivitas) yang paling mempengaruhi 
                    prediksi peminatan laboratorium. Nilai importance yang lebih tinggi berarti faktor tersebut 
                    memiliki kontribusi lebih besar dalam keputusan model.
                </div>  
            </div>  
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Model tidak memiliki informasi feature importance")

# =========================
# 19. TAB 2: DATA TABLE
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

    # ===== FILTERS =====
    c1, c2, c3 = st.columns([3, 2, 2])

    with c1:
        search = st.text_input(
            "Cari NIM atau Nama",
            "",
            label_visibility="collapsed",
            placeholder="Ketik NIM atau nama mahasiswa..."
        )

    with c2:
        lab_filter = st.multiselect(
            "Filter Lab",
            options=["SAGE", "DELTA"],
            default=["SAGE", "DELTA"],
            label_visibility="collapsed"
        )

    with c3:
        min_conf = st.slider(
            "Min Confidence",
            0, 100, 0,
            label_visibility="collapsed"
        )

    # ===== APPLY FILTERS =====
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

    # ===== DISPLAY TABLE =====
    st.dataframe(
        df_filter[[
            "NIM",
            "Nama Lengkap",
            "Kelas",
            "Prediksi Lab",
            "Confidence",
            "Probabilitas SAGE",
            "Probabilitas DELTA"
        ]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== INDIVIDUAL STUDENT PROFILE =====
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

    # ===== STUDENT INFO & METRICS =====
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

    # ===== RADAR CHART =====
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
            <div class='insight-text'>
                Grafik radar di atas menampilkan profil nilai mahasiswa dalam 6 mata kuliah utama. 
                Semakin jauh titik dari pusat, semakin tinggi nilainya. Pola ini membantu visualisasi 
                kekuatan akademik di berbagai bidang.
            </div>  
        </div>  
        """,
        unsafe_allow_html=True
    )

    # =========================
# 20. TAB 3: VISUALISASI
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

    # ===== KPI CARDS =====
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

    # ===== DONUT CHART =====
    col_chart, col_explanation = st.columns([1.3, 1], gap="large")

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

    with col_explanation:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """  
            <div class='insight-box'>  
                <div class='insight-title'>  
                    <i class='fas fa-info-circle'></i> Penjelasan 
                </div>  
                <div class='insight-text'>
                    Visualisasi ini menampilkan proporsi distribusi mahasiswa antara dua laboratorium. 
                    Warna maroon merepresentasikan SAGE (Software & Development), sedangkan warna abu-abu 
                    merepresentasikan DELTA (Data & Analytics). Ukuran setiap slice menunjukkan persentase 
                    mahasiswa yang termasuk dalam setiap laboratorium.
                </div>  
            </div>  
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== FEATURE IMPORTANCE BAR CHART =====
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
                xaxis=dict(
                    title=dict(
                        text="<b>Importance Score</b>",
                        font=dict(size=12, color="#111827")
                    ),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#E5E7EB",
                    zeroline=False
                ),
                yaxis=dict(showgrid=False, zeroline=False),
                showlegend=False,
                font=dict(size=11),
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
            )

            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown(
                """  
                <div class='insight-box'>  
                    <div class='insight-title'>  
                        <i class='fas fa-info-circle'></i> Penjelasan 
                    </div>  
                    <div class='insight-text'>  
                        Grafik di atas menunjukkan 10 faktor (mata kuliah atau aktivitas) yang paling berpengaruh 
                        dalam prediksi peminatan laboratorium. Nilai importance yang lebih tinggi berarti faktor 
                        tersebut memiliki kontribusi lebih besar dalam keputusan model.  
                    </div>  
                </div>  
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ Model tidak memiliki informasi feature importance")

    except Exception as e:
        st.error(f"❌ Error menampilkan feature importance: {str(e)}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== CONFIDENCE LEVEL DISTRIBUTION =====
    st.markdown(
        """  
        <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>  
            <i class='fas fa-chart-bar'></i> Distribusi Confidence Level Prediksi  
        </div>  
        """,
        unsafe_allow_html=True
    )

    # ===== BREAKDOWN METRICS =====
    try:
        confidence_data = df_out["Confidence"].copy()

        st.markdown(
            """
            <div style='font-size: 12px; font-weight: 700; color: #6B7280; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.8px;'>
                Breakdown Per Range
            </div>
            """,
            unsafe_allow_html=True
        )

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4, gap="large")

        # Calculate all values first
        count_0_25 = (confidence_data < 25).sum()
        pct_0_25 = count_0_25 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        count_25_50 = ((confidence_data >= 25) & (confidence_data < 50)).sum()
        pct_25_50 = count_25_50 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        count_50_75 = ((confidence_data >= 50) & (confidence_data < 75)).sum()
        pct_50_75 = count_50_75 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        count_75_100 = (confidence_data >= 75).sum()
        pct_75_100 = count_75_100 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        # ===== METRIC 1: 0-25% (RED) =====
        with col_stat1:
            st.markdown(
                f"""
                <div style='border-radius: 8px; padding: 16px; text-align: center; background: white; 
                            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-left: 4px solid #6B0F1A;'>
                    <div style='font-size: 12px; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>0-25%</div>
                    <div style='font-size: 28px; color: #6B0F1A; font-weight: 700;'>{int(count_0_25)}</div>
                    <div style='font-size: 11px; color: #9CA3AF;'>{pct_0_25:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ===== METRIC 2: 25-50% (ORANGE) =====
        with col_stat2:
            st.markdown(
                f"""
                <div style='border-radius: 8px; padding: 16px; text-align: center; background: white; 
                            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-left: 4px solid #6B0F1A;'>
                    <div style='font-size: 12px; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>25-50%</div>
                    <div style='font-size: 28px; color: #6B0F1A; font-weight: 700;'>{int(count_25_50)}</div>
                    <div style='font-size: 11px; color: #9CA3AF;'>{pct_25_50:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ===== METRIC 3: 50-75% (ORANGE) =====
        with col_stat3:
            st.markdown(
                f"""
                <div style='border-radius: 8px; padding: 16px; text-align: center; background: white; 
                            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-left: 4px solid #6B0F1A;'>
                    <div style='font-size: 12px; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>50-75%</div>
                    <div style='font-size: 28px; color: #6B0F1A; font-weight: 700;'>{int(count_50_75)}</div>
                    <div style='font-size: 11px; color: #9CA3AF;'>{pct_50_75:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ===== METRIC 4: 75-100% (GREEN) =====
        with col_stat4:
            st.markdown(
                f"""
                <div style='border-radius: 8px; padding: 16px; text-align: center; background: white; 
                            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-left: 4px solid #6B0F1A;'>
                    <div style='font-size: 12px; color: #6B7280; font-weight: 600; margin-bottom: 8px;'>75-100%</div>
                    <div style='font-size: 28px; color: #6B0F1A; font-weight: 700;'>{int(count_75_100)}</div>
                    <div style='font-size: 11px; color: #9CA3AF;'>{pct_75_100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error metrics: {str(e)}")

    # ===== CONFIDENCE BAR CHART =====
    col_chart, col_explanation = st.columns([1.3, 1], gap="large")

    with col_chart:
        try:
            if "Confidence" not in df_out.columns:
                st.error("Kolom Confidence tidak ada!")
            else:
                confidence_data = df_out["Confidence"].copy()

                bin_edges = np.array([0, 25, 50, 75, 100.1])
                bin_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]

                bin_indices = np.digitize(confidence_data.values, bin_edges)
                bin_indices = np.clip(bin_indices - 1, 0, 3)

                bin_counts = np.bincount(bin_indices, minlength=4).astype(int)

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

                fig_bar.update_layout(
                    title=dict(text="", x=0.5, xanchor="center"),
                    height=400,
                    margin=dict(l=60, r=20, t=20, b=50),
                    xaxis=dict(
                        title=dict(
                            text="<b>Confidence Range (%)</b>",
                            font=dict(size=12, color="#111827")
                        ),
                        tickfont=dict(size=11, color="#6B7280"),
                        showgrid=False,
                        zeroline=False,
                        showline=True,
                        linewidth=2,
                        linecolor="#E5E7EB",
                        type="category"
                    ),
                    yaxis=dict(
                        title=dict(
                            text="<b>Jumlah Mahasiswa</b>",
                            font=dict(size=12, color="#111827")
                        ),
                        tickfont=dict(size=11, color="#6B7280"),
                        autorange=True,
                        rangemode="tozero",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="#E5E7EB",
                        showline=True,
                        linewidth=2,
                        linecolor="#E5E7EB",
                        zeroline=False,
                    ),
                    showlegend=False,
                    font=dict(family="Segoe UI, sans-serif", size=11, color="#111827"),
                    plot_bgcolor="rgba(0, 0, 0, 0)",
                    paper_bgcolor="rgba(0, 0, 0, 0)",
                    hovermode="x unified",
                    bargap=0.3
                )

                st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Error saat membuat chart: {str(e)}")

    with col_explanation:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='insight-box'>
                <div class='insight-title'>
                    <i class='fas fa-lightbulb'></i> Penjelasan Confidence Level
                </div>
                <div class='insight-text'>
                    <strong>Confidence Level</strong> menunjukkan seberapa yakin model dalam memprediksi 
                    peminatan laboratorium setiap mahasiswa.
                    <br><br>
                    <strong>Interpretasi Range:</strong><br>
                    <strong>75-100%:</strong> Prediksi sangat akurat dan reliabel<br>
                    <strong>50-75%:</strong> Prediksi cukup baik<br>
                    <strong>25-50%:</strong> Prediksi kurang yakin<br>
                    <strong>0-25%:</strong> Model tidak yakin<br><br>
                    <strong>Hasil:</strong><br>
                    Mayoritas mahasiswa berada di range 75-100% untuk hasil prediksi berkualitas tinggi.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== ADVANCED ANALYSIS SECTION =====
    st.markdown(
        """  
        <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>  
            <i class='fas fa-chart-area'></i> Analisis Lanjutan  
        </div>  
        """,
        unsafe_allow_html=True
    )

    col_analysis1, col_analysis2 = st.columns(2, gap="large")

    # ===== HISTOGRAM PROBABILITAS SAGE =====
    with col_analysis1:
        st.markdown(
            """  
            <div style='font-size: 12px; font-weight: 700; color: #6B7280; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.8px;'>
                Distribusi Probabilitas SAGE
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
                        title=dict(text="", x=0.5, xanchor="center"),
                        height=350,
                        margin=dict(l=60, r=20, t=20, b=50),
                        xaxis=dict(
                            title=dict(
                                text="<b>Probabilitas SAGE (%)</b>",
                                font=dict(size=12, color="#111827")
                            ),
                            showgrid=False,
                            zeroline=False,
                            showline=True,
                            linewidth=1,
                            linecolor="#E5E7EB"
                        ),
                        yaxis=dict(
                            title=dict(
                                text="<b>Frekuensi</b>",
                                font=dict(size=12, color="#111827")
                            ),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="#E5E7EB",
                            showline=True,
                            linewidth=1,
                            linecolor="#E5E7EB"
                        ),
                        showlegend=False,
                        font=dict(size=11),
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist_sage, use_container_width=True)
                else:
                    st.warning("⚠️ Tidak ada data Probabilitas SAGE")
            else:
                st.warning("⚠️ Kolom Probabilitas SAGE tidak ditemukan")

        except Exception as e:
            st.error(f"❌ Gagal menampilkan histogram SAGE: {str(e)}")

    # ===== HISTOGRAM PROBABILITAS DELTA =====
    with col_analysis2:
        st.markdown(
            """  
            <div style='font-size: 12px; font-weight: 700; color: #6B7280; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.8px;'>
                Distribusi Probabilitas DELTA
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
                        title=dict(text="", x=0.5, xanchor="center"),
                        height=350,
                        margin=dict(l=60, r=20, t=20, b=50),
                        xaxis=dict(
                            title=dict(
                                text="<b>Probabilitas DELTA (%)</b>",
                                font=dict(size=12, color="#111827")
                            ),
                            showgrid=False,
                            zeroline=False,
                            showline=True,
                            linewidth=1,
                            linecolor="#E5E7EB"
                        ),
                        yaxis=dict(
                            title=dict(
                                text="<b>Frekuensi</b>",
                                font=dict(size=12, color="#111827")
                            ),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="#E5E7EB",
                            showline=True,
                            linewidth=1,
                            linecolor="#E5E7EB"
                        ),
                        showlegend=False,
                        font=dict(size=11),
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist_delta, use_container_width=True)
                else:
                    st.warning("⚠️ Tidak ada data Probabilitas DELTA")
            else:
                st.warning("⚠️ Kolom Probabilitas DELTA tidak ditemukan")

        except Exception as e:
            st.error(f"❌ Gagal menampilkan histogram DELTA: {str(e)}")

    # ===== HISTOGRAM EXPLANATION =====
    st.markdown(
        """
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-lightbulb'></i> Penjelasan Distribusi Probabilitas
            </div>
            <div class='insight-text'>
                <strong>Distribusi Probabilitas</strong> menunjukkan sebaran prediksi model dalam memprediksi 
                peminatan laboratorium setiap mahasiswa.
                <br><br>
                <strong>Grafik Histogram (Kiri & Kanan):</strong><br>
                • <strong>SAGE (kiri):</strong> Konsentrasi di sisi kanan berarti banyak mahasiswa cocok untuk SAGE<br>
                • <strong>DELTA (kanan):</strong> Konsentrasi di sisi kanan berarti banyak mahasiswa cocok untuk DELTA<br><br>
                <strong>Catatan Penting:</strong><br>
                Kedua histogram saling melengkapi karena Probabilitas SAGE + DELTA = 100%
                <br><br>
                <strong>Interpretasi Pola:</strong><br>
                • <strong>Bimodal (dua puncak):</strong> Dua kelompok mahasiswa yang jelas berbeda<br>
                • <strong>Unimodal (satu puncak):</strong> Populasi yang lebih homogen<br>
                • <strong>Spread (tersebar):</strong> Keberagaman profil mahasiswa
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# 21. TAB 4: EXPORT
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

    # ===== PREPARE OUTPUT DATAFRAME =====
    out_cols = [
        "NIM",
        "Nama Lengkap",
        "Kelas",
        "Prediksi Lab",
        "Probabilitas SAGE",
        "Probabilitas DELTA",
        "Confidence"
    ]
    out_df = df_out[out_cols].copy()

    # ===== DOWNLOAD BUTTONS =====
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
        st.caption("✓ Format: CSV (kompatibel dengan Excel, Google Sheets, dll)")

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
        st.caption("✓ Format: Excel (.xlsx) - 2 sheet (Data Lengkap & Prediksi)")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== EXPORT GUIDE =====
    st.markdown(
        """
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-download'></i> Panduan Export
            </div>
            <div class='insight-text'>
                <strong>CSV (Comma Separated Values):</strong>
                Gunakan untuk import ke aplikasi lain atau analisis lebih lanjut di Python/R. 
                Format ini universal dan dapat dibuka di berbagai aplikasi spreadsheet.
                <br><br>
                <strong>Excel (.xlsx):</strong>
                Gunakan untuk presentasi atau sharing dengan stakeholder yang lebih familiar dengan Excel.
                File ini memiliki 2 sheet terpisah untuk kemudahan navigasi.
                <br><br>
                <strong>Kolom yang di-export:</strong><br>
                &bull; <strong>NIM:</strong> Nomor identitas mahasiswa<br>
                &bull; <strong>Nama Lengkap:</strong> Nama lengkap mahasiswa<br>
                &bull; <strong>Kelas:</strong> Kelas/angkatan mahasiswa<br>
                &bull; <strong>Prediksi Lab:</strong> Hasil prediksi (SAGE atau DELTA)<br>
                &bull; <strong>Probabilitas SAGE:</strong> Persentase prediksi ke Lab SAGE (0-100%)<br>
                &bull; <strong>Probabilitas DELTA:</strong> Persentase prediksi ke Lab DELTA (0-100%)<br>
                &bull; <strong>Confidence:</strong> Tingkat kepercayaan model terhadap prediksi (0-100%)<br><br>
                <strong>Tips Penggunaan:</strong><br>
                &bull; Gunakan Confidence Score untuk filter data berkualitas tinggi (> 80%)<br>
                &bull; Bandingkan Probabilitas SAGE & DELTA untuk melihat margin prediksi<br>
                &bull; Ekspor secara berkala untuk tracking progress mahasiswa
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# 22. FOOTER
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style='text-align: center; padding: 20px 0; color: {COLORS['text_muted']}; font-size: 12px;'>
        <i class='fas fa-copyright' style='color: {COLORS['maroon']};'></i>
        <span style='color: {COLORS['maroon']};'>{datetime.now().year}</span>
        <span style='color: {COLORS['maroon']};'>Data Exploration, Learning & Translational Analytics Lab | </span>
        <span style='color: {COLORS['maroon']};'>δ Delta Lab</span>
        <br>
        <span style='font-size: 10px; color: #9CA3AF; margin-top: 8px;'>
            Dashboard v2.0 | Last Updated: {datetime.now().strftime('%d %B %Y')}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

