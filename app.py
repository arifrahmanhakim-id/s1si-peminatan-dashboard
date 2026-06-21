
# =========================
# app_streamlit.py - Version 2.0
# Dashboard Peminatan Laboratorium S1SI
# =========================

"""
APLIKASI PREDIKSI PENEMPATAN LABORATORIUM
==========================================
Deskripsi: Sistem berbasis machine learning untuk memprediksi penempatan
           laboratorium mahasiswa berdasarkan profil akademik dan aktivitas
           
Fitur Utama:
- Prediksi peminatan Lab SAGE (Software Development & Architecture)
- Prediksi peminatan Lab DELTA (Data Analytics & Business Intelligence)
- Visualisasi data dan insights komprehensif
- Ekspor laporan dalam format PDF, Excel, dan CSV
- Dashboard interaktif dengan Streamlit
"""

# =============================================================================
# BAGIAN 1: IMPORT LIBRARIES
# =============================================================================

# ===== 1.1 IMPORT LIBRARIES - MACHINE LEARNING & DATA PROCESSING =====
import pandas as pd              # Manipulasi dan analisis data
import numpy as np               # Komputasi numerik
import pickle                    # Serialisasi objek Python
import joblib                    # Serialisasi scikit-learn models
import warnings                  # Kontrol warning messages

warnings.filterwarnings('ignore')  # Suppress warning messages

# ===== 1.2 IMPORT LIBRARIES - WEB APPLICATION & UI =====
import streamlit as st           # Framework untuk web app

# ===== 1.3 IMPORT LIBRARIES - DATA VISUALIZATION =====
import plotly.express as px      # Visualisasi interaktif (high-level)
import plotly.graph_objects as go  # Visualisasi interaktif (low-level)

# ===== 1.4 IMPORT LIBRARIES - PDF GENERATION & FILE HANDLING =====
import io                        # Operasi I/O dalam memori
from io import BytesIO           # Buffer bytes untuk file operations
import os                        # Operasi sistem file
from datetime import datetime    # Manipulasi tanggal dan waktu

# ===== 1.5 IMPORT LIBRARIES - REPORTLAB (PDF GENERATION) =====
from reportlab.lib.pagesizes import letter, A4  # Ukuran halaman PDF
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Style teks
from reportlab.lib.units import inch  # Unit konversi inch
from reportlab.lib import colors  # Palette warna
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY  # Text alignment
from reportlab.platypus import (  # Elemen PDF
    SimpleDocTemplate,  # Dokumen PDF dasar
    Table,              # Tabel
    TableStyle,         # Style tabel
    Paragraph,          # Paragraf teks
    Spacer,             # Spasi
    PageBreak,          # Pemisah halaman
    Image               # Gambar
)

# ===== 1.6 IMPORT LIBRARIES - SYSTEM & SUBPROCESS =====
import subprocess  # Eksekusi perintah sistem
import sys         # Variabel sistem Python
import platform    # Informasi platform

# =============================================================================
# BAGIAN 2: FUNGSI UTILITY & HELPER
# =============================================================================

# ===== 2.1 FUNGSI EKSTRAKSI TAHUN ANGKATAN =====
def extract_tahun_angkatan(kelas_code):
    """
    Ekstrak tahun angkatan dari kode kelas.
    
    Format kode kelas: S1SI-KJ-23-01 (contoh)
    Bagian index [2] adalah "23" yang diubah menjadi "2023"
    
    Args:
        kelas_code (str): Kode kelas format "S1SI-KJ-YY-XX"
        
    Returns:
        int: Tahun angkatan 4 digit (e.g., 2023) atau None jika parsing gagal
    """
    try:
        parts = kelas_code.split('-')
        if len(parts) >= 3:
            tahun_2digit = parts[2]
            tahun_4digit = int(f"20{tahun_2digit}")
            return tahun_4digit
    except Exception:
        pass
    return None


def get_angkatan_info(df):
    """
    Dapatkan informasi distribusi angkatan dari dataframe.
    
    Fungsi ini menganalisis semua mahasiswa dan mengekstrak tahun angkatan
    mereka untuk menghasilkan statistik angkatan.
    
    Args:
        df (pd.DataFrame): DataFrame dengan kolom "Kelas"
        
    Returns:
        dict: Informasi angkatan berisi:
            - min_angkatan: Tahun angkatan minimum
            - max_angkatan: Tahun angkatan maksimum
            - angkatan_counts: Series dengan distribusi per tahun
            - tahun_list: List semua tahun angkatan
        Returns None jika parsing gagal
    """
    try:
        if "Kelas" in df.columns and len(df) > 0:
            angkatan_list = []
            
            # Iterasi setiap kelas dan ekstrak tahun
            for kelas in df["Kelas"]:
                tahun = extract_tahun_angkatan(str(kelas))
                if tahun:
                    angkatan_list.append(tahun)
            
            # Hitung statistik jika ada data
            if angkatan_list:
                min_angkatan = min(angkatan_list)
                max_angkatan = max(angkatan_list)
                angkatan_counts = pd.Series(angkatan_list).value_counts().sort_index()
                
                return {
                    "min_angkatan": min_angkatan,
                    "max_angkatan": max_angkatan,
                    "angkatan_counts": angkatan_counts,
                    "tahun_list": angkatan_list
                }
    except Exception as e:
        st.warning(f"Error extracting angkatan: {str(e)}")
    
    return None


# ===== 2.2 FUNGSI AUTO-INSTALL KALEIDO & CHROME =====
def ensure_kaleido_and_chrome():
    """
    Pastikan Kaleido dan Chrome terinstall untuk export image.
    
    Kaleido adalah library yang diperlukan Plotly untuk export ke image format.
    Chrome juga diperlukan sebagai rendering engine untuk Plotly export.
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        import kaleido
        print("✓ Kaleido sudah terinstall")
        return True
    except ImportError:
        print("⚠️ Menginstall Kaleido...")
        try:
            # Install kaleido via pip
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "kaleido", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓ Kaleido berhasil diinstall")
            
            # Install Chrome otomatis
            print("⚠️ Menginstall Chrome untuk Kaleido...")
            subprocess.check_call(
                [sys.executable, "-m", "plotly_get_chrome"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓ Chrome berhasil diinstall")
            return True
            
        except Exception as e:
            print(f"⚠️ Error install: {e}")
            return False

# Jalankan saat startup aplikasi
ensure_kaleido_and_chrome()


# ===== 2.3 FUNGSI SAFE IMAGE EXPORT =====
def safe_export_image(fig, format="png", width=700, height=500, scale=2):
    """
    Export Plotly figure ke image dengan error handling lengkap.
    
    Fungsi ini mencoba export figure ke format image (PNG/SVG/JPG).
    Jika gagal, akan return None dan bisa diganti dengan fallback tabel.
    
    Args:
        fig: Plotly figure object
        format (str): Format image ("png", "jpg", "svg")
        width (int): Lebar image dalam pixel
        height (int): Tinggi image dalam pixel
        scale (int): Scale factor untuk quality (1-4)
        
    Returns:
        BytesIO: Buffer image jika berhasil, None jika gagal
    """
    try:
        # Cek ketersediaan kaleido
        try:
            import kaleido
        except ImportError:
            st.warning("⚠️ Kaleido tidak terinstall. Menggunakan fallback tabel.")
            return None
        
        # Export image dari figure
        img_bytes = fig.to_image(format=format, width=width, height=height, scale=scale)
        
        if img_bytes:
            img_buffer = BytesIO(img_bytes)
            img_buffer.seek(0)
            return img_buffer
        else:
            return None
            
    except ImportError as e:
        st.warning(f"⚠️ Library tidak tersedia: {str(e)[:80]}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Gagal export chart ke image: {str(e)[:80]}")
        return None


# ===== 2.4 FUNGSI GENERATE PDF REPORT =====
def generate_pdf_report(df_out, detail_table, overall_avg_conf, sage_students, 
                        delta_students, total, angkatan_info=None):
    """
    Generate professional PDF report dengan fallback untuk semua chart.
    
    Fungsi ini membuat laporan PDF komprehensif berisi:
    - Ringkasan statistik
    - Visualisasi distribusi laboratorium
    - Analisis kualitas prediksi
    - Detail prediksi semua mahasiswa
    - Feature importance analysis
    - Distribusi confidence level
    - Rekomendasi implementasi
    - Kesimpulan dan findings
    
    Args:
        df_out (pd.DataFrame): DataFrame dengan hasil prediksi lengkap
        detail_table (pd.DataFrame): Tabel detail untuk ditampilkan di laporan
        overall_avg_conf (float): Rata-rata confidence keseluruhan
        sage_students (pd.DataFrame): Subset mahasiswa prediksi SAGE
        delta_students (pd.DataFrame): Subset mahasiswa prediksi DELTA
        total (int): Total jumlah mahasiswa
        angkatan_info (dict, optional): Informasi angkatan dari df
        
    Returns:
        BytesIO: Buffer PDF yang siap untuk di-download
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    story = []  # List elemen yang akan ditambahkan ke dokumen
    styles = getSampleStyleSheet()
    
    # ===== CUSTOM PARAGRAPH STYLES =====
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6B0F1A'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#6B0F1A'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        fontName='Helvetica',
        leading=13
    )
    
    small_style = ParagraphStyle(
        'CustomSmall',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        fontName='Helvetica',
        leading=12
    )
    
    # ===== HALAMAN PERTAMA: JUDUL & INFO DOKUMEN =====
    story.append(Paragraph("LAPORAN KOMPREHENSIF", title_style))
    story.append(Paragraph("Prediksi Penempatan Laboratorium", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Informasi dokumen (tanggal, total mahasiswa, angkatan)
    angkatan_text = ""
    if angkatan_info:
        min_ang = angkatan_info.get("min_angkatan", "")
        max_ang = angkatan_info.get("max_angkatan", "")
        if min_ang == max_ang:
            angkatan_text = f"Angkatan: {int(min_ang)}"
        else:
            angkatan_text = f"Angkatan: {int(min_ang)} - {int(max_ang)}"
    
    doc_info_data = [
        ['Tanggal Laporan:', datetime.now().strftime('%d %B %Y')],
        ['Total Mahasiswa:', str(total)],
        [angkatan_text, ''] if angkatan_text else ['', '']
    ]
    
    # Hapus baris kosong
    doc_info_data = [row for row in doc_info_data if row[0]]
    
    doc_info_table = Table(doc_info_data, colWidths=[2*inch, 3*inch])
    doc_info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6B0F1A')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(doc_info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ===== BAGIAN 1: RINGKASAN STATISTIK =====
    story.append(Paragraph("1. RINGKASAN STATISTIK KESELURUHAN", heading_style))
    
    # Hitung persentase
    sage_pct = len(sage_students) / total * 100 if total > 0 else 0
    delta_pct = len(delta_students) / total * 100 if total > 0 else 0
    sage_avg_conf = sage_students["Confidence"].mean() if len(sage_students) > 0 else 0
    delta_avg_conf = delta_students["Confidence"].mean() if len(delta_students) > 0 else 0
    
    # Data untuk tabel ringkasan
    summary_data = [
        ['Metrik', 'Nilai', 'Persentase'],
        ['Total Mahasiswa', str(total), '100%'],
        ['Lab SAGE', str(len(sage_students)), f'{sage_pct:.1f}%'],
        ['Lab DELTA', str(len(delta_students)), f'{delta_pct:.1f}%'],
        ['Avg Confidence SAGE', f'{sage_avg_conf:.2f}%', '-'],
        ['Avg Confidence DELTA', f'{delta_avg_conf:.2f}%', '-'],
        ['Avg Confidence Overall', f'{overall_avg_conf:.2f}%', '-']
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6B0F1A')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Narasi penjelasan ringkasan
    narasi_ringkasan = f"""
    <b>Penjelasan Ringkasan Statistik:</b><br/>
    Dataset ini berisi informasi akademik dan aktivitas dari <b>{total} mahasiswa</b> Program Studi Sistem Informasi. 
    Model machine learning menganalisis profil mahasiswa untuk memprediksi kesesuaian dengan dua laboratorium utama: 
    Lab SAGE (Software Development & Architecture) dan Lab DELTA (Data Analytics & Business Intelligence).<br/>
    <br/>
    <b>Distribusi Prediksi:</b> Lab SAGE menerima <b>{len(sage_students)} mahasiswa ({sage_pct:.1f}%)</b> sementara 
    Lab DELTA menerima <b>{len(delta_students)} mahasiswa ({delta_pct:.1f}%)</b>. Tingkat kepercayaan rata-rata 
    untuk Lab SAGE adalah <b>{sage_avg_conf:.2f}%</b> dan untuk Lab DELTA adalah <b>{delta_avg_conf:.2f}%</b>, 
    dengan confidence keseluruhan mencapai <b>{overall_avg_conf:.2f}%</b>.<br/>
    <br/>
    <b>Implikasi:</b> Prediksi yang dihasilkan berdasarkan analisis mendalam terhadap nilai mata kuliah 
    ({len(SUBJECT_COLS)} mata kuliah) dan keterlibatan mahasiswa dalam berbagai aktivitas ({len(ACTIVITY_COLS)} kategori). 
    Tingkat confidence yang tinggi menunjukkan keandalan rekomendasi penempatan laboratorium.
    """
    
    story.append(Paragraph(narasi_ringkasan, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # PAGE BREAK
    story.append(PageBreak())
    
    # ===== BAGIAN 2: VISUALISASI PIE CHART =====
    story.append(Paragraph("2. VISUALISASI DISTRIBUSI LABORATORIUM", heading_style))
    
    try:
        # Buat pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=[
                f'<b>Lab SAGE</b><br>Software Development<br>({len(sage_students)} mahasiswa)',
                f'<b>Lab DELTA</b><br>Data Analytics<br>({len(delta_students)} mahasiswa)'
            ],
            values=[len(sage_students), len(delta_students)],
            marker=dict(
                colors=['#6B0F1A', '#6B7280'],
                line=dict(color='white', width=2)
            ),
            textinfo='percent+value',
            textfont=dict(size=12, color='white'),
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value} mahasiswa<br>Persentase: %{percent}<extra></extra>',
            pull=[0.05, 0.05]
        )])
        
        fig_pie.update_layout(
            height=500,
            width=700,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=11, family="Arial"),
            showlegend=True,
            legend=dict(
                x=1,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#6B0F1A',
                borderwidth=1,
                font=dict(size=10)
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Gunakan fungsi safe export
        pie_img_buffer = safe_export_image(fig_pie, width=700, height=500)
        
        if pie_img_buffer:
            story.append(Image(pie_img_buffer, width=5.5*inch, height=4*inch))
            story.append(Spacer(1, 0.15*inch))
        else:
            # FALLBACK: Tabel jika export gagal
            fallback_pie_data = [
                ['Laboratorium', 'Jumlah', 'Persentase'],
                ['Lab SAGE', str(len(sage_students)), f'{sage_pct:.1f}%'],
                ['Lab DELTA', str(len(delta_students)), f'{delta_pct:.1f}%']
            ]
            fallback_pie_table = Table(fallback_pie_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            fallback_pie_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6B0F1A')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(fallback_pie_table)
            story.append(Spacer(1, 0.15*inch))
        
    except Exception as e:
        st.warning(f"⚠️ Gagal generate pie chart: {str(e)[:80]}")
        fallback_pie_data = [
            ['Laboratorium', 'Jumlah', 'Persentase'],
            ['Lab SAGE', str(len(sage_students)), f'{sage_pct:.1f}%'],
            ['Lab DELTA', str(len(delta_students)), f'{delta_pct:.1f}%']
        ]
        fallback_pie_table = Table(fallback_pie_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        fallback_pie_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6B0F1A')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
        ]))
        story.append(fallback_pie_table)
        story.append(Spacer(1, 0.15*inch))
    
    # Narasi pie chart
    narasi_pie = f"""
    <b>Penjelasan Pie Chart - Distribusi Laboratorium:</b><br/>
    Visualisasi di atas menampilkan proporsi distribusi mahasiswa antara dua laboratorium. 
    <b>Lab SAGE</b> (Software Development & Architecture) menerima <b>{len(sage_students)} mahasiswa ({sage_pct:.1f}%)</b> 
    yang memiliki profil akademik kuat dalam programming, sistem informasi, dan arsitektur enterprise. 
    <b>Lab DELTA</b> (Data Analytics & Business Intelligence) menerima <b>{len(delta_students)} mahasiswa ({delta_pct:.1f}%)</b> 
    yang menunjukkan kekuatan dalam matematika, statistik, dan analisis data. Distribusi yang seimbang menunjukkan 
    keberagaman minat dan kemampuan mahasiswa di program studi ini.
    """
    
    story.append(Paragraph(narasi_pie, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # PAGE BREAK
    story.append(PageBreak())
    
    # ===== BAGIAN 3: ANALISIS KUALITAS PREDIKSI =====
    story.append(Paragraph("3. ANALISIS KUALITAS PREDIKSI", heading_style))
    
    high_conf_count = len(df_out[df_out["Confidence"] >= 80])
    medium_conf_count = len(df_out[(df_out["Confidence"] >= 60) & (df_out["Confidence"] < 80)])
    low_conf_count = len(df_out[df_out["Confidence"] < 60])
    
    quality_data = [
        ['Kategori', 'Jumlah', '%', 'Deskripsi Kualitas & Tindakan'],
        ['Tinggi (≥80%)', str(high_conf_count), f'{high_conf_count/total*100:.1f}%', 
         'Prediksi sangat akurat dan reliabel. Mahasiswa dapat langsung ditempatkan tanpa konsultasi lebih lanjut.'],
        ['Sedang (60-80%)', str(medium_conf_count), f'{medium_conf_count/total*100:.1f}%', 
         'Prediksi baik dengan confidence cukup. Perlu diskusi dengan academic advisor untuk pertimbangan minat personal.'],
        ['Rendah (<60%)', str(low_conf_count), f'{low_conf_count/total*100:.1f}%', 
         'Model kurang yakin. Perlu assessment mendalam, wawancara tatap muka, dan konsultasi intensif dengan pembimbing.']
    ]
    
    quality_table = Table(quality_data, colWidths=[1.2*inch, 0.9*inch, 0.8*inch, 3*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (2, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6B0F1A')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 1), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('WORDWRAP', (3, 1), (3, -1), True),
    ]))
    
    story.append(quality_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Narasi kualitas prediksi
    narasi_kualitas = f"""
    <b>Penjelasan Analisis Kualitas Prediksi:</b><br/>
    Tabel di atas mengkategorisasi kualitas prediksi berdasarkan confidence score yang dihasilkan oleh model machine learning. 
    Kategori ini penting untuk menentukan tingkat tindak lanjut yang diperlukan.<br/>
    <br/>
    <b>1. Kualitas Tinggi (≥80%) - {high_conf_count} mahasiswa ({high_conf_count/total*100:.1f}%):</b><br/>
    Mahasiswa dalam kategori ini memiliki profil akademik yang sangat jelas sesuai dengan laboratorium yang diprediksi. 
    Model machine learning memberikan rekomendasi dengan tingkat kepercayaan sangat tinggi. Mahasiswa ini dapat langsung 
    ditempatkan di laboratorium prediksi tanpa memerlukan konsultasi tambahan.<br/>
    <br/>
    <b>2. Kualitas Sedang (60-80%) - {medium_conf_count} mahasiswa ({medium_conf_count/total*100:.1f}%):</b><br/>
    Mahasiswa dalam kategori ini memiliki prediksi yang baik namun masih ada beberapa aspek yang perlu dipertimbangkan. 
    Disarankan untuk melakukan diskusi dengan academic advisor untuk mengevaluasi minat personal, pengalaman, dan tujuan karir 
    sebelum penempatan final di laboratorium.<br/>
    <br/>
    <b>3. Kualitas Rendah (<60%) - {low_conf_count} mahasiswa ({low_conf_count/total*100:.1f}%):</b><br/>
    Mahasiswa dalam kategori ini memiliki profil yang ambigu atau tidak jelas kesesuaiannya dengan kedua laboratorium. 
    Model machine learning kurang yakin dengan prediksinya. Sangat disarankan melakukan assessment lebih mendalam melalui 
    wawancara tatap muka, evaluasi portfolio, atau instrumen assessment tambahan sebelum penempatan final.
    """
    
    story.append(Paragraph(narasi_kualitas, small_style))
    story.append(Spacer(1, 0.2*inch))
    
    # PAGE BREAK sebelum detail tabel
    story.append(PageBreak())
    
    # ===== BAGIAN 4: DETAIL PREDIKSI (DENGAN SPLIT PAGES) =====
    story.append(Paragraph("4. DETAIL PREDIKSI SEMUA MAHASISWA", heading_style))
    
    narasi_detail = f"""
    <b>Penjelasan Tabel Detail Prediksi:</b><br/>
    Tabel berikut menampilkan hasil prediksi lengkap untuk semua <b>{len(detail_table)} mahasiswa</b> yang dianalisis. 
    Setiap baris menunjukkan nama mahasiswa, laboratorium yang diprediksi, confidence level, dan probabilitas untuk masing-masing laboratorium. 
    Tabel ini dapat digunakan untuk verifikasi individual, tracking progress, dan follow-up dengan mahasiswa mengenai pilihan laboratorium mereka.
    """
    
    story.append(Paragraph(narasi_detail, small_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Split tabel detail ke multiple pages dengan repeat header
    rows_per_page = 25
    total_rows = len(detail_table)
    
    for page_num in range(0, total_rows, rows_per_page):
        if page_num > 0:
            story.append(PageBreak())
            story.append(Paragraph(f"4. DETAIL PREDIKSI SEMUA MAHASISWA (Lanjutan - Halaman {page_num // rows_per_page + 1})", heading_style))
            story.append(Spacer(1, 0.1*inch))
        
        start_idx = page_num
        end_idx = min(page_num + rows_per_page, total_rows)
        detail_sample = detail_table.iloc[start_idx:end_idx].copy()
        
        detail_data = [['No', 'Nama Mahasiswa', 'Lab', 'Confidence', 'Prob SAGE', 'Prob DELTA']]
        
        for idx_in_page, (_, row) in enumerate(detail_sample.iterrows(), 1):
            actual_no = start_idx + idx_in_page
            try:
                detail_data.append([
                    str(actual_no),
                    str(row.get('Nama Lengkap', row.get('Nama Mahasiswa', '')))[:28],
                    str(row.get('Prediksi Lab', row.get('Lab Prediksi', ''))),
                    str(row.get('Confidence', '')),
                    str(row.get('Probabilitas SAGE', row.get('Prob SAGE', ''))),
                    str(row.get('Probabilitas DELTA', row.get('Prob DELTA', '')))
                ])
            except Exception:
                continue
        
        detail_pred_table = Table(detail_data, colWidths=[0.5*inch, 2*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        detail_pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        story.append(detail_pred_table)
        story.append(Spacer(1, 0.15*inch))
    
    # PAGE BREAK sebelum feature importance
    story.append(PageBreak())
    
    # ===== BAGIAN 5: FEATURE IMPORTANCE =====
    story.append(Paragraph("5. ANALISIS FAKTOR PALING BERPENGARUH", heading_style))
    
    narasi_fi = """
    <b>Penjelasan Analisis Faktor Paling Berpengaruh:</b><br/>
    Model machine learning (Random Forest) menganalisis kontribusi relatif setiap variabel (mata kuliah dan aktivitas) 
    dalam memprediksi peminatan laboratorium. Grafik berikut menampilkan 12 faktor dengan importance score tertinggi. 
    Semakin panjang bar, semakin besar pengaruhnya terhadap keputusan model.
    """
    
    story.append(Paragraph(narasi_fi, small_style))
    story.append(Spacer(1, 0.15*inch))
    
    try:
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True).tail(12)
            
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
                height=400,
                margin=dict(l=280, r=100, t=40, b=60),
                xaxis=dict(
                    title=dict(text="<b>Importance Score</b>", font=dict(size=11, color="#111827")),
                    showgrid=True, gridwidth=1, gridcolor="#E5E7EB", zeroline=False, tickfont=dict(size=10)
                ),
                yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
                showlegend=False,
                font=dict(size=11),
                plot_bgcolor="rgba(255, 255, 255, 1)",
                paper_bgcolor="rgba(255, 255, 255, 1)",
            )
            
            # Gunakan safe export
            fi_img_buffer = safe_export_image(fig_fi, width=900, height=400)
            
            if fi_img_buffer:
                story.append(Image(fi_img_buffer, width=5.5*inch, height=3*inch))
                story.append(Spacer(1, 0.2*inch))
            else:
                # FALLBACK: Tabel
                fallback_fi_data = [['Feature', 'Importance Score']]
                for _, row in fi_df.iterrows():
                    fallback_fi_data.append([
                        str(row['Feature'])[:35],
                        f"{row['Importance']:.4f}"
                    ])
                
                fallback_fi_table = Table(fallback_fi_data, colWidths=[3*inch, 1.5*inch])
                fallback_fi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6B0F1A')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(fallback_fi_table)
                story.append(Spacer(1, 0.2*inch))
        else:
            story.append(Paragraph("⚠️ Model tidak memiliki informasi feature importance", normal_style))
    
    except Exception as e:
        st.warning(f"⚠️ Error feature importance: {str(e)[:80]}")
        story.append(Paragraph("⚠️ Gagal membuat feature importance chart", normal_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # PAGE BREAK sebelum confidence histogram
    story.append(PageBreak())
    
    # ===== BAGIAN 6: CONFIDENCE HISTOGRAM =====
    story.append(Paragraph("6. DISTRIBUSI CONFIDENCE LEVEL", heading_style))
    
    narasi_hist = """
    <b>Penjelasan Histogram Distribusi Confidence:</b><br/>
    Visualisasi ini menampilkan sebaran confidence level untuk kedua laboratorium secara bersamaan. 
    Sumbu horizontal menunjukkan confidence level (0-100%), sedangkan sumbu vertikal menunjukkan jumlah mahasiswa.
    """
    
    story.append(Paragraph(narasi_hist, small_style))
    story.append(Spacer(1, 0.15*inch))
    
    try:
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=sage_students["Confidence"],
            name="SAGE",
            nbinsx=15,
            marker=dict(color="#6B0F1A", opacity=0.7, line=dict(color="#4A0A13", width=1)),
            hovertemplate="<b>SAGE</b><br>Confidence: %{x:.1f}%<br>Frekuensi: %{y}<extra></extra>"
        ))
        
        fig_hist.add_trace(go.Histogram(
            x=delta_students["Confidence"],
            name="DELTA",
            nbinsx=15,
            marker=dict(color="#6B7280", opacity=0.7, line=dict(color="#374151", width=1)),
            hovertemplate="<b>DELTA</b><br>Confidence: %{x:.1f}%<br>Frekuensi: %{y}<extra></extra>"
        ))
        
        fig_hist.update_layout(
            title="Histogram Distribusi Confidence Level per Laboratorium",
            xaxis_title="Confidence Level (%)",
            yaxis_title="Jumlah Mahasiswa",
            barmode="overlay",
            height=400,
            margin=dict(l=70, r=50, t=60, b=60),
            plot_bgcolor="rgba(255, 255, 255, 1)",
            paper_bgcolor="rgba(255, 255, 255, 1)",
            font=dict(size=11),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB", tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB", tickfont=dict(size=10)),
            hovermode="x unified",
            legend=dict(x=0.7, y=1.0, font=dict(size=10))
        )
        
        # Gunakan safe export
        hist_img_buffer = safe_export_image(fig_hist, width=850, height=400)
        
        if hist_img_buffer:
            story.append(Image(hist_img_buffer, width=5.5*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        else:
            # FALLBACK: Tabel confidence breakdown
            conf_0_25 = len(df_out[df_out["Confidence"] < 25])
            conf_25_50 = len(df_out[(df_out["Confidence"] >= 25) & (df_out["Confidence"] < 50)])
            conf_50_75 = len(df_out[(df_out["Confidence"] >= 50) & (df_out["Confidence"] < 75)])
            conf_75_100 = len(df_out[df_out["Confidence"] >= 75])
            
            fallback_hist_data = [
                ['Range Confidence', 'SAGE', 'DELTA', 'Total', 'Persentase'],
                ['0-25%', str(len(sage_students[sage_students["Confidence"] < 25])), 
                 str(len(delta_students[delta_students["Confidence"] < 25])), str(conf_0_25), 
                 f'{conf_0_25/total*100:.1f}%'],
                ['25-50%', str(len(sage_students[(sage_students["Confidence"] >= 25) & (sage_students["Confidence"] < 50)])), 
                 str(len(delta_students[(delta_students["Confidence"] >= 25) & (delta_students["Confidence"] < 50)])), 
                 str(conf_25_50), f'{conf_25_50/total*100:.1f}%'],
                ['50-75%', str(len(sage_students[(sage_students["Confidence"] >= 50) & (sage_students["Confidence"] < 75)])), 
                 str(len(delta_students[(delta_students["Confidence"] >= 50) & (delta_students["Confidence"] < 75)])), 
                 str(conf_50_75), f'{conf_50_75/total*100:.1f}%'],
                ['75-100%', str(len(sage_students[sage_students["Confidence"] >= 75])), 
                 str(len(delta_students[delta_students["Confidence"] >= 75])), str(conf_75_100), 
                 f'{conf_75_100/total*100:.1f}%']
            ]
            
            fallback_hist_table = Table(fallback_hist_data, colWidths=[1.2*inch, 1*inch, 1*inch, 0.9*inch, 1*inch])
            fallback_hist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B0F1A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6B0F1A')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(fallback_hist_table)
            story.append(Spacer(1, 0.2*inch))
    
    except Exception as e:
        st.warning(f"⚠️ Error histogram: {str(e)[:80]}")
        story.append(Paragraph("<b>Confidence Level Breakdown:</b><br/>", small_style))
        conf_stats = f"Confidence rata-rata keseluruhan: {overall_avg_conf:.2f}%"
        story.append(Paragraph(conf_stats, small_style))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(Spacer(1, 0.2*inch))
    
    # PAGE BREAK sebelum rekomendasi
    story.append(PageBreak())
    story.append(Paragraph("7. REKOMENDASI IMPLEMENTASI", heading_style))
    
    narasi_rekomendasi = f"""
    <b>Penjelasan Strategi Implementasi Penempatan:</b><br/>
    Berdasarkan analisis hasil prediksi dan kualitas confidence level, berikut adalah strategi implementasi berjenjang 
    yang direkomendasikan untuk memaksimalkan efektivitas penempatan laboratorium dan kepuasan mahasiswa:<br/>
    """
    
    story.append(Paragraph(narasi_rekomendasi, small_style))
    story.append(Spacer(1, 0.1*inch))
    
    recommendation_text = f"""
    <b>A. Immediate Action (Kualitas Tinggi - {high_conf_count} mahasiswa):</b><br/>
    Mahasiswa dengan confidence ≥80% dapat langsung ditempatkan di laboratorium prediksi tanpa konsultasi lebih lanjut. 
    Tingkat kepercayaan model terhadap prediksi ini sangat tinggi (accuracy ≥80%), sehingga keputusan dapat langsung diimplementasikan.<br/>
    <br/>
    <b>B. Consultation Required (Kualitas Sedang - {medium_conf_count} mahasiswa):</b><br/>
    Mahasiswa dengan confidence 60-80% memiliki prediksi yang baik tetapi memerlukan diskusi dengan academic advisor. 
    Pertimbangan tambahan seperti minat personal, pengalaman praktik, dan career goals perlu dievaluasi sebelum penempatan final.<br/>
    <br/>
    <b>C. Special Review (Kualitas Rendah - {low_conf_count} mahasiswa):</b><br/>
    Mahasiswa dengan confidence <60% memerlukan assessment mendalam dan konsultasi intensif. 
    Pertimbangkan untuk melakukan evaluasi lebih lanjut menggunakan instrumen assessment tambahan, wawancara tatap muka, 
    atau portfolio review sebelum keputusan final dibuat.<br/>
    """
    
    story.append(Paragraph(recommendation_text, small_style))
    story.append(Spacer(1, 0.2*inch))
    
    # PAGE BREAK sebelum kesimpulan
    story.append(PageBreak())
    story.append(Paragraph("8. KESIMPULAN DAN TEMUAN UTAMA", heading_style))
    
    narasi_kesimpulan = f"""
    <b>Ringkasan Analisis Komprehensif:</b><br/>
    Analisis prediksi laboratorium untuk <b>{total} mahasiswa</b> Program Studi Sistem Informasi menunjukkan bahwa 
    model machine learning (Random Forest) memberikan rekomendasi dengan tingkat kepercayaan rata-rata <b>{overall_avg_conf:.2f}%</b>.<br/>
    <br/>
    <b>Temuan Distribusi:</b><br/>
    Lab SAGE (<b>{len(sage_students)} / {sage_pct:.1f}%</b>) dan Lab DELTA (<b>{len(delta_students)} / {delta_pct:.1f}%</b>)<br/>
    <br/>
    <b>Rekomendasi Utama:</b><br/>
    Implementasikan strategi penempatan berjenjang sesuai kualitas prediksi untuk memaksimalkan kepuasan mahasiswa 
    dan outcome pembelajaran yang optimal.
    """
    
    story.append(Paragraph(narasi_kesimpulan, small_style))
    
    # ===== FOOTER =====
    story.append(Spacer(1, 0.4*inch))
    footer_text = f"Laporan ini dihasilkan secara otomatis pada {datetime.now().strftime('%d %B %Y, %H:%M:%S')} | Dashboard v2.0"
    story.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )))
    
    # Build PDF document
    try:
        doc.build(story)
    except Exception as e:
        st.error(f"Error building PDF: {str(e)}")
    
    buffer.seek(0)
    return buffer


# =============================================================================
# BAGIAN 3: KONFIGURASI KONSTANTA & VARIABEL GLOBAL
# =============================================================================

# ===== 3.1 DEFINISI KOLOM FEATURES & LABELS =====
# Mata kuliah yang digunakan sebagai feature untuk prediksi
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

# Aktivitas yang digunakan sebagai feature untuk prediksi
ACTIVITY_COLS = [
    "Asisten Praktikum",
    "MBKM",
    "KP",
    "Lomba",
    "Penelitian",
    "Abdimas",
    "Sertifikasi",
]

# Kombinasi semua feature (mata kuliah + aktivitas)
FEATURE_COLS = SUBJECT_COLS + ACTIVITY_COLS

# Kolom identitas mahasiswa
ID_COLS = ["NIM", "Nama Lengkap", "Kelas"]

# Kolom minat (opsional, untuk referensi)
MINAT_COLS = ["Minat SAGE", "Minat DELTA"]

# Kolom target/label
LABEL_COL = "Target"

# ===== 3.2 PALET WARNA BRAND =====
COLORS = {
    "maroon": "#6B0F1A",          # Warna primary maroon
    "maroon_light": "#8B1325",    # Maroon terang
    "maroon_dark": "#4A0A13",     # Maroon gelap
    "gray": "#6B7280",            # Warna gray netral
    "gray_light": "#9CA3AF",      # Gray terang
    "gray_dark": "#374151",       # Gray gelap
    "neutral": "#F3F4F6",         # Background netral
    "border": "#E5E7EB",          # Warna garis border
    "text_primary": "#111827",    # Teks utama
    "text_muted": "#6B7280",      # Teks secondary
    "white": "#FFFFFF",           # Putih
}

# =============================================================================
# BAGIAN 4: KONFIGURASI STREAMLIT
# =============================================================================

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard Peminatan Lab",    # Judul browser tab
    page_icon="🔥",                          # Icon browser tab
    layout="wide",                           # Layout halaman (wide atau centered)
    initial_sidebar_state="expanded",        # Sidebar awal (expanded atau collapsed)
)

# =============================================================================
# BAGIAN 5: CSS CUSTOM STYLING
# =============================================================================

st.markdown(
    f"""
    <link rel="stylesheet" 
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">

    <style>
    /* ===== GLOBAL STYLES ===== */
    * {{
        font-family: 'Segoe UI', 'Helvetica Neue', -apple-system, sans-serif;
    }}

    .stApp {{
        background-color: {COLORS['neutral']};
    }}

    /* ===== HEADER STYLING ===== */
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

    /* ===== KPI CARD STYLING ===== */
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

    /* ===== CARD STYLING ===== */
    .card {{
        background: {COLORS['white']};
        border-radius: 10px;
        padding: 20px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}

    /* ===== INSIGHT BOX STYLING ===== */
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

    /* ===== DIVIDER STYLING ===== */
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, {COLORS['maroon']} 0%, transparent 100%);
        margin: 28px 0;
        border: none;
    }}

    /* ===== BUTTON STYLING ===== */
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

    /* ===== METRIC CONTAINER STYLING ===== */
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

    /* ===== UTILITY CLASSES ===== */
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

# =============================================================================
# BAGIAN 6: INISIALISASI SESSION STATE
# =============================================================================

# ===== 6.1 INISIALISASI VARIABEL SESSION STATE =====
# Session state digunakan untuk menyimpan status modal dialogs
if "show_reset_dataset_modal" not in st.session_state:
    st.session_state.show_reset_dataset_modal = False

if "show_reset_cache_modal" not in st.session_state:
    st.session_state.show_reset_cache_modal = False

if "show_reset_all_modal" not in st.session_state:
    st.session_state.show_reset_all_modal = False

if "file_loaded_message" not in st.session_state:
    st.session_state.file_loaded_message = False

# =============================================================================
# BAGIAN 7: FUNGSI MODAL MANAGEMENT
# =============================================================================

# ===== 7.1 FUNGSI SHOW MODAL =====
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
    """Tutup semua modal dialogs"""
    st.session_state.show_reset_dataset_modal = False
    st.session_state.show_reset_cache_modal = False
    st.session_state.show_reset_all_modal = False

# ===== 7.2 FUNGSI CONFIRM ACTION =====
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
    """Konfirmasi dan jalankan reset semua (dataset + cache)"""
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

# =============================================================================
# BAGIAN 8: RENDER MODAL DIALOGS
# =============================================================================

# ===== 8.1 MODAL RESET DATASET =====
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

# ===== 8.2 MODAL RESET CACHE =====
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

# ===== 8.3 MODAL RESET ALL =====
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

# =============================================================================
# BAGIAN 9: FUNGSI HELPER UNTUK DATA LOADING
# =============================================================================

# ===== 9.1 LOAD MODEL & SCALER DARI FILE =====
@st.cache_resource
def load_model_scaler_info():
    """
    Load model machine learning, scaler, dan informasi model dari file.
    
    Files yang diperlukan:
    - model_peminatan.pkl: Model Random Forest yang sudah ditraining
    - scaler_peminatan.pkl: StandardScaler untuk normalisasi feature
    - model_info.pkl: Dictionary berisi informasi model (accuracy, algoritma, dll)
    
    Returns:
        tuple: (model, scaler, model_info_dict, error_message)
    """
    try:
        model = joblib.load("model_peminatan.pkl")
        scaler = joblib.load("scaler_peminatan.pkl")
        with open("model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        return model, scaler, model_info, None
    except Exception as e:
        return None, None, {}, str(e)

# ===== 9.2 READ EXCEL FILE DENGAN TYPE HANDLING =====
@st.cache_data
def safe_read_excel(path):
    """
    Baca file Excel dengan type handling otomatis.
    
    Fungsi ini mengkonversi kolom NIM, Nama Lengkap, dan Kelas ke string
    untuk menghindari masalah konversi tipe data.
    
    Args:
        path (str): Path file Excel
        
    Returns:
        pd.DataFrame: DataFrame dari file Excel
    """
    return pd.read_excel(
        path,
        dtype={"NIM": str, "Nama Lengkap": str, "Kelas": str}
    )

# =============================================================================
# BAGIAN 10: LOAD MODEL DAN VALIDASI
# =============================================================================

# Load model, scaler, dan info
model, scaler, model_info, load_error = load_model_scaler_info()
if load_error:
    st.error(f"⚠️ Error loading model: {load_error}")

# =============================================================================
# BAGIAN 11: SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    # ===== 11.1 UPLOAD DATASET SECTION =====
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
            <i class='fas fa-file-excel' style='font-size: 20px; color: #6B0F1A;'></i>
            <div style='font-size: 16px; font-weight: 700; color: #111827;'>Upload Dataset</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # File uploader untuk dataset
    uploaded_file = st.file_uploader(
        "Pilih file Excel (.xlsx)",
        type=["xlsx"],
        label_visibility="collapsed"
    )

    # Simpan file yang di-upload
    if uploaded_file:
        with open("dataset_aktif.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✓ Dataset berhasil disimpan")

    # ===== 11.2 DATASET INFO SECTION =====
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

            # Tampilkan informasi angkatan
            try:
                angkatan_info = get_angkatan_info(df_info)
                
                if angkatan_info:
                    min_angkatan = angkatan_info["min_angkatan"]
                    max_angkatan = angkatan_info["max_angkatan"]
                    
                    if min_angkatan == max_angkatan:
                        st.caption(f"**Angkatan:** {int(min_angkatan)}")
                    else:
                        st.caption(f"**Angkatan:** {int(min_angkatan)} - {int(max_angkatan)}")
                    
                    # Breakdown angkatan
                    angkatan_counts = angkatan_info["angkatan_counts"]
                    if len(angkatan_counts) > 0:
                        st.caption("**Breakdown Angkatan:**")
                        for tahun in sorted(angkatan_counts.index):
                            count = int(angkatan_counts[tahun])
                            st.caption(f"  • {int(tahun)}: {count} mahasiswa")
            except Exception:
                pass
        
        except Exception:
            pass

    # ===== 11.3 TEMPLATE DOWNLOAD SECTION =====
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
        # Buat template data
        template = {
            "NIM": ["311300001"],
            "Nama Lengkap": ["Budi Santoso"],
            "Kelas": ["S1SI-KJ-23-01"],
        }
        
        # Tambah kolom mata kuliah
        for c in SUBJECT_COLS:
            template[c] = [75]
        
        # Tambah kolom aktivitas
        for c in ACTIVITY_COLS:
            template[c] = [1]
        
        # Tambah kolom minat (optional)
        template["Minat SAGE"] = [50]
        template["Minat DELTA"] = [60]
        template["Target"] = ["SAGE"]

        # Generate Excel file
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

    # ===== 11.4 MODEL INFORMATION SECTION =====
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

    # ===== 11.5 RESET & CACHE MANAGEMENT SECTION =====
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px;'>
            <i class='fas fa-sync' style='font-size: 16px; color: #6B0F1A;'></i>
            <div style='font-size: 14px; font-weight: 700; color: #111827;'>Reset & Cache</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tombol reset
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

# =============================================================================
# BAGIAN 12: MAIN HEADER
# =============================================================================

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

# =============================================================================
# BAGIAN 13: VALIDASI DATASET & MODEL
# =============================================================================

# ===== 13.1 CEK DATASET TERSEDIA =====
if not os.path.exists("dataset_aktif.xlsx"):
    st.info("👈 **Langkah pertama:** Upload file Excel di sidebar untuk memulai analisis")
    st.stop()

# ===== 13.2 LOAD DAN VALIDASI DATA =====
try:
    df = safe_read_excel("dataset_aktif.xlsx")
    df["NIM"] = df["NIM"].astype(str).str.strip()

    # Tampilkan notifikasi file berhasil dimuat
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

# ===== 13.3 CEK KOLOM YANG DIPERLUKAN =====
required_for_pred = ID_COLS[:2] + FEATURE_COLS
missing = [c for c in required_for_pred if c not in df.columns]
if missing:
    st.error(f"❌ Kolom yang hilang: {', '.join(missing)}")
    st.stop()

# ===== 13.4 CEK MODEL TERSEDIA =====
if model is None:
    st.error("❌ Model belum tersedia. Jalankan training terlebih dahulu dengan menjalankan train_model.py")
    st.stop()

# =============================================================================
# BAGIAN 14: PREDIKSI & PROCESSING
# =============================================================================

# ===== 14.1 EKSTRAKSI FEATURES & PREDIKSI =====
try:
    # Ekstrak feature dari dataframe
    X = df[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Normalisasi dengan scaler
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"❌ Error saat standardisasi: {e}")
            st.stop()
    else:
        X_scaled = X

    # Dapatkan prediksi dari model
    pred = model.predict(X_scaled)
    pred_labels = pd.Series(pred).map({0: "SAGE", 1: "DELTA"}).astype(str)

    # Dapatkan probabilitas dan confidence score
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

# ===== 14.2 BUAT OUTPUT DATAFRAME =====
df_out = df.copy()
df_out["NIM"] = df_out["NIM"].astype(str)
df_out["Prediksi Lab"] = pred_labels
df_out["Probabilitas SAGE"] = prob_sage
df_out["Probabilitas DELTA"] = prob_delta
df_out["Confidence"] = confidence

# Konversi feature columns ke numeric
for col in FEATURE_COLS:
    if col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

# ===== 14.3 HITUNG SUMMARY METRICS =====
pred_counts = df_out["Prediksi Lab"].value_counts()
total = len(df_out)
count_sage = int((df_out["Prediksi Lab"] == "SAGE").sum())
count_delta = int((df_out["Prediksi Lab"] == "DELTA").sum())

# =============================================================================
# BAGIAN 15: DEFINISI TABS
# =============================================================================

# Buat 6 tabs untuk berbagai analisis
tab_summary, tab_table, tab_vis, tab_comparative, tab_report, tab_profile, tab_rec, tab_analytics, tab_export = st.tabs([
    "⊞ Ringkasan",
    "⊟ Data Tabel",
    "◈ Visualisasi",
    "⇄ Perbandingan",
    "▤ Laporan",
    "◎ Profil Lab",
    "◇ Rekomendasi",
    "◬ Analytics",
    "⇩ Export"
    ])

# =============================================================================
# BAGIAN 16: TAB 1 - RINGKASAN (SUMMARY)
# =============================================================================

with tab_summary:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-chart-pie' style='margin-right: 10px;'></i>Ringkasan Hasil Prediksi
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tampilkan KPI cards
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

    # Main insight
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

    # Feature importance
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
            use_container_width='stretch',
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

# =============================================================================
# BAGIAN 17: TAB 2 - DATA TABLE
# =============================================================================

with tab_table:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-table'></i> Data Prediksi Lengkap
        </div>
        """,
        unsafe_allow_html=True
    )

    # Filter section
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

    # Apply filters
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

    # Display table
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

    # Individual student profile
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

    # Student info
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

    # Radar chart
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

    st.plotly_chart(fig_radar, use_container_width='stretch')

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

# =============================================================================
# BAGIAN 18: TAB 3 - VISUALISASI
# =============================================================================

with tab_vis:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-chart-line'></i> Analisis Visual & Insights
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI cards
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

    # Donut chart
    col_chart, col_explanation = st.columns([1.4, 1], gap="large")

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
            sage_pct = count_sage / total * 100
            delta_pct = count_delta / total * 100

            # Create donut chart
            fig_donut = go.Figure(
                data=[go.Pie(
                    labels=[
                        f"<b>SAGE</b><br>Software Development & Architecture",
                        f"<b>DELTA</b><br>Data Analytics & Business Intelligence"
                    ],
                    values=pred_counts_safe.values,
                    hole=0.4,
                    marker=dict(
                        colors=["#6B0F1A", "#6B7280"],
                        line=dict(color='white', width=3)
                    ),
                    textinfo='value+percent',
                    textfont=dict(size=13, color='white'),
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Jumlah: %{value} mahasiswa<br>Persentase: %{percent}<extra></extra>',
                    pull=[0.08, 0.08]
                )]
            )
            
            fig_donut.update_layout(
                height=500,
                width=700,
                margin=dict(l=50, r=250, t=50, b=50),
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=1,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='#6B0F1A',
                    borderwidth=2,
                    font=dict(size=11),
                    itemsizing='constant'
                ),
                font=dict(size=12, family="Arial"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_donut, use_container_width='stretch')

        except Exception as e:
            st.error(f"❌ Error rendering donut chart: {str(e)}")

    with col_explanation:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""  
            <div class='insight-box'>  
                <div class='insight-title'>  
                    <i class='fas fa-info-circle'></i> Penjelasan Distribusi 
                </div>  
                <div class='insight-text'>
                    <strong>Pie Chart - Distribusi Laboratorium:</strong><br/>
                    Grafik menunjukkan proporsi mahasiswa di kedua laboratorium. SAGE: {count_sage} ({sage_pct:.1f}%), DELTA: {count_delta} ({delta_pct:.1f}%)
                    <br><br>
                    <strong>Warna dan Label:</strong><br/>
                    • <strong>Maroon:</strong> Lab SAGE - Software Development & Architecture<br/>
                    • <strong>Abu-abu:</strong> Lab DELTA - Data Analytics & Business Intelligence<br/>
                    <br/>
                    <strong>Insight:</strong><br/>
                    Distribusi ini mencerminkan keseimbangan minat mahasiswa terhadap kedua bidang.
                </div>  
            </div>  
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Feature importance
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

            st.plotly_chart(fig_fi, use_container_width='stretch')

            st.markdown(
                """  
                <div class='insight-box'>  
                    <div class='insight-title'>  
                        <i class='fas fa-info-circle'></i> Penjelasan 
                    </div>  
                    <div class='insight-text'>  
                        Grafik di atas menunjukkan 10 faktor yang paling berpengaruh dalam prediksi. 
                        Semakin panjang bar, semakin besar kontribusinya dalam keputusan model.  
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

    # Confidence distribution
    st.markdown(
        """  
        <div style='font-size: 18px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;'>  
            <i class='fas fa-chart-bar'></i> Distribusi Confidence Level Prediksi  
        </div>  
        """,
        unsafe_allow_html=True
    )

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

        # Calculate all values
        count_0_25 = (confidence_data < 25).sum()
        pct_0_25 = count_0_25 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        count_25_50 = ((confidence_data >= 25) & (confidence_data < 50)).sum()
        pct_25_50 = count_25_50 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        count_50_75 = ((confidence_data >= 50) & (confidence_data < 75)).sum()
        pct_50_75 = count_50_75 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        count_75_100 = (confidence_data >= 75).sum()
        pct_75_100 = count_75_100 / len(confidence_data) * 100 if len(confidence_data) > 0 else 0

        # Display metrics
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

    # Bar chart
    col_chart, col_explanation = st.columns([1.3, 1], gap="large")

    with col_chart:
        try:
            if "Confidence" not in df_out.columns:
                st.error("Kolom Confidence tidak ada!")
            else:
                confidence_data = df_out["Confidence"].copy()

                # Create bins
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

                st.plotly_chart(fig_bar, use_container_width='stretch')

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
                    <strong>Confidence Level</strong> menunjukkan seberapa yakin model dalam prediksi.
                    <br><br>
                    <strong>Interpretasi Range:</strong><br>
                    <strong>75-100%:</strong> Prediksi sangat akurat<br>
                    <strong>50-75%:</strong> Prediksi cukup baik<br>
                    <strong>25-50%:</strong> Prediksi kurang yakin<br>
                    <strong>0-25%:</strong> Model tidak yakin<br><br>
                    <strong>Semakin tinggi confidence, semakin reliabel prediksi.</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== ANALISIS LANJUTAN: HISTOGRAM PROBABILITAS =====
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
                    st.plotly_chart(fig_hist_sage, use_container_width='stretch')
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
                    st.plotly_chart(fig_hist_delta, use_container_width='stretch')
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
                <strong>Distribusi Probabilitas</strong> menunjukkan sebaran prediksi model untuk setiap mahasiswa.
                <br><br>
                <strong>Grafik Histogram (Kiri & Kanan):</strong><br>
                • <strong>SAGE (kiri):</strong> Konsentrasi menunjukkan skor probabilitas SAGE<br>
                • <strong>DELTA (kanan):</strong> Konsentrasi menunjukkan skor probabilitas DELTA<br><br>
                <strong>Catatan Penting:</strong><br>
                Kedua histogram saling melengkapi karena Prob SAGE + Prob DELTA = 100%
                <br><br>
                <strong>Interpretasi Pola:</strong><br>
                • <strong>Puncak tinggi di 50-100%:</strong> Mayoritas mahasiswa cocok dengan lab<br>
                • <strong>Distribusi merata:</strong> Keberagaman profil mahasiswa tinggi<br>
                • <strong>Puncak di atas 80%:</strong> Prediksi sangat confident
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 19: TAB 4 - ANALISIS KOMPARASI (PERBANDINGAN)
# =============================================================================

with tab_comparative:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-exchange-alt' style='margin-right: 10px;'></i>Analisis Komparasi SAGE vs DELTA
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== PERBANDINGAN PROFIL LABORATORIUM =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-chart-bar'></i> Perbandingan Karakteristik Lab
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Pisahkan mahasiswa berdasarkan prediksi
    sage_students = df_out[df_out["Prediksi Lab"] == "SAGE"]
    delta_students = df_out[df_out["Prediksi Lab"] == "DELTA"]
    
    # Buat tabel perbandingan
    comparison_data = pd.DataFrame({
        "Metrik": [
            "Jumlah Mahasiswa",
            "Confidence Rata-rata",
            "Probabilitas Rata-rata",
            "Rata-rata Nilai Akademik"
        ],
        "SAGE": [
            len(sage_students),
            f"{sage_students['Confidence'].mean():.2f}%",
            f"{sage_students['Probabilitas SAGE'].mean():.2f}%",
            f"{sage_students[SUBJECT_COLS].mean().mean():.2f}"
        ],
        "DELTA": [
            len(delta_students),
            f"{delta_students['Confidence'].mean():.2f}%",
            f"{delta_students['Probabilitas DELTA'].mean():.2f}%",
            f"{delta_students[SUBJECT_COLS].mean().mean():.2f}"
        ]
    })
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    sage_pct = len(sage_students) / total * 100
    delta_pct = len(delta_students) / total * 100

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Penjelasan Tabel Perbandingan
            </div>
            <div class='insight-text'>
                Tabel ini menampilkan perbandingan <strong>4 metrik utama</strong> antara Lab SAGE dan DELTA.
                <br><br>
                <strong>1. Jumlah Mahasiswa:</strong><br>
                Lab SAGE: {len(sage_students)} ({sage_pct:.1f}%), Lab DELTA: {len(delta_students)} ({delta_pct:.1f}%). 
                Menunjukkan distribusi mahasiswa di kedua laboratorium.
                <br><br>
                <strong>2. Confidence Rata-rata:</strong><br>
                Tingkat kepercayaan model untuk setiap lab. Nilai lebih tinggi berarti prediksi lebih akurat.
                <br><br>
                <strong>3. Probabilitas Rata-rata:</strong><br>
                Rata-rata probabilitas prediksi untuk masing-masing lab. Menunjukkan kekuatan profil rata-rata.
                <br><br>
                <strong>4. Rata-rata Nilai Akademik:</strong><br>
                Rata-rata nilai semua mata kuliah per lab. Indikator level akademik keseluruhan di setiap lab.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ===== SCATTER PLOT: CONFIDENCE vs PROBABILITAS =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-scatter'></i> Scatter Plot: Confidence vs Probabilitas
        </div>
        """,
        unsafe_allow_html=True
    )
    
    fig_scatter = go.Figure()
    
    # Tambah trace untuk SAGE
    fig_scatter.add_trace(go.Scatter(
        x=sage_students["Confidence"],
        y=sage_students["Probabilitas SAGE"],
        mode="markers",
        name="SAGE",
        marker=dict(size=8, color="#6B0F1A", opacity=0.6),
        text=sage_students["Nama Lengkap"],
        hovertemplate="<b>%{text}</b><br>Confidence: %{x:.1f}%<br>Prob SAGE: %{y:.1f}%<extra></extra>"
    ))
    
    # Tambah trace untuk DELTA
    fig_scatter.add_trace(go.Scatter(
        x=delta_students["Confidence"],
        y=delta_students["Probabilitas DELTA"],
        mode="markers",
        name="DELTA",
        marker=dict(size=8, color="#6B7280", opacity=0.6),
        text=delta_students["Nama Lengkap"],
        hovertemplate="<b>%{text}</b><br>Confidence: %{x:.1f}%<br>Prob DELTA: %{y:.1f}%<extra></extra>"
    ))
    
    fig_scatter.update_layout(
        title="Scatter Plot: Confidence Level vs Probabilitas Per Lab",
        xaxis_title="Confidence Level (%)",
        yaxis_title="Probabilitas Lab (%)",
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        transition=dict(duration=800, easing="cubic-out"),
        font=dict(size=11),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB")
    )
    
    st.plotly_chart(fig_scatter, use_container_width='stretch')

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-lightbulb'></i> Penjelasan Scatter Plot
            </div>
            <div class='insight-text'>
                Scatter plot ini memvisualisasikan hubungan antara confidence dan probabilitas untuk setiap mahasiswa.
                <br><br>
                <strong>Sumbu X (Horizontal) - Confidence Level (%):</strong><br>
                Tingkat kepercayaan model. Semakin ke kanan, semakin tinggi confidence-nya.
                <br><br>
                <strong>Sumbu Y (Vertikal) - Probabilitas Lab (%):</strong><br>
                Probabilitas prediksi ke laboratorium masing-masing. Semakin ke atas, semakin cocok profil.
                <br><br>
                <strong>Interpretasi Pola Cluster:</strong><br>
                • <strong>Cluster Atas-Kanan:</strong> Prediksi sangat akurat dan profil cocok (IDEAL)<br>
                • <strong>Cluster Atas-Kiri:</strong> Prediksi kurang yakin meski cocok (PERLU REVIEW)<br>
                • <strong>Cluster Bawah-Kanan:</strong> Prediksi akurat tapi profil kurang cocok (BORDERLINE)<br>
                • <strong>Cluster Bawah-Kiri:</strong> Prediksi lemah dan profil tidak cocok (KONSULTASI KHUSUS)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== PERBANDINGAN NILAI MATA KULIAH =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-books'></i> Perbandingan Nilai Mata Kuliah
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== LAB SAGE (ATAS) =====
    st.markdown(
        """
        <div style='font-size: 14px; font-weight: 600; color: #6B0F1A; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'>
            <i class='fas fa-code'></i> Lab SAGE - Top 8 Mata Kuliah
        </div>
        """,
        unsafe_allow_html=True
    )
    
    sage_subjects = sage_students[SUBJECT_COLS].mean().sort_values(ascending=False).head(8)
    
    fig_sage = px.bar(
        x=sage_subjects.values,
        y=sage_subjects.index,
        orientation="h",
        labels={"x": "Nilai Rata-rata", "y": "Mata Kuliah"},
        color_discrete_sequence=["#6B0F1A"]
    )
    
    # Tambahkan label nilai di luar bar
    fig_sage.update_traces(
        text=sage_subjects.values.round(2),
        textposition="outside",
        textfont=dict(size=10, color="#6B0F1A")
    )
    
    fig_sage.update_layout(
        height=380,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=220, r=80, t=40, b=30),
        transition=dict(duration=1000, easing="cubic-in-out"),
        xaxis=dict(
            title="<b>Nilai Rata-rata</b>",
            showgrid=True,
            gridwidth=1,
            gridcolor="#E5E7EB",
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont=dict(size=10)
        )
    )
    
    st.plotly_chart(fig_sage, use_container_width='stretch')

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # ===== LAB DELTA (BAWAH) =====
    st.markdown(
        """
        <div style='font-size: 14px; font-weight: 600; color: #6B7280; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'>
            <i class='fas fa-database'></i> Lab DELTA - Top 8 Mata Kuliah
        </div>
        """,
        unsafe_allow_html=True
    )
    
    delta_subjects = delta_students[SUBJECT_COLS].mean().sort_values(ascending=False).head(8)
    
    fig_delta = px.bar(
        x=delta_subjects.values,
        y=delta_subjects.index,
        orientation="h",
        labels={"x": "Nilai Rata-rata", "y": "Mata Kuliah"},
        color_discrete_sequence=["#6B7280"]
    )
    
    # Tambahkan label nilai di luar bar
    fig_delta.update_traces(
        text=delta_subjects.values.round(2),
        textposition="outside",
        textfont=dict(size=10, color="#6B7280")
    )
    
    fig_delta.update_layout(
        height=380,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=220, r=80, t=40, b=30),
        transition=dict(duration=1000, easing="cubic-in-out"),
        xaxis=dict(
            title="<b>Nilai Rata-rata</b>",
            showgrid=True,
            gridwidth=1,
            gridcolor="#E5E7EB",
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont=dict(size=10)
        )
    )
    
    st.plotly_chart(fig_delta, use_container_width='stretch')

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Penjelasan Perbandingan Mata Kuliah
            </div>
            <div class='insight-text'>
                Visualisasi menampilkan <strong>8 mata kuliah dengan nilai tertinggi</strong> di setiap lab.
                <br><br>
                <strong>Lab SAGE (Atas - Maroon):</strong><br>
                Menunjukkan kekuatan mahasiswa di bidang software development, programming, dan architecture.
                <br><br>
                <strong>Lab DELTA (Bawah - Abu-abu):</strong><br>
                Menunjukkan kekuatan mahasiswa di bidang data analytics, statistics, dan data science.
                <br><br>
                <strong>Insight untuk Pengambilan Keputusan:</strong><br>
                • Perbedaan urutan menunjukkan karakteristik unik setiap lab<br>
                • Mahasiswa dengan nilai tinggi di kedua area fleksibel memilih lab<br>
                • Panjang bar menunjukkan perbedaan relatif antar mata kuliah
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== PERBANDINGAN AKTIVITAS TAMBAHAN =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-tasks'></i> Perbandingan Aktivitas Tambahan
        </div>
        """,
        unsafe_allow_html=True
    )

    sage_activities = sage_students[ACTIVITY_COLS].mean()
    delta_activities = delta_students[ACTIVITY_COLS].mean()

    activity_comparison = pd.DataFrame({
        "Aktivitas": ACTIVITY_COLS,
        "SAGE (Avg)": sage_activities.values,
        "DELTA (Avg)": delta_activities.values
    })

    st.dataframe(activity_comparison, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Penjelasan Perbandingan Aktivitas
            </div>
            <div class='insight-text'>
                Tabel ini membandingkan tingkat keterlibatan mahasiswa dalam berbagai aktivitas tambahan.
                <br><br>
                <strong>7 Kategori Aktivitas:</strong><br>
                • <strong>Asisten Praktikum:</strong> Pengalaman mengajar/membimbing<br>
                • <strong>MBKM:</strong> Program magang atau kerja sambil kuliah<br>
                • <strong>KP:</strong> Kerja praktik/internship formal<br>
                • <strong>Lomba:</strong> Partisipasi dalam kompetisi<br>
                • <strong>Penelitian:</strong> Keterlibatan dalam research project<br>
                • <strong>Abdimas:</strong> Pengabdian masyarakat<br>
                • <strong>Sertifikasi:</strong> Sertifikat profesional yang dimiliki<br>
                <br>
                Perbedaan signifikan menunjukkan preferensi atau komitmen berbeda di setiap lab.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== KEY INSIGHTS & SUMMARY =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-star'></i> Key Insights & Rekomendasi Strategis
        </div>
        """,
        unsafe_allow_html=True
    )

    sage_avg_conf = sage_students["Confidence"].mean()
    delta_avg_conf = delta_students["Confidence"].mean()
    sage_high_conf_pct = (sage_students["Confidence"] >= 80).sum() / len(sage_students) * 100 if len(sage_students) > 0 else 0
    delta_high_conf_pct = (delta_students["Confidence"] >= 80).sum() / len(delta_students) * 100 if len(delta_students) > 0 else 0
    
    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-check-circle'></i> Ringkasan Temuan & Kesimpulan Utama
            </div>
            <div class='insight-text'>
                <strong>1. Distribusi & Keseimbangan:</strong><br>
                Lab SAGE: {len(sage_students)} ({sage_pct:.1f}%), Lab DELTA: {len(delta_students)} ({delta_pct:.1f}%). 
                Menunjukkan keberagaman profil mahasiswa.
                <br><br>
                <strong>2. Akurasi Prediksi per Lab:</strong><br>
                • <strong>Lab SAGE:</strong> Confidence rata-rata {sage_avg_conf:.2f}% dengan {sage_high_conf_pct:.1f}% prediksi berkualitas tinggi<br>
                • <strong>Lab DELTA:</strong> Confidence rata-rata {delta_avg_conf:.2f}% dengan {delta_high_conf_pct:.1f}% prediksi berkualitas tinggi<br>
                <br>
                {'Lab SAGE memiliki prediksi lebih akurat' if sage_avg_conf > delta_avg_conf else 'Lab DELTA memiliki prediksi lebih akurat'}.
                <br><br>
                <strong>3. Karakteristik Unik Setiap Lab:</strong><br>
                • <strong>SAGE:</strong> Fokus engineering, architecture, dan software development<br>
                • <strong>DELTA:</strong> Fokus analytics, statistics, dan data science
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 20: TAB 5 - LAPORAN (REPORT)
# =============================================================================

with tab_report:
    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-file-alt' style='margin-right: 10px;'></i>Laporan Komprehensif Prediksi Lab
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== RINGKASAN STATISTIK =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-chart-pie'></i> Ringkasan Statistik Keseluruhan
        </div>
        """,
        unsafe_allow_html=True
    )

    sage_students = df_out[df_out["Prediksi Lab"] == "SAGE"]
    delta_students = df_out[df_out["Prediksi Lab"] == "DELTA"]
    
    sage_pct = len(sage_students) / total * 100
    delta_pct = len(delta_students) / total * 100
    sage_avg_conf = sage_students["Confidence"].mean()
    delta_avg_conf = delta_students["Confidence"].mean()
    overall_avg_conf = df_out["Confidence"].mean()

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.metric(
            label="Total Mahasiswa",
            value=total,
            delta=None
        )

    with col2:
        st.metric(
            label="Lab SAGE",
            value=len(sage_students),
            delta=f"{sage_pct:.1f}%"
        )

    with col3:
        st.metric(
            label="Lab DELTA",
            value=len(delta_students),
            delta=f"{delta_pct:.1f}%"
        )

    with col4:
        st.metric(
            label="Avg Confidence",
            value=f"{overall_avg_conf:.2f}%",
            delta=None
        )

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Penjelasan Ringkasan Statistik
            </div>
            <div class='insight-text'>
                <strong>Total Mahasiswa:</strong> {total} orang dianalisis dalam sistem prediksi ini.
                <br><br>
                <strong>Distribusi Lab SAGE:</strong> {len(sage_students)} mahasiswa ({sage_pct:.1f}%) diprediksi masuk Lab SAGE.
                <br><br>
                <strong>Distribusi Lab DELTA:</strong> {len(delta_students)} mahasiswa ({delta_pct:.1f}%) diprediksi masuk Lab DELTA.
                <br><br>
                <strong>Confidence Rata-rata Keseluruhan:</strong> {overall_avg_conf:.2f}%. 
                Menunjukkan tingkat akurasi keseluruhan prediksi model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== DISTRIBUSI CONFIDENCE =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-chart-area'></i> Distribusi Confidence Level
        </div>
        """,
        unsafe_allow_html=True
    )

    fig_conf_dist = go.Figure()

    fig_conf_dist.add_trace(go.Histogram(
        x=sage_students["Confidence"],
        name="SAGE",
        nbinsx=20,
        marker_color="#6B0F1A",
        opacity=0.7
    ))

    fig_conf_dist.add_trace(go.Histogram(
        x=delta_students["Confidence"],
        name="DELTA",
        nbinsx=20,
        marker_color="#6B7280",
        opacity=0.7
    ))

    fig_conf_dist.update_layout(
        title="Distribusi Confidence Level per Laboratorium",
        xaxis_title="Confidence Level (%)",
        yaxis_title="Jumlah Mahasiswa",
        barmode="overlay",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        transition=dict(duration=800, easing="cubic-out"),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#E5E7EB")
    )

    st.plotly_chart(fig_conf_dist, use_container_width='stretch')

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-lightbulb'></i> Penjelasan Distribusi Confidence
            </div>
            <div class='insight-text'>
                Histogram menunjukkan sebaran confidence level untuk kedua laboratorium.
                <br><br>
                <strong>Puncak Distribusi (Peak):</strong><br>
                Batang tertinggi menunjukkan range confidence yang paling sering terjadi.
                <br><br>
                <strong>Perbandingan SAGE vs DELTA:</strong><br>
                • Tinggi puncak menunjukkan konsentrasi data<br>
                • Lebar distribusi menunjukkan variabilitas prediksi<br>
                • Overlap menunjukkan kemiripan profil di kedua lab
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== TABEL DETAIL PREDIKSI =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-table'></i> Tabel Detail Prediksi Semua Mahasiswa
        </div>
        """,
        unsafe_allow_html=True
    )

    detail_table = df_out[[
        "Nama Lengkap",
        "Prediksi Lab",
        "Confidence",
        "Probabilitas SAGE",
        "Probabilitas DELTA"
    ]].copy()

    detail_table["Confidence"] = detail_table["Confidence"].apply(lambda x: f"{x:.2f}%")
    detail_table["Probabilitas SAGE"] = detail_table["Probabilitas SAGE"].apply(lambda x: f"{x:.2f}%")
    detail_table["Probabilitas DELTA"] = detail_table["Probabilitas DELTA"].apply(lambda x: f"{x:.2f}%")
    detail_table.columns = ["Nama Mahasiswa", "Lab Prediksi", "Confidence", "Prob SAGE", "Prob DELTA"]

    st.dataframe(detail_table, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Penjelasan Tabel Detail Prediksi
            </div>
            <div class='insight-text'>
                Tabel ini menampilkan detail prediksi untuk setiap mahasiswa dengan informasi lengkap.
                <br><br>
                <strong>Kolom:</strong><br>
                • <strong>Nama Mahasiswa:</strong> Identitas unik untuk tracking<br>
                • <strong>Lab Prediksi:</strong> Rekomendasi laboratorium (SAGE atau DELTA)<br>
                • <strong>Confidence:</strong> Tingkat kepercayaan model (0-100%)<br>
                • <strong>Prob SAGE/DELTA:</strong> Probabilitas untuk masing-masing lab<br>
                <br>
                <strong>Catatan Penting:</strong><br>
                • Selisih Prob SAGE dan Prob DELTA menunjukkan kejelasan keputusan<br>
                • Selisih kecil (<10%) = borderline case yang perlu pertimbangan khusus<br>
                • Data dapat digunakan untuk follow-up dengan mahasiswa
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== KESIMPULAN & REKOMENDASI =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-flag-checkered'></i> Kesimpulan & Rekomendasi Akhir
        </div>
        """,
        unsafe_allow_html=True
    )

    high_conf_count = len(df_out[df_out["Confidence"] >= 80])
    medium_conf_count = len(df_out[(df_out["Confidence"] >= 60) & (df_out["Confidence"] < 80)])
    low_conf_count = len(df_out[df_out["Confidence"] < 60])

    st.markdown(
        f"""
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-check-circle'></i> Kesimpulan Analisis Komprehensif
            </div>
            <div class='insight-text'>
                <strong>Hasil Prediksi Keseluruhan:</strong><br>
                Dari {total} mahasiswa, model memberikan prediksi dengan akurasi rata-rata {overall_avg_conf:.2f}%.
                <br><br>
                <strong>Kualitas Prediksi:</strong><br>
                • <strong>Tinggi (≥80%):</strong> {high_conf_count} mahasiswa ({high_conf_count/total*100:.1f}%) - Dapat langsung diimplementasikan<br>
                • <strong>Sedang (60-80%):</strong> {medium_conf_count} mahasiswa ({medium_conf_count/total*100:.1f}%) - Perlu pertimbangan minor<br>
                • <strong>Rendah (<60%):</strong> {low_conf_count} mahasiswa ({low_conf_count/total*100:.1f}%) - Perlu konsultasi mendalam<br>
                <br>
                <strong>Rekomendasi Implementasi:</strong><br>
                ✓ <strong>Immediate Action:</strong> Mahasiswa berkualitas tinggi langsung ditempatkan<br>
                ✓ <strong>Consultation Required:</strong> Mahasiswa berkualitas sedang perlu diskusi dengan advisor<br>
                ✓ <strong>Special Review:</strong> Mahasiswa berkualitas rendah perlu assessment ulang<br>
                ✓ <strong>Monitoring:</strong> Pantau performa setelah penempatan untuk validasi<br>
                ✓ <strong>Continuous Improvement:</strong> Gunakan feedback untuk improve model
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== DOWNLOAD PDF REPORT =====
    st.markdown(
        """
        <div style='font-size: 16px; font-weight: 700; color: #6B0F1A; margin-bottom: 16px;'>
            <i class='fas fa-file-pdf'></i> Unduh Laporan PDF
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate PDF
    pdf_buffer = generate_pdf_report(
        df_out=df_out,
        detail_table=detail_table,
        overall_avg_conf=overall_avg_conf,
        sage_students=sage_students,
        delta_students=delta_students,
        total=total
    )

    col_pdf1, col_pdf2 = st.columns(2, gap="large")

    with col_pdf1:
        st.download_button(
            label="↓ Unduh PDF Lengkap",
            data=pdf_buffer.getvalue(),
            file_name="laporan_prediksi_lab.pdf",
            mime="application/pdf",
            help="Download laporan lengkap dalam format PDF profesional",
            use_container_width=True
        )

    with col_pdf2:
        st.markdown(
        """
        <div style="
            background-color:#e8f4ff;
            padding:12px;
            border-radius:8px;
            font-size:14px;
            border-left:5px solid #1f77b4;">
            Format PDF mencakup: Ringkasan, Kualitas, Detail, Rekomendasi, dan Kesimpulan.
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-info-circle'></i> Fitur Laporan PDF
            </div>
            <div class='insight-text'>
                <strong>Konten Laporan:</strong><br/>
                Ringkasan statistik, analisis kualitas prediksi, tabel detail sampel, rekomendasi implementasi, 
                kesimpulan strategis, dan footer dengan timestamp otomatis.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 21: TAB PROFIL LAB
# =============================================================================

with tab_profile:

    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-users' style='margin-right: 10px;'></i>Profil Laboratorium
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    """
        <div style="font-size:20px;font-weight:700;margin-bottom:10px;">
            <i class='fas fa-robot' style='margin-right:10px;'></i>
            COMING SOON
        </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background-color:#e8f4ff;
            padding:12px;
            border-radius:8px;
            font-size:12px;
            border-left:5px solid #1f77b4;">
            Halaman Profil Laboratorium sedang dalam tahap pengembangan. Fitur ini akan tersedia pada versi berikutnya.<br>
            Fitur ini akan tersedia pada versi berikutnya. 
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 22: TAB REKOMENDASI
# =============================================================================

with tab_rec:

    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-lightbulb' style='margin-right: 10px;'></i>
            Sistem Rekomendasi
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size:20px;font-weight:700;margin-bottom:10px;">
            <i class='fas fa-robot' style='margin-right:10px;'></i>
            COMING SOON
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background-color:#e8f4ff;
            padding:12px;
            border-radius:8px;
            font-size:12px;
            border-left:5px solid #1f77b4;">
            Smart Recommendation Engine berbasis machine learning yang menyesuaikan hasil prediksi dan berkembang menjadi AI-based system.<br>
            Fitur ini akan tersedia pada versi berikutnya.
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 23: TAB ANALYTICS
# =============================================================================

with tab_analytics:

    st.markdown(
        """
        <div style='font-size: 20px; font-weight: 700; color: #6B0F1A; margin-bottom: 20px;'>
            <i class='fas fa-chart-line' style='margin-right: 10px;'></i>
            Analytics Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size:20px;font-weight:700;margin-bottom:10px;">
            <i class='fas fa-robot' style='margin-right:10px;'></i>
            COMING SOON
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background-color:#e8f4ff;
            padding:12px;
            border-radius:8px;
            font-size:12px;
            border-left:5px solid #1f77b4;">
            Menampilkan analisis data, visualisasi, dan insight dari model machine learning.<br>
            Fitur ini akan tersedia pada versi berikutnya.
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 24: TAB 6 - EXPORT DATA
# =============================================================================

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

    # ===== DOWNLOAD BUTTONS - 3 KOLOM =====
    col1, col2, col3 = st.columns(3, gap="large")

    # ===== EXPORT CSV =====
    with col1:
        st.markdown(
            """
            <div style='text-align: center; padding: 15px; background-color: #F3F4F6; border-radius: 8px; border-left: 4px;'>
                <div style='font-size: 14px; font-weight: 700; color: #6B0F1A; margin-bottom: 10px;'>
                    <i class='fas fa-file-csv'></i> Format CSV
                </div>
                <div style='font-size: 12px; color: #6B7280; margin-bottom: 15px;'>
                    Universal dan kompatibel dengan semua aplikasi
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        csv = out_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "↓ Unduh CSV",
            csv,
            "hasil_peminatan.csv",
            "text/csv",
            use_container_width=True,
            key="csv_export"
        )
        st.caption("✓ Format: CSV (kompatibel dengan Excel, Google Sheets, dll)")

    # ===== EXPORT EXCEL =====
    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 15px; background-color: #F3F4F6; border-radius: 8px; border-left: 4px ;'>
                <div style='font-size: 14px; font-weight: 700; color: #6B0F1A; margin-bottom: 10px;'>
                    <i class='fas fa-file-excel'></i> Format Excel
                </div>
                <div style='font-size: 12px; color: #6B7280; margin-bottom: 15px;'>
                    Dengan 2 sheet untuk kemudahan navigasi
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Data Lengkap", index=False)
            out_df.to_excel(writer, sheet_name="Prediksi", index=False)
        buffer.seek(0)

        st.download_button(
            "↓ Unduh Excel",
            buffer,
            "hasil_peminatan_full.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="excel_export"
        )
        st.caption("✓ Format: Excel (.xlsx) - 2 sheet (Data Lengkap & Prediksi)")

    # ===== EXPORT PDF =====
    with col3:
        st.markdown(
            """
            <div style='text-align: center; padding: 15px; background-color: #F3F4F6; border-radius: 8px; border-left: 4px;'>
                <div style='font-size: 14px; font-weight: 700; color: #6B0F1A; margin-bottom: 10px;'>
                    <i class='fas fa-file-pdf'></i> Format PDF
                </div>
                <div style='font-size: 12px; color: #6B7280; margin-bottom: 15px;'>
                    Laporan profesional siap cetak
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Generate PDF Report
        sage_students = df_out[df_out["Prediksi Lab"] == "SAGE"]
        delta_students = df_out[df_out["Prediksi Lab"] == "DELTA"]
        overall_avg_conf = df_out["Confidence"].mean()

        detail_table = out_df.copy()
        detail_table["Probabilitas SAGE"] = detail_table["Probabilitas SAGE"].apply(lambda x: f"{x:.2f}%")
        detail_table["Probabilitas DELTA"] = detail_table["Probabilitas DELTA"].apply(lambda x: f"{x:.2f}%")
        detail_table["Confidence"] = detail_table["Confidence"].apply(lambda x: f"{x:.2f}%")

        pdf_buffer = generate_pdf_report(
            df_out=df_out,
            detail_table=detail_table,
            overall_avg_conf=overall_avg_conf,
            sage_students=sage_students,
            delta_students=delta_students,
            total=total
        )

        st.download_button(
            "↓ Unduh PDF",
            pdf_buffer.getvalue(),
            "laporan_prediksi_peminatan.pdf",
            "application/pdf",
            use_container_width=True,
            key="pdf_export"
        )
        st.caption("✓ Format: PDF - Laporan lengkap profesional")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ===== EXPORT GUIDE =====
    st.markdown(
        """
        <div class='insight-box'>
            <div class='insight-title'>
                <i class='fas fa-book'></i> Panduan Penggunaan Export
            </div>
            <div class='insight-text'>
                <strong>CSV (Comma Separated Values):</strong><br>
                Gunakan untuk import ke aplikasi lain atau analisis lebih lanjut di Python/R.
                <br><br>
                <strong>Excel (.xlsx):</strong><br>
                Gunakan untuk presentasi dengan Excel. File memiliki 2 sheet untuk navigasi mudah.
                <br><br>
                <strong>PDF (Portable Document Format):</strong><br>
                Gunakan untuk laporan formal, presentasi, atau archiving. Siap untuk printing.
                <br><br>
                <strong>Kolom yang di-export:</strong><br>
                • <strong>NIM:</strong> Nomor identitas mahasiswa<br>
                • <strong>Nama Lengkap:</strong> Nama lengkap mahasiswa<br>
                • <strong>Kelas:</strong> Kelas/angkatan mahasiswa<br>
                • <strong>Prediksi Lab:</strong> Hasil prediksi (SAGE atau DELTA)<br>
                • <strong>Probabilitas SAGE:</strong> Persentase prediksi SAGE (0-100%)<br>
                • <strong>Probabilitas DELTA:</strong> Persentase prediksi DELTA (0-100%)<br>
                • <strong>Confidence:</strong> Tingkat kepercayaan model (0-100%)<br>
                <br>
                <strong>Tips Penggunaan:</strong><br>
                • Filter data berkualitas tinggi (Confidence > 80%)<br>
                • Bandingkan Prob SAGE & DELTA untuk melihat margin<br>
                • Ekspor berkala untuk tracking progress<br>
                • Gunakan PDF untuk dokumentasi resmi
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# BAGIAN 25: FOOTER
# =============================================================================

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

# =====================================================================
# END OF CODE
# =====================================================================
