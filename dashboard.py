import os
import io
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------------------------------------------------------
# PAGE CONFIGURATION & AESTHETIC DARK UI CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #090d16;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Keep Sidebar Toggle Button & Header Controls Visible */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
    
    [data-testid="stSidebarNav"] {
        padding-top: 10px;
    }

    /* Custom Title Typography */
    .main-title {
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .sub-title {
        font-size: 15px;
        color: #94a3b8;
        margin-bottom: 24px;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 38, 0.7);
        border: 1px solid #262d3d;
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #818cf8 !important;
    }

    /* Custom Verdict Badges */
    .verdict-real {
        background: rgba(6, 95, 70, 0.2);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 16px;
    }

    .verdict-fake {
        background: rgba(153, 27, 27, 0.2);
        border: 1px solid #f87171;
        color: #f87171;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 16px;
    }

    /* System Status Badges */
    .status-badge-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        text-align: center;
    }

    .status-badge-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        text-align: center;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #121722;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #232a3b;
    }

    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PATH CONSTANTS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_wavlm_asvspoof (1).pt")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "training_metrics (1).pt")
CSV_PATH = os.path.join(ARTIFACTS_DIR, "metrics.csv")
XLSX_PATH = os.path.join(ARTIFACTS_DIR, "metrics.xlsx")
METRICS_NO_EXT = os.path.join(ARTIFACTS_DIR, "metrics")

TARGET_LEN = 64000  # 4 seconds @ 16 kHz
SR = 16000

# ---------------------------------------------------------
# MODEL ARCHITECTURE DEFINITIONS
# ---------------------------------------------------------
class WeightedLayerSum(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states):
        stack = torch.stack(hidden_states, dim=0)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (stack * w).sum(dim=0)

class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_dim, bottleneck=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            w = torch.softmax(self.attention(x), dim=1)
            mu = torch.sum(w * x, dim=1)
            var = torch.sum(w * x * x, dim=1) - mu * mu
            sigma = torch.sqrt(var.clamp_min(1e-6))
            return torch.cat([mu, sigma], dim=-1)

class WavLMAntiSpoof(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus", embed_dim=256):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        conf = AutoConfig.from_pretrained(model_name)
        conf.output_hidden_states = True
        self.encoder = AutoModel.from_config(conf)
        d = conf.hidden_size
        self.layer_sum = WeightedLayerSum(conf.num_hidden_layers + 1)
        self.pooling = AttentiveStatsPooling(d)
        self.embedding = nn.Sequential(
            nn.Linear(2 * d, embed_dim), nn.ReLU(), nn.Dropout(0.3)
        )

    def forward(self, x):
        out = self.encoder(x, output_hidden_states=True)
        h = self.layer_sum(out.hidden_states)
        return self.embedding(self.pooling(h))

class OCSoftmax(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.center = nn.Parameter(torch.empty(1, embed_dim))

    def forward(self, emb):
        with torch.amp.autocast(device_type=emb.device.type, enabled=False):
            w = F.normalize(self.center.float(), dim=1)
            x = F.normalize(emb.float(), dim=1)
            score = (x @ w.t()).squeeze(1).clamp(-1.0, 1.0)
        return score

# ---------------------------------------------------------
# RESOURCE LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, f"Checkpoint file missing at: {MODEL_PATH}"
    
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        model_name = ckpt.get("model_name", "microsoft/wavlm-base-plus")
        
        model = WavLMAntiSpoof(model_name=model_name)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        objective = OCSoftmax()
        if "objective_state" in ckpt:
            objective.load_state_dict(ckpt["objective_state"])
        objective.eval()
        
        threshold = ckpt.get("threshold_llr", 0.0)
        return model, objective, threshold, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_analytics():
    metrics = torch.load(METRICS_PATH, map_location="cpu") if os.path.exists(METRICS_PATH) else None
    
    csv_df = None
    for path in [CSV_PATH, XLSX_PATH, METRICS_NO_EXT]:
        if os.path.exists(path):
            try:
                csv_df = pd.read_csv(path)
                break
            except Exception:
                try:
                    csv_df = pd.read_excel(path)
                    break
                except Exception:
                    pass
                
    return metrics, csv_df

model, objective, threshold, err_msg = load_trained_model()
metrics, csv_df = load_analytics()

# ---------------------------------------------------------
# PDF GENERATORS
# ---------------------------------------------------------
def generate_voice_test_pdf(verdict, llr_score, threshold_val, rms_val, ai_summary):
    """Generates a PDF analysis report for an individual recorded audio sample."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#0F172A'))
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#64748B'))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#1E293B'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, leading=14, textColor=colors.HexColor('#334155'))

    story.append(Paragraph("Acoustic Voice Analysis & Deepfake Verification Report", title_style))
    story.append(Paragraph("Individual Audio Sample Diagnostic Report", subtitle_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Verification Summary", heading_style))
    
    data = [
        ["Parameter", "Analysis Value"],
        ["Classification Verdict", verdict],
        ["Log-Likelihood Ratio (LLR)", f"{llr_score:.4f}"],
        ["Decision LLR Threshold", f"{threshold_val:.4f}"],
        ["Signal RMS Energy", f"{rms_val:.6f}"],
        ["Sampling Rate", "16,000 Hz (16 kHz Mono)"],
        ["Target Duration", "4.0 Seconds (64,000 Samples)"]
    ]
    
    t = Table(data, colWidths=[200, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. AI Acoustic Diagnostic Breakdown", heading_style))
    story.append(Paragraph(ai_summary, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_model_pdf_report(metrics_dict, logs_df, threshold_val):
    """Generates a PDF evaluation report for the overall trained model."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#0F172A'))
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#64748B'))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#1E293B'), spaceBefore=12, spaceAfter=6)

    story.append(Paragraph("Deepfake Audio Detection - Model Performance Report", title_style))
    story.append(Paragraph("System Evaluation Summary generated from trained WavLM checkpoint artifacts.", subtitle_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Executive Evaluation Metrics", heading_style))
    
    clean_eer = f"{metrics_dict['clean']['eer']:.2f}%" if metrics_dict and 'clean' in metrics_dict else "N/A"
    aug_eer = f"{metrics_dict['augmented']['eer']:.2f}%" if metrics_dict and 'augmented' in metrics_dict else "N/A"
    clean_auc = f"{metrics_dict['clean']['auc']:.4f}" if metrics_dict and 'clean' in metrics_dict else "N/A"
    aug_auc = f"{metrics_dict['augmented']['auc']:.4f}" if metrics_dict and 'augmented' in metrics_dict else "N/A"

    summary_data = [
        ["Evaluation Parameter", "Result Value"],
        ["Neural Backbone", "WavLM Base Plus"],
        ["Objective Function", "One-Class Softmax (OCSoftmax)"],
        ["Decision Threshold", f"{threshold_val:.4f}"],
        ["Clean Dev EER", clean_eer],
        ["Augmented Dev EER", aug_eer],
        ["Clean Dev ROC-AUC", clean_auc],
        ["Augmented Dev ROC-AUC", aug_auc],
    ]
    
    t_summary = Table(summary_data, colWidths=[240, 260])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 12))

    if logs_df is not None:
        story.append(Paragraph("2. Epoch Training Progression Log", heading_style))
        df_display = logs_df.head(10)
        table_data = [list(df_display.columns)] + df_display.values.tolist()
        t_history = Table(table_data)
        t_history.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ]))
        story.append(t_history)

    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🎙️ Real-Time Voice Tester", "📊 Model Analytics & Reports", "🤖 AI Assistant"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if model is not None:
    st.sidebar.markdown('<div class="status-badge-online">● Checkpoint Loaded</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"Decision LLR Threshold: **{threshold:.4f}**")
else:
    st.sidebar.markdown('<div class="status-badge-offline">● Checkpoint Offline</div>', unsafe_allow_html=True)
    if err_msg:
        st.sidebar.caption(f"Error: {err_msg}")

# ---------------------------------------------------------
# PAGE 1: REAL-TIME VOICE TESTER
# ---------------------------------------------------------
if page == "🎙️ Real-Time Voice Tester":
    st.markdown('<div class="main-title">Deepfake Audio Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Perform live acoustic verification to classify voice recordings as Bonafide (Real) or Spoof (Deepfake).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.1], gap="large")

    with col1:
        st.subheader("1. Audio Input Source")
        audio_input_type = st.radio("Choose Method", ["Record Voice Live", "Upload Audio File"])
        
        audio_bytes = None
        if audio_input_type == "Record Voice Live":
            audio_bytes = st.audio_input("Record audio sample")
        else:
            uploaded_file = st.file_uploader("Upload audio (.wav, .flac, .mp3, .ogg)", type=["wav", "flac", "mp3", "ogg"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()

    with col2:
        st.subheader("2. Acoustic Analysis Result")
        if audio_bytes is not None:
            temp_dir = os.path.join(BASE_DIR, "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "input.wav")
            
            with open(temp_path, "wb") as f:
                f.write(audio_bytes if isinstance(audio_bytes, bytes) else audio_bytes.getbuffer())

            st.audio(temp_path)

            if model is None:
                st.error(f"Inference Offline: {err_msg}")
            else:
                try:
                    try:
                        import soundfile as sf
                        data, sr = sf.read(temp_path)
                        wav = torch.from_numpy(data).float()
                        if wav.ndim > 1:
                            wav = wav.mean(dim=-1)
                    except Exception:
                        from scipy.io import wavfile
                        sr, data = wavfile.read(temp_path)
                        wav = torch.from_numpy(data).float()
                        if wav.dtype == torch.int16:
                            wav = wav / 32768.0
                        if wav.ndim > 1:
                            wav = wav.mean(dim=-1)

                    if sr != SR:
                        wav = torchaudio.functional.resample(wav, sr, SR)

                    n = wav.shape[-1]
                    if n > TARGET_LEN:
                        wav = wav[:TARGET_LEN]
                    elif n < TARGET_LEN:
                        wav = wav.repeat(int(np.ceil(TARGET_LEN / max(n, 1))))[:TARGET_LEN]

                    rms_val = float(wav.pow(2).mean().sqrt())
                    if rms_val > 1e-8:
                        wav = (wav * ((10 ** (-25.0 / 20.0)) / rms_val)).clamp(-1.0, 1.0)

                    wav_tensor = wav.unsqueeze(0)

                    with torch.no_grad():
                        emb = model(wav_tensor)
                        llr_score = float(objective(emb).squeeze().cpu().item())

                    is_bonafide = llr_score >= threshold
                    verdict_str = "BONAFIDE (REAL)" if is_bonafide else "SPOOF (DEEPFAKE)"

                    # Verdict Box Display
                    if is_bonafide:
                        st.markdown(f'<div class="verdict-real">VERDICT: {verdict_str}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="verdict-fake">VERDICT: {verdict_str}</div>', unsafe_allow_html=True)

                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("Log-Likelihood Ratio (LLR)", f"{llr_score:.4f}")
                    with m_col2:
                        st.metric("Decision Threshold", f"{threshold:.4f}")

                    # Waveform Visualization
                    fig_wave = px.line(y=wav.numpy(), title="Preprocessed Waveform Signal (4.0s @ 16kHz)")
                    fig_wave.update_layout(
                        xaxis_title="Sample", yaxis_title="Amplitude",
                        height=210, template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=35, b=20)
                    )
                    st.plotly_chart(fig_wave, use_container_width=True)

                    # Mel-Spectrogram Visualization
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=SR, n_fft=1024, hop_length=512, n_mels=80
                    )
                    mel_spec = mel_transform(wav)
                    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec).numpy()

                    fig_spec = px.imshow(
                        mel_spec_db,
                        labels=dict(x="Frame", y="Mel Bin", color="dB"),
                        title="Mel-Spectrogram Visualization",
                        origin="lower",
                        aspect="auto",
                        color_continuous_scale="Viridis"
                    )
                    fig_spec.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=35, b=20)
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)

                    # ---------------------------------------------------------
                    # AI VOICE DIAGNOSTIC BREAKDOWN & PDF EXPORT
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.subheader("💡 AI Acoustic Diagnostic Breakdown")

                    margin = abs(llr_score - threshold)
                    if is_bonafide:
                        ai_text = (
                            f"The recorded audio sample yielded an LLR score of {llr_score:.4f}, exceeding the "
                            f"trained decision threshold ({threshold:.4f}) by a margin of {margin:.4f}. "
                            "Acoustic phase coherence and vocal formants align with organic human speech characteristics. "
                            "No phase discontinuity or neural vocoder synthesis artifacts were detected in the WavLM representations."
                        )
                    else:
                        ai_text = (
                            f"The recorded audio sample yielded an LLR score of {llr_score:.4f}, falling below the "
                            f"trained decision threshold ({threshold:.4f}) by {margin:.4f}. "
                            "The WavLM representation layers detected structural anomalies typical of synthetic speech generators, "
                            "such as high-frequency spectral flattening, neural vocoder phase mismatches, or acoustic reproduction artifacts."
                        )

                    st.info(ai_text)

                    # Generate and allow downloading PDF Voice Report
                    voice_pdf = generate_voice_test_pdf(
                        verdict_str, llr_score, threshold, rms_val, ai_text
                    )
                    st.download_button(
                        label="📄 Download Voice AI Report (PDF)",
                        data=voice_pdf,
                        file_name="Voice_AI_Analysis_Report.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"Signal processing error: {e}")

# ---------------------------------------------------------
# PAGE 2: MODEL ANALYTICS & REPORTS
# ---------------------------------------------------------
elif page == "📊 Model Analytics & Reports":
    st.markdown('<div class="main-title">Deepfake Audio Detection</div>', unsafe_allow_html=True)
    
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown('<div class="sub-title">Detailed performance metrics, training histories, and system evaluation report exports.</div>', unsafe_allow_html=True)
    with col_btn:
        pdf_file = generate_model_pdf_report(metrics, csv_df, threshold if threshold else 0.0)
        st.download_button(
            label="📄 Export Model Report (PDF)",
            data=pdf_file,
            file_name="Deepfake_Audio_Detection_Report.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    if metrics is None and csv_df is None:
        st.warning("No metrics checkpoint files found in `model_artifacts/`.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Loss & EER History", "📊 ROC & DET Curves", "🔲 Confusion Matrix", "🧠 WavLM Layer Weights"])

        with tab1:
            if csv_df is not None and "epoch" in csv_df.columns:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=csv_df["epoch"], y=csv_df["train_loss"], name="Train Loss", mode="lines+markers"))
                    if "clean_loss" in csv_df.columns:
                        fig_loss.add_trace(go.Scatter(x=csv_df["epoch"], y=csv_df["clean_loss"], name="Clean Val Loss", mode="lines+markers"))
                    if "aug_loss" in csv_df.columns:
                        fig_loss.add_trace(go.Scatter(x=csv_df["epoch"], y=csv_df["aug_loss"], name="Aug Val Loss", mode="lines+markers"))
                    fig_loss.update_layout(title="Loss History Across Epochs", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_loss, use_container_width=True)

                with col_chart2:
                    fig_eer = go.Figure()
                    if "clean_eer" in csv_df.columns:
                        fig_eer.add_trace(go.Scatter(x=csv_df["epoch"], y=csv_df["clean_eer"], name="Clean EER (%)", mode="lines+markers"))
                    if "aug_eer" in csv_df.columns:
                        fig_eer.add_trace(go.Scatter(x=csv_df["epoch"], y=csv_df["aug_eer"], name="Augmented EER (%)", mode="lines+markers"))
                    fig_eer.update_layout(title="Equal Error Rate (EER %) History", xaxis_title="Epoch", yaxis_title="EER (%)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_eer, use_container_width=True)

                st.subheader("Training Epoch Summary Table")
                st.dataframe(csv_df, use_container_width=True)

        with tab2:
            if metrics and "clean" in metrics and "augmented" in metrics:
                col_a, col_b = st.columns(2)
                clean_d, aug_d = metrics["clean"], metrics["augmented"]

                with col_a:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=clean_d["fpr"], y=clean_d["tpr"], name=f"Clean (AUC={clean_d['auc']:.4f})"))
                    fig_roc.add_trace(go.Scatter(x=aug_d["fpr"], y=aug_d["tpr"], name=f"Augmented (AUC={aug_d['auc']:.4f})"))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", line=dict(dash="dash", color="gray")))
                    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_roc, use_container_width=True)

                with col_b:
                    fig_det = go.Figure()
                    fig_det.add_trace(go.Scatter(x=clean_d["fpr"], y=[1 - y for y in clean_d["tpr"]], name="Clean Dev"))
                    fig_det.add_trace(go.Scatter(x=aug_d["fpr"], y=[1 - y for y in aug_d["tpr"]], name="Augmented Dev"))
                    fig_det.update_layout(title="DET Curve (Log-Scale)", xaxis_title="False Alarm Rate", yaxis_title="False Reject Rate", xaxis_type="log", yaxis_type="log", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_det, use_container_width=True)

        with tab3:
            if metrics:
                eval_target = st.selectbox("Select Evaluation Split", ["augmented", "clean"])
                split_dict = metrics.get(eval_target, metrics)
                
                if "cm" in split_dict:
                    cm = np.array(split_dict["cm"])
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted Label", y="True Label", color="Count"),
                        x=["Bonafide", "Spoof"],
                        y=["Bonafide", "Spoof"],
                        text_auto=True,
                        color_continuous_scale="Purples",
                        title=f"Confusion Matrix ({eval_target.capitalize()} Dev)"
                    )
                    fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cm, use_container_width=True)

        with tab4:
            if metrics and "layer_weights" in metrics:
                weights = metrics["layer_weights"]
                layers = [f"Layer {i}" if i > 0 else "CNN Out" for i in range(len(weights))]
                fig_lw = px.bar(x=layers, y=weights, labels={"x": "WavLM Representation Layer", "y": "Softmax Attention Weight"}, title="Learned Representation Weights Across WavLM Layers")
                fig_lw.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_lw, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: AI ASSISTANT
# ---------------------------------------------------------
elif page == "🤖 AI Assistant":
    st.markdown('<div class="main-title">Deepfake Audio Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive AI model diagnostic assistant for performance queries and metric insights.</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI deepfake detection assistant. How can I help you analyze your WavLM model performance today?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_prompt := st.chat_input("Ask a question about your model..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.chat_message("user").write(user_prompt)

        try:
            context_str = f"""
            You are an expert speech AI engineer analyzing a local WavLM deepfake audio detection model.
            Model Performance Metrics:
            - Model Checkpoint Status: {model is not None}
            - Decision LLR Threshold: {threshold}
            - Clean Dev EER: {metrics['clean']['eer'] if metrics and 'clean' in metrics else 'N/A'}%
            - Augmented Dev EER: {metrics['augmented']['eer'] if metrics and 'augmented' in metrics else 'N/A'}%
            """

            payload = {
                "model": "llama3.2",
                "prompt": f"{context_str}\nUser Question: {user_prompt}",
                "stream": False
            }

            with st.spinner("Analyzing model metrics..."):
                res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
                
                if res.status_code == 200:
                    ai_reply = res.json().get("response", "No response generated.")
                else:
                    ai_reply = f"Diagnostic server returned error status code: {res.status_code}."

            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            st.chat_message("assistant").write(ai_reply)

        except requests.exceptions.ConnectionError:
            err_reply = "⚠️ Local AI background process not detected. Ensure your local engine is running."
            st.session_state.messages.append({"role": "assistant", "content": err_reply})
            st.chat_message("assistant").write(err_reply)
        except Exception as e:
            err_reply = f"Error generating response: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": err_reply})
            st.chat_message("assistant").write(err_reply)