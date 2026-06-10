import streamlit as st
import textwrap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calidad del Agua",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fondo principal */
.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1a2e;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Ocultar elementos de Streamlit — mantenemos header para el botón del sidebar */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; background: transparent !important; }
header [data-testid="stHeader"] { background: transparent; }

/* Botón de colapsar/expandir sidebar — siempre visible */
button[kind="header"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #38bdf8 !important;
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
}
[data-testid="collapsedControl"]:hover {
    background: #1e3a5f !important;
}

/* Tarjetas de métricas */
.metric-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #0d2244 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value.good { color: #34d399; }
.metric-value.mid  { color: #fbbf24; }
.metric-value.low  { color: #f87171; }

/* Resultado de predicción */
.result-potable {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #34d399;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-no-potable {
    background: linear-gradient(135deg, #1c0a0a 0%, #3b0f0f 100%);
    border: 1px solid #f87171;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 8px;
}
.result-subtitle {
    font-size: 14px;
    color: #94a3b8;
}
.prob-bar-container {
    background: #1e293b;
    border-radius: 8px;
    height: 10px;
    margin: 16px 0 6px 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* Sección hero */
.hero {
    padding: 32px 0 24px 0;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 32px;
}
.hero-tag {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8;
    font-weight: 600;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 36px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.hero-sub {
    font-size: 15px;
    color: #64748b;
    margin-top: 8px;
}

/* Separador de sección */
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* Info chips */
.chip {
    display: inline-block;
    background: #0f2040;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #94a3b8;
    margin: 2px;
}

/* Variable description card */
.var-card {
    background: #0f1a2e;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* ── HOME PAGE ── */

/* Header con logo y firma */
.home-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px 32px 20px 32px;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 40px;
    background: linear-gradient(180deg, #0d1b33 0%, transparent 100%);
    border-radius: 12px 12px 0 0;
}

/* Hero section */
.home-hero {
    text-align: center;
    padding: 20px 40px 36px 40px;
}
.home-tag {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38bdf8;
    font-weight: 600;
    margin-bottom: 14px;
}
.home-title {
    font-size: 34px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.25;
    max-width: 720px;
    margin: 0 auto 14px auto;
}
.home-subtitle {
    font-size: 16px;
    color: #64748b;
    max-width: 580px;
    margin: 0 auto;
    font-style: italic;
    line-height: 1.6;
}

/* Info cards grid */
.info-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 40px 0;
}
.info-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 22px 20px;
}
.info-card-icon {
    font-size: 22px;
    margin-bottom: 10px;
}
.info-card-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
}
.info-card-item {
    font-size: 13px;
    color: #94a3b8;
    padding: 4px 0;
    display: flex;
    align-items: flex-start;
    gap: 7px;
    line-height: 1.5;
}
.info-card-item::before {
    content: "·";
    color: #38bdf8;
    font-weight: 700;
    flex-shrink: 0;
}

/* Sección cards navegación */
.nav-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 8px;
}
.nav-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #091526 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 28px 24px 24px 24px;
    transition: border-color 0.2s, transform 0.2s;
}
.nav-card:hover {
    border-color: #38bdf8;
}
.nav-card-icon  { font-size: 28px; margin-bottom: 10px; }
.nav-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
}
.nav-card-desc  { font-size: 13px; color: #64748b; line-height: 1.55; margin-bottom: 18px; }

/* Botón de navegación */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
}
.var-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #38bdf8;
    font-weight: 500;
}
.var-desc {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 2px;
}
</style>
""").strip(), unsafe_allow_html=True)

# ── Colores para matplotlib ────────────────────────────────────────────────────
DARK_BG   = "#0b1120"
CARD_BG   = "#0f1a2e"
BORDER    = "#1e3a5f"
ACCENT    = "#38bdf8"
GREEN     = "#34d399"
RED       = "#f87171"
YELLOW    = "#fbbf24"
TEXT_MAIN = "#e2e8f0"
TEXT_MUTED= "#64748b"

def style_fig(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.yaxis.label.set_color(TEXT_MUTED)
        ax.title.set_color(TEXT_MAIN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

# ── Carga de datos y modelo ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("modelo_agua.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

@st.cache_data
def get_test_split():
    df = load_data()
    X = df.drop(columns="Potability")
    y = df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test, y_test

try:
    modelo = load_model()
    df     = load_data()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── Imágenes embebidas ────────────────────────────────────────────────────────
import base64, os

def img_to_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

firma_b64 = img_to_b64("Firma.png")
logo_b64  = img_to_b64("Logo.png")

# ── Estado de navegación ──────────────────────────────────────────────────────
if "pagina" not in st.session_state:
    st.session_state.pagina = "Home"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo en sidebar
    if logo_b64:
        st.markdown(textwrap.dedent(f"""
        <div style='text-align:center; padding: 8px 0 4px 0;'>
            <img src='data:image/png;base64,{logo_b64}' style='width:90%; border-radius:8px;'/>
        </div>
        """).strip(), unsafe_allow_html=True)
    else:
        st.markdown("### 💧 Calidad del Agua")
    st.markdown("---")

    opciones = ["🏠  Home", "🔬  Predicción", "📊  Rendimiento del modelo", "🗂️  Exploración de datos"]
    labels   = ["Home", "Predicción", "Rendimiento del modelo", "Exploración de datos"]

    seleccion = st.radio(
        "Navegación",
        opciones,
        index=labels.index(st.session_state.pagina) if st.session_state.pagina in labels else 0,
        label_visibility="collapsed"
    )
    pagina = labels[opciones.index(seleccion)]
    st.session_state.pagina = pagina

    st.markdown("---")
    st.markdown(textwrap.dedent("""
    <div style='font-size:12px; color:#475569; line-height:1.7'>
    <b style='color:#64748b'>Modelo</b><br>Random Forest<br><br>
    <b style='color:#64748b'>Dataset</b><br>Water Potability<br>3.276 muestras<br><br>
    <b style='color:#64748b'>Variables</b><br>9 parámetros fisicoquímicos<br><br>
    <b style='color:#64748b'>Proyecto</b><br>Machine Learning · Python
    </div>
    """).strip(), unsafe_allow_html=True)

# ── Aviso si falta el modelo ───────────────────────────────────────────────────
if not MODEL_LOADED:
    st.error("⚠️ No se encontró `modelo_agua.pkl`. Ejecutá todas las celdas del notebook primero.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Home":

    # ── Header: Logo + Firma ──────────────────────────────────────────────────
    col_logo, col_firma = st.columns([1.6, 1], gap="large")

    with col_logo:
        if logo_b64:
            st.markdown(textwrap.dedent(f"""
            <div style='padding: 16px 0 8px 0;'>
                <img src='data:image/png;base64,{logo_b64}'
                     style='max-width:420px; width:100%; border-radius:10px;'/>
            </div>
            """).strip(), unsafe_allow_html=True)

    with col_firma:
        if firma_b64:
            st.markdown(textwrap.dedent(f"""
            <div style='padding: 16px 0 8px 0; text-align:right;'>
                <img src='data:image/png;base64,{firma_b64}'
                     style='max-width:280px; width:100%; border-radius:10px;'/>
            </div>
            """).strip(), unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid #1e3a5f; margin: 8px 0 36px 0;'>", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(textwrap.dedent("""
    <div class='home-hero'>
        <div class='home-tag'>Machine Learning · Water Quality</div>
        <div class='home-title'>Predicción de la Potabilidad del Agua usando Machine Learning</div>
        <div class='home-subtitle'>
            "Evaluar la calidad del agua utilizando parámetros fisicoquímicos
            y un modelo optimizado de Random Forest."
        </div>
    </div>
    """).strip(), unsafe_allow_html=True)

    # ── Info cards ────────────────────────────────────────────────────────────
# ── Info cards ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='info-grid'>
        <div class='info-card'>
            <div class='info-card-icon'>🗄️</div>
            <div class='info-card-title'>Dataset</div>
            <div class='info-card-item'>3,276 water samples</div>
            <div class='info-card-item'>9 physicochemical features</div>
        </div>
        <div class='info-card'>
            <div class='info-card-icon'>🤖</div>
            <div class='info-card-title'>Model</div>
            <div class='info-card-item'>Random Forest (Optimized)</div>
            <div class='info-card-item'>GridSearchCV</div>
            <div class='info-card-item'>Threshold Optimization</div>
        </div>
        <div class='info-card'>
            <div class='info-card-icon'>📈</div>
            <div class='info-card-title'>Performance</div>
            <div class='info-card-item'>Accuracy: 0.60</div>
            <div class='info-card-item'>Recall: 0.63</div>
            <div class='info-card-item'>ROC-AUC: 0.67</div>
        </div>
        <div class='info-card'>
            <div class='info-card-icon'>🔬</div>
            <div class='info-card-title'>Features Analyzed</div>
            <div class='info-card-item'>pH, Hardness, Sulfate, Chloramines, Solids, Conductivity, Organic Carbon, Trihalomethanes, Turbidity</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Secciones con botones ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Secciones</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown(textwrap.dedent("""
        <div class='nav-card'>
            <div class='nav-card-icon'>🔬</div>
            <div class='nav-card-title'>Prediction</div>
            <div class='nav-card-desc'>
                Ingresá los parámetros fisicoquímicos de una muestra y obtené
                una predicción de potabilidad con probabilidad y gauge visual.
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
        if st.button("Ir a Predicción", key="btn_pred"):
            st.session_state.pagina = "Predicción"
            st.rerun()

    with c2:
        st.markdown(textwrap.dedent("""
        <div class='nav-card'>
            <div class='nav-card-icon'>📊</div>
            <div class='nav-card-title'>Model Performance</div>
            <div class='nav-card-desc'>
                Explorá las métricas del modelo: matriz de confusión, curva ROC,
                importancia de variables y comparativa de umbrales de decisión.
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
        if st.button("Ir a Rendimiento del modelo", key="btn_model"):
            st.session_state.pagina = "Rendimiento del modelo"
            st.rerun()

    with c3:
        st.markdown(textwrap.dedent("""
        <div class='nav-card'>
            <div class='nav-card-icon'>🗂️</div>
            <div class='nav-card-title'>Data Exploration</div>
            <div class='nav-card-desc'>
                Analizá el dataset original: distribuciones por clase, valores
                faltantes, mapa de correlación y estadísticas descriptivas.
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
        if st.button("Ir a Exploración de datos", key="btn_data"):
            st.session_state.pagina = "Exploración de datos"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Predicción":

    st.markdown(textwrap.dedent("""
    <div class='hero'>
        <div class='hero-tag'>Clasificación de potabilidad</div>
        <div class='hero-title'>¿Es segura esta muestra de agua?</div>
        <div class='hero-sub'>Ingresá los parámetros fisicoquímicos para obtener una predicción.</div>
    </div>
    """).strip(), unsafe_allow_html=True)

    # Rangos del dataset — límites según notebook (descripción de variables)
    VARIABLES = {
        "ph":              {"label": "pH",                      "min": 0.0,   "max": 14.0,   "default": 7.0,   "step": 0.1,  "desc": "Acidez o alcalinidad del agua.",              "limite": "Rango aceptado: 6.5 – 8.5"},
        "Hardness":        {"label": "Dureza (mg/L)",           "min": 47.0,  "max": 323.0,  "default": 180.0, "step": 1.0,  "desc": "Calcio y magnesio disueltos.",                "limite": "Máximo permitido: 400 mg/L"},
        "Solids":          {"label": "Sólidos disueltos (mg/L)","min": 320.0, "max": 61000.0,"default":20000.0,"step":100.0, "desc": "Total de sólidos disueltos (TDS).",           "limite": "Máximo permitido: 1500 mg/L"},
        "Chloramines":     {"label": "Cloraminas (mg/L)",       "min": 0.35,  "max": 13.0,   "default": 7.0,   "step": 0.1,  "desc": "Compuesto desinfectante del agua.",           "limite": "Máximo permitido: 2 mg/L"},
        "Sulfate":         {"label": "Sulfato (mg/L)",          "min": 129.0, "max": 481.0,  "default": 330.0, "step": 1.0,  "desc": "Sulfatos disueltos en el agua.",              "limite": "Máximo permitido: 400 mg/L"},
        "Conductivity":    {"label": "Conductividad (μS/cm)",   "min": 181.0, "max": 753.0,  "default": 420.0, "step": 1.0,  "desc": "Indicador indirecto de iones disueltos.",     "limite": "Rango aceptado: 200 – 800 μS/cm"},
        "Organic_carbon":  {"label": "Carbono orgánico (mg/L)", "min": 2.0,   "max": 28.0,   "default": 14.0,  "step": 0.1,  "desc": "Materia orgánica presente en el agua.",      "limite": "Máximo permitido: 2 mg/L"},
        "Trihalomethanes": {"label": "Trihalometanos (μg/L)",   "min": 0.74,  "max": 124.0,  "default": 66.0,  "step": 0.5,  "desc": "Subproductos de la reacción cloro–materia orgánica.", "limite": "Máximo permitido: 100 μg/L"},
        "Turbidity":       {"label": "Turbidez (NTU)",          "min": 1.45,  "max": 6.50,   "default": 3.9,   "step": 0.05, "desc": "Claridad del agua — partículas suspendidas.", "limite": "Máximo permitido: 3 NTU"},
    }

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-header'>Parámetros de la muestra</div>", unsafe_allow_html=True)
        valores = {}
        for key, meta in VARIABLES.items():
            valores[key] = st.slider(
                meta["label"],
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
                help=meta["desc"]
            )

    with col_result:
        st.markdown("<div class='section-header'>Resultado</div>", unsafe_allow_html=True)

        muestra = pd.DataFrame([valores])
        prob    = modelo.predict_proba(muestra)[0][1]
        pred    = int(prob >= 0.42)  # umbral optimizado

        if pred == 1:
            bar_color = "#34d399"
            card_class = "result-potable"
            icono = "✅"
            titulo = "Agua Potable"
            mensaje = "Los parámetros están dentro de rangos aceptables."
        else:
            bar_color = "#f87171"
            card_class = "result-no-potable"
            icono = "⚠️"
            titulo = "No Potable"
            mensaje = "Uno o más parámetros superan los límites recomendados."

        bar_width = int(prob * 100)

        st.markdown(textwrap.dedent(f"""
        <div class='{card_class}'>
            <div class='result-title'>{icono} {titulo}</div>
            <div class='result-subtitle'>{mensaje}</div>
            <div class='prob-bar-container'>
                <div class='prob-bar-fill' style='width:{bar_width}%; background:{bar_color};'></div>
            </div>
            <div style='font-family:"JetBrains Mono",monospace; font-size:28px; font-weight:700; color:{bar_color}'>
                {prob:.1%}
            </div>
            <div style='font-size:12px; color:#475569; margin-top:4px'>
                probabilidad de potabilidad · umbral 0.42
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge visual con matplotlib
        fig, ax = plt.subplots(figsize=(5, 2.8))
        style_fig(fig, ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Arco de fondo
        theta = np.linspace(np.pi, 0, 100)
        ax.plot(0.5 + 0.38 * np.cos(theta), 0.15 + 0.38 * np.sin(theta),
                color=BORDER, linewidth=14, solid_capstyle="round")
        # Arco de valor
        theta_val = np.linspace(np.pi, np.pi - np.pi * prob, 100)
        ax.plot(0.5 + 0.38 * np.cos(theta_val), 0.15 + 0.38 * np.sin(theta_val),
                color=bar_color, linewidth=14, solid_capstyle="round")

        ax.text(0.5, 0.30, f"{prob:.0%}", ha="center", va="center",
                fontsize=24, fontweight="bold", color=bar_color,
                fontfamily="monospace")
        ax.text(0.5, 0.08, "Probabilidad de potabilidad", ha="center",
                fontsize=9, color=TEXT_MUTED)
        ax.text(0.10, 0.05, "0%",  ha="center", fontsize=9, color=TEXT_MUTED)
        ax.text(0.90, 0.05, "100%", ha="center", fontsize=9, color=TEXT_MUTED)

        st.pyplot(fig, width='stretch')
        plt.close()

        # Referencia de variables
        st.markdown("<br><div class='section-header'>Referencia de variables</div>", unsafe_allow_html=True)
        for key, meta in VARIABLES.items():
            val_actual = valores[key]
            # Determinar si el valor actual supera el límite (solo para máximos)
            en_rango = True
            limite_txt = meta['limite']
            if "Máximo" in limite_txt:
                try:
                    lim_val = float(limite_txt.split(":")[1].strip().split(" ")[0])
                    en_rango = val_actual <= lim_val
                except Exception:
                    pass
            elif "Rango" in limite_txt:
                try:
                    partes = limite_txt.split(":")[1].strip().split("–")
                    lim_min = float(partes[0].strip().split(" ")[0])
                    lim_max = float(partes[1].strip().split(" ")[0])
                    en_rango = lim_min <= val_actual <= lim_max
                except Exception:
                    pass

            color_limite = "#34d399" if en_rango else "#f87171"
            icono_limite = "✓" if en_rango else "✗"
            st.markdown(textwrap.dedent(f"""
            <div class='var-card'>
                <div class='var-name'>{meta['label']}</div>
                <div class='var-desc'>{meta['desc']}</div>
                <div style='font-size:11px; margin-top:5px; color:{color_limite}; font-weight:600;'>
                    {icono_limite} {limite_txt}
                </div>
            </div>
            """).strip(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — RENDIMIENTO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Rendimiento del modelo":

    st.markdown(textwrap.dedent("""
    <div class='hero'>
        <div class='hero-tag'>Evaluación del modelo</div>
        <div class='hero-title'>Rendimiento de Random Forest</div>
        <div class='hero-sub'>Métricas, curvas y análisis de variables sobre el conjunto de prueba.</div>
    </div>
    """).strip(), unsafe_allow_html=True)

    X_test, y_test = get_test_split()
    y_pred  = modelo.predict(X_test)
    y_probs = modelo.predict_proba(X_test)[:, 1]
    y_pred_42 = (y_probs >= 0.42).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred_42, zero_division=0)
    rec  = recall_score(y_test, y_pred_42, zero_division=0)
    f1   = f1_score(y_test, y_pred_42, zero_division=0)
    auc  = roc_auc_score(y_test, y_probs)

    def color_class(v):
        if v >= 0.65: return "good"
        if v >= 0.50: return "mid"
        return "low"

    # Métricas
    st.markdown("<div class='section-header'>Métricas principales · umbral 0.42</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in zip(
        [c1, c2, c3, c4, c5],
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        [acc, prec, rec, f1, auc]
    ):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value {color_class(val)}'>{val:.2f}</div>
            </div>
            """).strip(), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    # ── Matriz de confusión ────────────────────────────────────────────────────
    with col_a:
        st.markdown("<div class='section-header'>Matriz de confusión</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred_42)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        style_fig(fig, ax)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, linecolor=BORDER,
                    annot_kws={"size": 16, "weight": "bold", "color": TEXT_MAIN},
                    cbar=False)
        ax.set_xlabel("Predicción", color=TEXT_MUTED, fontsize=11)
        ax.set_ylabel("Valor real", color=TEXT_MUTED, fontsize=11)
        ax.set_xticklabels(["No potable", "Potable"], color=TEXT_MUTED)
        ax.set_yticklabels(["No potable", "Potable"], color=TEXT_MUTED, rotation=0)
        ax.set_title("Confusión · umbral 0.42", color=TEXT_MAIN, fontsize=12, pad=12)
        st.pyplot(fig, width='stretch')
        plt.close()

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    with col_b:
        st.markdown("<div class='section-header'>Curva ROC</div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        style_fig(fig, ax)
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color=TEXT_MUTED, linewidth=1, label="Aleatorio")
        ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
        ax.set_xlabel("Tasa de Falsos Positivos", color=TEXT_MUTED, fontsize=11)
        ax.set_ylabel("Tasa de Verdaderos Positivos", color=TEXT_MUTED, fontsize=11)
        ax.set_title("Curva ROC", color=TEXT_MAIN, fontsize=12, pad=12)
        ax.legend(frameon=False, labelcolor=TEXT_MUTED, fontsize=10)
        st.pyplot(fig, width='stretch')
        plt.close()

    # ── Importancia de variables ───────────────────────────────────────────────
    st.markdown("<br><div class='section-header'>Importancia de variables</div>", unsafe_allow_html=True)
    importances = modelo.named_steps["classifier"].feature_importances_
    feat_df = (
        pd.DataFrame({"Variable": df.drop(columns="Potability").columns, "Importancia": importances})
        .sort_values("Importancia", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    colors = [ACCENT if v == feat_df["Importancia"].max() else "#1e4a6e" for v in feat_df["Importancia"]]
    bars = ax.barh(feat_df["Variable"], feat_df["Importancia"], color=colors, height=0.6)
    for bar, val in zip(bars, feat_df["Importancia"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9,
                color=ACCENT if val == feat_df["Importancia"].max() else TEXT_MUTED,
                fontfamily="monospace")
    ax.set_xlabel("Importancia (reducción de impureza promedio)", color=TEXT_MUTED, fontsize=10)
    ax.set_title("Contribución de cada variable al modelo", color=TEXT_MAIN, fontsize=12, pad=12)
    ax.set_xlim(0, feat_df["Importancia"].max() * 1.18)
    st.pyplot(fig, width='stretch')
    plt.close()

    # ── Comparación de umbrales ────────────────────────────────────────────────
    st.markdown("<br><div class='section-header'>Comparación de umbrales de decisión</div>", unsafe_allow_html=True)

    umbrales = np.round(np.arange(0.30, 0.71, 0.05), 2)
    rows = []
    for u in umbrales:
        yp = (y_probs >= u).astype(int)
        rows.append({
            "Umbral": u,
            "Precision": precision_score(y_test, yp, zero_division=0),
            "Recall":    recall_score(y_test, yp, zero_division=0),
            "F1-score":  f1_score(y_test, yp, zero_division=0),
        })
    thresh_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    style_fig(fig, ax)
    ax.plot(thresh_df["Umbral"], thresh_df["Precision"], color=YELLOW,   linewidth=2, marker="o", markersize=4, label="Precision")
    ax.plot(thresh_df["Umbral"], thresh_df["Recall"],    color=GREEN,    linewidth=2, marker="o", markersize=4, label="Recall")
    ax.plot(thresh_df["Umbral"], thresh_df["F1-score"],  color=ACCENT,   linewidth=2, marker="o", markersize=4, label="F1-score")
    ax.axvline(0.42, color="#94a3b8", linewidth=1.2, linestyle="--", alpha=0.7, label="Umbral elegido (0.42)")
    ax.set_xlabel("Umbral de decisión", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Puntaje", color=TEXT_MUTED, fontsize=10)
    ax.set_title("Tradeoff Precision / Recall según umbral", color=TEXT_MAIN, fontsize=12, pad=12)
    ax.legend(frameon=False, labelcolor=TEXT_MUTED, fontsize=10)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.5)
    st.pyplot(fig, width='stretch')
    plt.close()

    st.markdown(textwrap.dedent("""
    <div style='background:#0f1a2e; border:1px solid #1e3a5f; border-radius:10px; padding:16px 20px; font-size:13px; color:#94a3b8; line-height:1.7; margin-top:8px'>
    💡 <b style='color:#e2e8f0'>¿Cómo elegir el umbral?</b> Bajar el umbral aumenta el recall (detecta más casos positivos reales) a costa de más falsos positivos. En problemas de agua potable, clasificar incorrectamente agua <i>no potable</i> como potable es el error más costoso, por lo que se prefiere un umbral más bajo que priorice el recall.
    </div>
    """).strip(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — EXPLORACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Exploración de datos":

    st.markdown(textwrap.dedent("""
    <div class='hero'>
        <div class='hero-tag'>Dataset · Water Potability</div>
        <div class='hero-title'>Exploración de datos</div>
        <div class='hero-sub'>Distribuciones, correlaciones y estadísticas del dataset original.</div>
    </div>
    """).strip(), unsafe_allow_html=True)

    # Estadísticas rápidas
    st.markdown("<div class='section-header'>Resumen del dataset</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Muestras</div><div class='metric-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Variables</div><div class='metric-value'>{df.shape[1]-1}</div></div>", unsafe_allow_html=True)
    with c3:
        pct_potable = df["Potability"].mean()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Agua potable</div><div class='metric-value mid'>{pct_potable:.1%}</div></div>", unsafe_allow_html=True)
    with c4:
        pct_null = df.isna().mean().mean()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Valores faltantes</div><div class='metric-value mid'>{pct_null:.1%}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribución de variable objetivo
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("<div class='section-header'>Balance de clases</div>", unsafe_allow_html=True)
        counts = df["Potability"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        style_fig(fig, ax)
        bars = ax.bar(["No potable", "Potable"], counts.values, color=[RED, GREEN], width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=10,
                    color=TEXT_MUTED)
        ax.set_title("Distribución de potabilidad", color=TEXT_MAIN, fontsize=12, pad=12)
        ax.set_ylabel("Cantidad de muestras", color=TEXT_MUTED, fontsize=10)
        ax.set_ylim(0, counts.max() * 1.22)
        st.pyplot(fig, width='stretch')
        plt.close()

    with col_b:
        st.markdown("<div class='section-header'>Valores faltantes por variable</div>", unsafe_allow_html=True)
        missing = (df.isna().mean() * 100).sort_values(ascending=True)
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        style_fig(fig, ax)
        ax.barh(missing.index, missing.values, color=YELLOW, height=0.5)
        for i, (idx, val) in enumerate(missing.items()):
            ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=10,
                    color=TEXT_MUTED, fontfamily="monospace")
        ax.set_xlabel("% de valores faltantes", color=TEXT_MUTED, fontsize=10)
        ax.set_title("Variables con datos ausentes", color=TEXT_MAIN, fontsize=12, pad=12)
        ax.set_xlim(0, missing.max() * 1.3)
        st.pyplot(fig, width='stretch')
        plt.close()

    # Distribuciones por variable
    st.markdown("<br><div class='section-header'>Distribución de variables por clase</div>", unsafe_allow_html=True)

    num_cols = [c for c in df.columns if c != "Potability"]
    sel_var = st.selectbox("Seleccioná una variable", num_cols, index=0)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    style_fig(fig, ax)
    for clase, color, etiqueta in [(0, RED, "No potable"), (1, GREEN, "Potable")]:
        data = df[df["Potability"] == clase][sel_var].dropna()
        ax.hist(data, bins=40, alpha=0.55, color=color, label=etiqueta, density=True)
        ax.axvline(data.median(), color=color, linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel(sel_var, color=TEXT_MUTED, fontsize=11)
    ax.set_ylabel("Densidad", color=TEXT_MUTED, fontsize=10)
    ax.set_title(f"Distribución de {sel_var} por clase (línea punteada = mediana)", color=TEXT_MAIN, fontsize=12, pad=12)
    handles = [mpatches.Patch(color=RED, alpha=0.7, label="No potable"),
               mpatches.Patch(color=GREEN, alpha=0.7, label="Potable")]
    ax.legend(handles=handles, frameon=False, labelcolor=TEXT_MUTED, fontsize=10)
    st.pyplot(fig, width='stretch')
    plt.close()

    # Mapa de correlación
    st.markdown("<br><div class='section-header'>Mapa de correlación</div>", unsafe_allow_html=True)
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    style_fig(fig, ax)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 9, "color": TEXT_MAIN},
                linewidths=0.5, linecolor=BORDER,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación entre variables", color=TEXT_MAIN, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    st.pyplot(fig, width='stretch')
    plt.close()

    # Estadísticas descriptivas
    st.markdown("<br><div class='section-header'>Estadísticas descriptivas</div>", unsafe_allow_html=True)
    desc = df.describe().T.round(3)
    st.dataframe(
        desc.style.background_gradient(cmap="Blues", axis=1, subset=["mean", "std"])
        .format("{:.3f}"),
        width='stretch'
    )
