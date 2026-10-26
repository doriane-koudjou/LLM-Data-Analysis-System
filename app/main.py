"""
main.py
-------
Streamlit web interface - v3 with response cards, SQL preview,
conversation starter, dark/light mode and language switcher.
"""

import sys
import os
import time
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / '.env')

import json
import streamlit as st
import pandas as pd
from database.db_connector import DatabaseConnector
from database.multi_db_connector import MultiDatabaseConnector
from agent.analyst_agent import DataAnalystAgent
from utils.hallucination_detector import HallucinationDetector
from utils.query_history import QueryHistory
from utils.file_uploader import process_upload

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind – LLM Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Translations ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "app_subtitle": "Ask questions about your data in natural language",
        "sidebar_add_db": "Connect Database",
        "conn_name_placeholder": "e.g. sales, finance",
        "host": "Host", "port": "Port",
        "db_name": "Database name", "user": "User", "password": "Password",
        "btn_connect": "Connect", "btn_disconnect": "Disconnect",
        "active_connections": "Active Connections",
        "no_connections": "No databases connected",
        "tables": "Tables",
        "upload_title": "Upload Data", "upload_target": "Import into",
        "upload_table": "Table name (optional)", "upload_btn": "Import",
        "history_title": "Query History",
        "history_queries": "queries this session",
        "btn_csv": "CSV", "btn_excel": "Excel",
        "btn_stats": "Statistics", "btn_reset": "Reset Chat",
        "llm_settings": "LLM Settings",
        "chat_placeholder": "Ask a question about your data...",
        "welcome_title": "Welcome to DataMind",
        "welcome_subtitle": "Connect a database to start analyzing your data",
        "welcome_feature1": "SQL queries generated automatically",
        "welcome_feature2": "Charts and visualizations",
        "welcome_feature3": "Predictive analytics with 10 models",
        "welcome_feature4": "Multi-database support",
        "example_questions": "Try asking:",
        "rate_answer": "Rate this answer",
        "hallucination_warning": "⚠️ Potential hallucination – please verify the numbers.",
        "analyzing": "Analyzing...",
        "connect_first": "Connect a database first",
        "import_success": "Imported", "import_error": "Import failed",
        "connect_success": "Connected", "connect_error": "Connection failed",
        "dark_mode": "Dark", "language": "Language",
        "avg_rating": "Avg. rating",
        "sql_used": "SQL Query",
        "sql_preview_title": "Preview SQL before running",
        "sql_preview_msg": "I will execute this SQL query:",
        "sql_run": "▶ Run Query",
        "sql_edit": "✏️ Edit",
        "sql_cancel": "✕ Cancel",
        "response_time": "Response time",
        "database": "Database",
        "conversation_starter": "Here's what I found in your database:",
        "conversation_starter_prompt": "What would you like to analyze?",
        "tables_found": "tables found",
        "copy": "Copy",
        "download_csv": "Download CSV",
    },
    "de": {
        "app_subtitle": "Stellen Sie Fragen zu Ihren Daten in natürlicher Sprache",
        "sidebar_add_db": "Datenbank verbinden",
        "conn_name_placeholder": "z.B. vertrieb, finanzen",
        "host": "Host", "port": "Port",
        "db_name": "Datenbankname", "user": "Benutzer", "password": "Passwort",
        "btn_connect": "Verbinden", "btn_disconnect": "Trennen",
        "active_connections": "Aktive Verbindungen",
        "no_connections": "Keine Datenbanken verbunden",
        "tables": "Tabellen",
        "upload_title": "Daten hochladen", "upload_target": "Importieren in",
        "upload_table": "Tabellenname (optional)", "upload_btn": "Importieren",
        "history_title": "Abfrageverlauf",
        "history_queries": "Abfragen in dieser Sitzung",
        "btn_csv": "CSV", "btn_excel": "Excel",
        "btn_stats": "Statistiken", "btn_reset": "Chat zurücksetzen",
        "llm_settings": "LLM-Einstellungen",
        "chat_placeholder": "Stellen Sie eine Frage zu Ihren Daten...",
        "welcome_title": "Willkommen bei DataMind",
        "welcome_subtitle": "Verbinden Sie eine Datenbank und analysieren Sie Ihre Daten",
        "welcome_feature1": "SQL-Abfragen automatisch generiert",
        "welcome_feature2": "Diagramme und Visualisierungen",
        "welcome_feature3": "Prädiktive Analysen mit 10 Modellen",
        "welcome_feature4": "Multi-Datenbank-Unterstützung",
        "example_questions": "Probieren Sie:",
        "rate_answer": "Antwort bewerten",
        "hallucination_warning": "⚠️ Mögliche Halluzination – bitte überprüfen Sie die Zahlen.",
        "analyzing": "Analysiere...",
        "connect_first": "Verbinden Sie zuerst eine Datenbank",
        "import_success": "Importiert", "import_error": "Import fehlgeschlagen",
        "connect_success": "Verbunden", "connect_error": "Verbindung fehlgeschlagen",
        "dark_mode": "Dunkel", "language": "Sprache",
        "avg_rating": "Ø Bewertung",
        "sql_used": "SQL-Abfrage",
        "sql_preview_title": "SQL-Vorschau vor Ausführung",
        "sql_preview_msg": "Ich werde diese SQL-Abfrage ausführen:",
        "sql_run": "▶ Ausführen",
        "sql_edit": "✏️ Bearbeiten",
        "sql_cancel": "✕ Abbrechen",
        "response_time": "Antwortzeit",
        "database": "Datenbank",
        "conversation_starter": "Folgendes habe ich in Ihrer Datenbank gefunden:",
        "conversation_starter_prompt": "Was möchten Sie analysieren?",
        "tables_found": "Tabellen gefunden",
        "copy": "Kopieren",
        "download_csv": "CSV herunterladen",
    },
}

EXAMPLE_QUESTIONS = {
    "en": [
        "How many customers are in the database?",
        "Top 5 products by revenue?",
        "Show orders per country as a chart",
        "Predict revenue for next 6 months",
    ],
    "de": [
        "Wie viele Kunden gibt es?",
        "Top 5 Produkte nach Umsatz?",
        "Bestellungen pro Land als Diagramm",
        "Umsatzvorhersage für 6 Monate",
    ],
}

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "agent": None,
    "multi_connector": MultiDatabaseConnector(),
    "connected": False,
    "query_history": QueryHistory(),
    "hallucination_detector": None,
    "ratings": {},
    "dark_mode": False,
    "language": "en",
    "prefill_question": None,
    "pending_sql": None,       # SQL waiting for user approval
    "pending_question": None,  # Question that triggered the SQL
    "conversation_started": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def t(key):
    lang = st.session_state.language
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# ── Helper: extract SQL from response ─────────────────────────────────────────
def extract_first_sql(text: str) -> str | None:
    """Extracts first SQL SELECT from text."""
    if not isinstance(text, str):
        return None
    patterns = [
        r'```sql\s*(SELECT[^`]+)```',
        r'```\s*(SELECT[^`]+)```',
        r'"query"\s*:\s*"(SELECT[^"]+)"',   # JSON tool call format
        r"'query'\s*:\s*'(SELECT[^']+)'",
        r'(SELECT\s+.+?(?:;|\Z))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

# ── CSS ───────────────────────────────────────────────────────────────────────
def get_css(dark: bool) -> str:
    if dark:
        bg="#0F172A"; bg2="#1E293B"; bg3="#334155"
        text="#F1F5F9"; text2="#94A3B8"
        accent="#818CF8"; accent2="#6366F1"
        border="#334155"; success="#10B981"
        warning="#F59E0B"; danger="#EF4444"
        code_bg="#0D1117"; input_bg="#1E293B"
    else:
        bg="#F8FAFC"; bg2="#FFFFFF"; bg3="#F1F5F9"
        text="#1E293B"; text2="#64748B"
        accent="#6366F1"; accent2="#4F46E5"
        border="#E2E8F0"; success="#10B981"
        warning="#F59E0B"; danger="#EF4444"
        code_bg="#F8FAFC"; input_bg="#FFFFFF"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {text};
    }}
    .stApp {{ background-color: {bg}; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {bg2};
        border-right: 1px solid {border};
    }}
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {accent}, {accent2});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px {accent}40;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px {accent}60;
    }}

    /* ── Main layout ── */
    .main .block-container {{
        padding-top: 1.5rem;
        max-width: 880px;
    }}

    /* ── App header ── */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.15rem;
    }}
    .app-title {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {text};
        letter-spacing: -0.75px;
        margin: 0;
    }}
    .app-title span {{
        background: linear-gradient(135deg, {accent}, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .app-subtitle {{
        color: {text2};
        font-size: 0.875rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}
    .app-subtitle::before {{
        content: '';
        display: inline-block;
        width: 6px; height: 6px;
        background: {success};
        border-radius: 50%;
        box-shadow: 0 0 6px {success};
    }}

    /* ── Welcome card ── */
    .welcome-card {{
        background: {bg2};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 2.75rem 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }}
    .welcome-card::before {{
        content: '';
        position: absolute;
        top: -60px; left: 50%;
        transform: translateX(-50%);
        width: 300px; height: 200px;
        background: radial-gradient(ellipse, {accent}20 0%, transparent 70%);
        pointer-events: none;
    }}
    .welcome-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 0.4rem;
        letter-spacing: -0.4px;
    }}
    .welcome-subtitle {{
        color: {text2};
        margin-bottom: 2rem;
        font-size: 0.9rem;
    }}
    .features-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
    }}
    .feature-item {{
        background: {bg3};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: {text};
        display: flex;
        align-items: center;
        gap: 0.6rem;
        text-align: left;
        transition: border-color 0.2s, transform 0.2s;
    }}
    .feature-item:hover {{
        border-color: {accent}60;
        transform: translateY(-1px);
    }}
    .feature-icon {{
        width: 30px; height: 30px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }}
    .fi-yellow {{ background: rgba(245,158,11,0.15); }}
    .fi-blue   {{ background: rgba(59,130,246,0.15); }}
    .fi-purple {{ background: rgba(139,92,246,0.15); }}
    .fi-green  {{ background: rgba(16,185,129,0.15); }}

    /* ── Conversation starter ── */
    .starter-card {{
        background: linear-gradient(135deg, {accent}15, {accent}05);
        border: 1px solid {accent}35;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }}
    .starter-title {{
        font-size: 0.75rem;
        font-weight: 700;
        color: {accent};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }}
    .starter-db {{
        font-size: 0.875rem;
        color: {text};
        margin-bottom: 0.25rem;
    }}
    .starter-table-list {{
        font-size: 0.8rem;
        color: {text2};
    }}

    /* ── Example chips label ── */
    .example-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: {text2};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }}

    /* ── Response card ── */
    .response-card {{
        background: {bg2};
        border: 1px solid {border};
        border-radius: 14px;
        overflow: hidden;
        margin-top: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .response-meta {{
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.55rem 1rem;
        background: {bg3};
        border-bottom: 1px solid {border};
        font-size: 0.75rem;
        color: {text2};
    }}
    .response-meta-item {{
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }}
    .response-body {{ padding: 1rem; }}

    /* ── SQL block ── */
    .sql-block {{
        background: {code_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 0.875rem 1rem;
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: {accent};
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.6;
    }}
    .sql-label {{
        font-size: 0.68rem;
        font-weight: 700;
        color: {text2};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }}

    /* ── SQL Preview card ── */
    .sql-preview-card {{
        background: {bg2};
        border: 2px solid {accent}45;
        border-radius: 14px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 0 20px {accent}10;
    }}
    .sql-preview-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {accent};
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* ── Status badges ── */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.775rem;
        font-weight: 500;
    }}
    .status-connected {{
        background: rgba(16,185,129,0.12);
        color: {success};
        border: 1px solid rgba(16,185,129,0.3);
    }}
    .status-disconnected {{
        background: rgba(239,68,68,0.08);
        color: {danger};
        border: 1px solid rgba(239,68,68,0.25);
    }}

    /* ── Hallucination alert ── */
    .hallucination-alert {{
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.35);
        border-left: 3px solid {warning};
        border-radius: 8px;
        padding: 0.65rem 1rem;
        color: {warning};
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}

    /* ── Sidebar section labels ── */
    .sidebar-section {{
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {text2};
        margin: 1rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {{
        background: {bg2};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}

    /* ── Chat input glow ── */
    [data-testid="stChatInput"] textarea {{
        border-radius: 14px !important;
        border-color: {border} !important;
        background-color: {input_bg} !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 3px {accent}20 !important;
    }}

    hr {{ border-color: {border}; margin: 0.875rem 0; opacity: 0.6; }}

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background-color: {input_bg};
        color: {text};
        border-color: {border};
        border-radius: 9px;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.2s;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 2px {accent}20 !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: {border};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {text2}; }}
    </style>
    """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    c1, c2 = st.columns(2)
    with c1:
        lang = st.selectbox(
            "lang", options=["en", "de"],
            index=0 if st.session_state.language == "en" else 1,
            key="lang_select", label_visibility="collapsed",
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
    with c2:
        dark = st.toggle(t("dark_mode"), value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

    st.divider()
    st.markdown(f'<p class="sidebar-section">🗄️ {t("sidebar_add_db")}</p>',
                unsafe_allow_html=True)

    db_alias = st.text_input("alias", placeholder=t("conn_name_placeholder"),
                             label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1:
        host = st.text_input("host", value="localhost",
                             label_visibility="collapsed", placeholder="Host")
    with c2:
        port = st.number_input("port", value=5432, min_value=1,
                               max_value=65535, label_visibility="collapsed")

    dbname = st.text_input("dbname", placeholder=t("db_name"),
                           label_visibility="collapsed")
    c3, c4 = st.columns(2)
    with c3:
        user = st.text_input("user", value="postgres",
                             label_visibility="collapsed", placeholder="User")
    with c4:
        password = st.text_input("pwd", type="password",
                                 label_visibility="collapsed", placeholder="••••••••")

    if st.button(f"🔌 {t('btn_connect')}", use_container_width=True):
        if not db_alias:
            st.error("Please provide a connection name.")
        else:
            with st.spinner(f"Connecting '{db_alias}'..."):
                success = st.session_state.multi_connector.add_connection(
                    name=db_alias, host=host, port=int(port),
                    dbname=dbname, user=user, password=password,
                )
                if success:
                    try:
                        st.session_state.agent = DataAnalystAgent(
                            st.session_state.multi_connector)
                        st.session_state.hallucination_detector = HallucinationDetector(
                            st.session_state.multi_connector)
                        st.session_state.connected = True
                        st.session_state.conversation_started = False
                        st.success(f"✅ {t('connect_success')}: {db_alias}")
                        st.rerun()
                    except Exception as agent_err:
                        import traceback
                        print("[AGENT INIT ERROR]", traceback.format_exc(), flush=True)
                        st.error(f"❌ Agent-Initialisierung fehlgeschlagen: {agent_err}")
                else:
                    st.error(f"❌ {t('connect_error')}")

    st.divider()
    names = st.session_state.multi_connector.get_connection_names()
    st.markdown(f'<p class="sidebar-section">🟢 {t("active_connections")}</p>',
                unsafe_allow_html=True)

    if names:
        for name in names:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f'<span class="status-badge status-connected">● {name}</span>',
                    unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{name}"):
                    st.session_state.multi_connector.remove_connection(name)
                    if len(st.session_state.multi_connector) == 0:
                        st.session_state.connected = False
                        st.session_state.agent = None
                        st.session_state.conversation_started = False
                    st.rerun()
            with st.expander(f"📋 {t('tables')}", expanded=False):
                conn = st.session_state.multi_connector.get_connector(name)
                if conn:
                    for tbl in conn.get_table_names():
                        st.caption(f"• {tbl}")
    else:
        st.markdown(
            f'<span class="status-badge status-disconnected">● {t("no_connections")}</span>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<p class="sidebar-section">📂 {t("upload_title")}</p>',
                unsafe_allow_html=True)
    if st.session_state.connected and names:
        target_db = st.selectbox(t("upload_target"), names,
                                 label_visibility="collapsed")
        custom_name = st.text_input(t("upload_table"),
                                    placeholder=t("upload_table"),
                                    label_visibility="collapsed")
        uploaded_file = st.file_uploader(
            "file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file and st.button(f"⬆️ {t('upload_btn')}",
                                       use_container_width=True):
            with st.spinner("Importing..."):
                try:
                    conn = st.session_state.multi_connector.get_connector(target_db)
                    result = process_upload(uploaded_file, conn,
                                           custom_table_name=custom_name or None)
                    st.success(
                        f"✅ {t('import_success')}: '{result['table_name']}' "
                        f"({result['rows_imported']} rows)")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {t('import_error')}: {e}")
    else:
        st.caption(t("connect_first"))

    st.divider()
    st.markdown(f'<p class="sidebar-section">📜 {t("history_title")}</p>',
                unsafe_allow_html=True)
    history = st.session_state.query_history.get_session_history()
    st.caption(f"{len(history)} {t('history_queries')}")
    if st.session_state.ratings:
        avg = sum(st.session_state.ratings.values()) / len(st.session_state.ratings)
        st.caption(f"⭐ {t('avg_rating')}: {avg:.1f}/5")

    c1, c2 = st.columns(2)
    with c1:
        if st.button(t("btn_csv"), use_container_width=True):
            st.session_state.query_history.export_to_csv()
            st.success("✅")
    with c2:
        if st.button(t("btn_excel"), use_container_width=True):
            st.session_state.query_history.export_to_excel()
            st.success("✅")
    if st.button(f"📈 {t('btn_stats')}", use_container_width=True):
        st.json(st.session_state.query_history.get_statistics())

    st.divider()
    if st.button(f"🔄 {t('btn_reset')}", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ratings = {}
        st.session_state.conversation_started = False
        st.session_state.pending_sql = None
        st.session_state.pending_question = None
        if st.session_state.agent:
            st.session_state.agent.reset_memory()
        st.session_state.query_history.clear_session()
        st.rerun()

    st.markdown(f'<p class="sidebar-section">🤖 {t("llm_settings")}</p>',
                unsafe_allow_html=True)
    provider = os.getenv("LLM_PROVIDER", "anthropic").upper()
    model_name = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    st.caption(f"{provider} · {model_name}")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="app-header">'
    f'<span style="font-size:2rem;">🧠</span>'
    f'<div class="app-title">Data<span>Mind</span></div>'
    f'</div>'
    f'<p class="app-subtitle">{t("app_subtitle")}</p>',
    unsafe_allow_html=True,
)

# ── Welcome screen (no DB connected) ─────────────────────────────────────────
if not st.session_state.connected and not st.session_state.messages:
    st.markdown(
        f"""
        <div class="welcome-card">
            <div class="welcome-title">{t("welcome_title")}</div>
            <p class="welcome-subtitle">{t("welcome_subtitle")}</p>
            <div class="features-grid">
                <div class="feature-item">
                    <div class="feature-icon fi-yellow">⚡</div>
                    {t("welcome_feature1")}
                </div>
                <div class="feature-item">
                    <div class="feature-icon fi-blue">📊</div>
                    {t("welcome_feature2")}
                </div>
                <div class="feature-item">
                    <div class="feature-icon fi-purple">🔮</div>
                    {t("welcome_feature3")}
                </div>
                <div class="feature-item">
                    <div class="feature-icon fi-green">🗄️</div>
                    {t("welcome_feature4")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

# ── Conversation starter (DB just connected, no messages yet) ─────────────────
if (st.session_state.connected and not st.session_state.messages
        and names):

    # Build summary of all connected databases
    starter_lines = []
    for db_name in names:
        conn = st.session_state.multi_connector.get_connector(db_name)
        if conn:
            tables = conn.get_table_names()
            starter_lines.append(
                f"<div class='starter-db'>🗄️ <strong>{db_name}</strong> "
                f"— {len(tables)} {t('tables_found')}: "
                f"{', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}</div>"
            )

    st.markdown(
        f"""
        <div class="starter-card">
            <div class="starter-title">🧠 {t("conversation_starter")}</div>
            {''.join(starter_lines)}
            <div class="starter-table-list" style="margin-top:0.5rem;">
                {t("conversation_starter_prompt")}
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

    # Example question chips
    st.markdown(f'<p class="example-label">{t("example_questions")}</p>',
                unsafe_allow_html=True)
    examples = EXAMPLE_QUESTIONS.get(st.session_state.language,
                                     EXAMPLE_QUESTIONS["en"])
    cols = st.columns(len(examples))
    for i, (col, question) in enumerate(zip(cols, examples)):
        with col:
            if st.button(question, key=f"ex_{i}", use_container_width=True):
                st.session_state.prefill_question = question
                st.session_state.conversation_started = True
                st.rerun()

# ── SQL Preview modal ─────────────────────────────────────────────────────────
if st.session_state.pending_sql and st.session_state.pending_question:
    st.markdown(
        f"""
        <div class="sql-preview-card">
            <div class="sql-preview-title">🔍 {t("sql_preview_title")}</div>
            <p style="font-size:0.85rem; margin-bottom:0.5rem;">{t("sql_preview_msg")}</p>
        </div>
        """, unsafe_allow_html=True,
    )
    edited_sql = st.text_area(
        "SQL", value=st.session_state.pending_sql,
        height=100, label_visibility="collapsed",
    )
    c1, c2, c3 = st.columns([2, 2, 6])
    with c1:
        if st.button(t("sql_run"), type="primary", use_container_width=True):
            st.session_state.pending_sql = None
            prompt_to_run = st.session_state.pending_question
            st.session_state.pending_question = None
            st.session_state.prefill_question = prompt_to_run
            st.rerun()
    with c2:
        if st.button(t("sql_cancel"), use_container_width=True):
            st.session_state.pending_sql = None
            st.session_state.pending_question = None
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Response card with metadata
            resp_time = message.get("response_time", "")
            db_used = message.get("db_used", "")
            sql = message.get("sql_used", "")

            meta_html = f"""
            <div class="response-card">
                <div class="response-meta">
                    {"<div class='response-meta-item'>⏱ " + str(resp_time) + "s</div>" if resp_time else ""}
                    {"<div class='response-meta-item'>🗄️ " + db_used + "</div>" if db_used else ""}
                </div>
                <div class="response-body">
            """
            st.markdown(meta_html, unsafe_allow_html=True)
            st.markdown(message["content"])

            if sql:
                st.markdown(
                    f'<div class="sql-label">🔍 {t("sql_used")}</div>'
                    f'<div class="sql-block">{sql}</div>',
                    unsafe_allow_html=True,
                )

            if message.get("hallucination_warning"):
                st.markdown(
                    f'<div class="hallucination-alert">'
                    f'{message["hallucination_warning"]}</div>',
                    unsafe_allow_html=True,
                )

            if message.get("chart_path") and os.path.exists(message["chart_path"]):
                st.image(message["chart_path"], use_container_width=True)

            if message.get("table_data") is not None:
                df = pd.DataFrame(message["table_data"])
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 {t('download_csv')}",
                    data=csv,
                    file_name="results.csv",
                    mime="text/csv",
                    key=f"dl_{message.get('id')}",
                )

            st.markdown("</div></div>", unsafe_allow_html=True)

            # Star rating
            msg_id = message.get("id")
            existing = st.session_state.ratings.get(msg_id)
            if existing:
                st.caption(f"{'⭐' * existing} ({existing}/5)")
            else:
                st.caption(t("rate_answer"))
                rating = st.feedback("stars", key=f"rating_{msg_id}")
                if rating is not None:
                    st.session_state.ratings[msg_id] = rating + 1
                    st.session_state.query_history.update_feedback(
                        msg_id, str(rating + 1))
                    st.rerun()
        else:
            st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = None

chat_input = st.chat_input(
    t("chat_placeholder"),
    disabled=not st.session_state.connected,
)
if chat_input:
    prompt = chat_input
elif st.session_state.prefill_question:
    prompt = st.session_state.prefill_question
    st.session_state.prefill_question = None

# ── Process prompt ────────────────────────────────────────────────────────────
print(f"[DEBUG2] chat_input={repr(chat_input)} prompt={repr(prompt)} connected={st.session_state.connected} agent={st.session_state.agent is not None}", flush=True)
if prompt and st.session_state.connected and not st.session_state.pending_sql:
    msg_id = len(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t("analyzing")):
            start_time = time.time()

            # Capture stdout to extract SQL from agent tool calls
            import io as _io
            _stdout_capture = _io.StringIO()
            _old_stdout = sys.stdout
            sys.stdout = _stdout_capture

            response = st.session_state.agent.run(prompt)

            sys.stdout = _old_stdout
            _captured = _stdout_capture.getvalue()
            print(_captured, end="", flush=True)  # still show in terminal

            response_time = round(time.time() - start_time, 2)
            # Normalize response to string
            if isinstance(response, list):
                parts = []
                for item in response:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                response = "\n".join(parts) if parts else str(response)
            elif not isinstance(response, str):
                response = str(response)

            chart_path = None
            hallucination_warning = None
            hallucination_detected = None
            table_data = None
            # Try to get SQL from response text first, then from captured stdout
            sql_used = extract_first_sql(response) or extract_first_sql(_captured)

            # Extract SQL from terminal output too (agent uses multi_db_query_tool)
            import glob as _glob
            png_files = sorted(_glob.glob("*.png"), key=os.path.getmtime, reverse=True)
            if png_files:
                chart_path = png_files[0]

            if chart_path and os.path.exists(chart_path):
                st.image(chart_path, use_container_width=True)

            # Hallucination check
            if st.session_state.hallucination_detector and names:
                report = st.session_state.hallucination_detector.verify_response(
                    question=prompt, response=response, db_name=names[0],
                )
                if report.get("verified") is False:
                    hallucination_warning = t("hallucination_warning")
                hallucination_detected = report.get("verified") is False

            # Render response
            try:
                data = json.loads(response)
                if isinstance(data, list) and len(data) > 0:
                    table_data = data
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 {t('download_csv')}",
                        data=csv, file_name="results.csv",
                        mime="text/csv", key=f"dl_new_{msg_id}",
                    )
                elif isinstance(data, dict) and "historical" in data:
                    st.caption(f"Model: `{data.get('model_used', 'N/A')}`")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.dataframe(pd.DataFrame(data["historical"]),
                                     use_container_width=True)
                    with c2:
                        st.dataframe(
                            pd.DataFrame({"forecast": data["forecast"]}),
                            use_container_width=True)
                else:
                    st.markdown(response)
            except Exception:
                st.markdown(response)

            # Show SQL used
            if sql_used:
                st.markdown(
                    f'<div class="sql-label">🔍 {t("sql_used")}</div>'
                    f'<div class="sql-block">{sql_used}</div>',
                    unsafe_allow_html=True,
                )

            if hallucination_warning:
                st.markdown(
                    f'<div class="hallucination-alert">'
                    f'{hallucination_warning}</div>',
                    unsafe_allow_html=True,
                )

            # Rating
            st.caption(t("rate_answer"))
            rating = st.feedback("stars", key=f"rating_new_{msg_id}")
            if rating is not None:
                st.session_state.ratings[msg_id] = rating + 1

            # Log
            st.session_state.query_history.log(
                question=prompt, response=response,
                response_time=response_time, db_names=names,
                hallucination_detected=hallucination_detected,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "chart_path": chart_path,
            "hallucination_warning": hallucination_warning,
            "sql_used": sql_used,
            "response_time": response_time,
            "db_used": names[0] if names else "",
            "table_data": table_data,
            "id": msg_id,
        })
        st.rerun()