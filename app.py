import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from reportlab.pdfgen import canvas
from io import BytesIO
from gtts import gTTS
import tempfile
import time
import matplotlib.pyplot as plt
from datetime import datetime

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #06142e, #020617, #0b0f1a);
    color: white;
}
.title {
    font-size: 2.6rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.user {
    background: linear-gradient(90deg,#1e293b,#334155);
    padding:10px;
    border-radius:12px;
    margin:6px;
    text-align:right;
    color:#a7f3d0;
}
.ai {
    background: linear-gradient(90deg,#0f172a,#111827);
    padding:10px;
    border-radius:12px;
    margin:6px;
    border-left:3px solid #00d4ff;
    color:#e0f2fe;
}
.card {
    background:#0f172a;
    padding:14px;
    border-radius:12px;
    border-left:4px solid #00d4ff;
    margin:10px 0;
    box-shadow:0px 0px 10px rgba(0,212,255,0.4);
}
.stButton>button {
    border-radius:14px;
    padding:10px 16px;
    font-weight:700;
    transition:0.3s;
    color:white;
    box-shadow:0px 0px 12px rgba(0,212,255,0.6);
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow:0px 0px 20px rgba(255,61,129,0.8);
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator</div>", unsafe_allow_html=True)

# ================= SESSION =================
if "users" not in st.session_state:
    st.session_state.users = {}

username = st.sidebar.text_input("👤 Enter your name", "Guest")
if username not in st.session_state
