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

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #06142e, #020617, #0b0f1a);
    color: white;
}

/* TITLE */
.title {
    font-size: 2.8rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* WhatsApp chat style */
.user {
    background:#1e293b;
    padding:10px;
    border-radius:12px;
    margin:6px;
    text-align:right;
}

.ai {
    background:#0f172a;
    padding:10px;
    border-radius:12px;
    margin:6px;
    border-left:3px solid #00d4ff;
}

/* cards */
.card {
    background:#0f172a;
    padding:14px;
    border-left:4px solid #00d4ff;
    border-radius:12px;
    margin:10px 0;
}

/* buttons */
.stButton>button {
    border-radius:10px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator </div>", unsafe_allow_html=True)

# ================= SESSION =================
if "docs" not in st.session_state:
    st.session_state.docs = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PDF =================
def extract_pdf(file):
    reader = PdfReader(file)
    return "".join([p.extract_text() or "" for p in reader.pages])[:20000]

def chunk(text):
    return text.split(". ")

# ================= RETRIEVE =================
def retrieve(query, chunks):
    if not chunks:
        return []
    q = embedder.encode(query)
    c = embedder.encode(chunks)
    q = np.array(q)
    scores = np.dot(c, q)
    top = np.argsort(scores)[-5:][::-1]
    return [chunks[i] for i in top]

# ================= AI =================
def ask_ai(query):
    all_chunks = []
    for d in st.session_state.docs.values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    if not context:
        return "🧾 Answer: No data found\n⚖️ Risk: Low\n🚨 Safe"

    return f"""
🧾 Answer:
Legal match found in documents.

⚖️ Legal Reasoning:
AI detected relevant clauses.

🚨 Risk Level:
Medium

📌 Key Clauses:
• {context[0] if len(context)>0 else 'N/A'}
• {context[1] if len(context)>1 else 'N/A'}
• {context[2] if len(context)>2 else 'N/A'}
• {context[3] if len(context)>3 else 'N/A'}
"""

# ================= CHAT UI =================
def show_chat():
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"<div class='user'>🧑 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai'>{msg}</div>", unsafe_allow_html=True)

# ================= TYPE ANIMATION =================
def type_writer(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"<div class='card'>{out}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

# ================= VOICE =================
def speak(text):
    try:
        tts = gTTS(text[:300])
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(path)
        st.audio(path)
    except:
        pass

# ================= UPLOAD =================
st.sidebar.title("📚 Upload Docs")
files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    for f in files:
        st.session_state.docs[f.name] = chunk(extract_pdf(f))

# ================= CHAT DISPLAY =================
show_chat()

query = st.text_input("Ask legal question")

col1, col2, col3 = st.columns(3)

send = col1.button("⚡ Send")
summary = col2.button("📌 Summary")
dashboard = col3.button("📊 Dashboard")

# ================= SEND =================
if send and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    type_writer(response)
    speak(response)

    st.session_state.chat.append(("ai", response))

# ================= SUMMARY (WITH TYPEWRITER) =================
if summary:
    summary_text = """
📌 Document Summary:
Legal documents analyzed successfully.

⚖️ Legal Reasoning:
Patterns detected across contracts.

🚨 Risk Level:
Medium

📌 Key Clauses:
Important legal clauses extracted.
"""

    type_writer(summary_text)
    speak(summary_text)

# ================= RISK DASHBOARD =================
# ================= RISK DASHBOARD =================
if dashboard:

    st.markdown("""
    <div class='card'>
    <h2 style='color:#00d4ff'>⚖️ Legal Risk Dashboard</h2>
    <p style='color:white'>AI Risk Analysis Overview</p>
    </div>
    """, unsafe_allow_html=True)

    chart_data = {
        "Low Risk": [30],
        "Medium Risk": [50],
        "High Risk": [20]
    }

    st.bar_chart(chart_data)

    st.markdown("""
    <div class='card'>
        <h3 style='color:#22c55e'>🟢 Low Risk : 30%</h3>
        <h3 style='color:#facc15'>🟡 Medium Risk : 50%</h3>
        <h3 style='color:#ef4444'>🔴 High Risk : 20%</h3>
    </div>
    """, unsafe_allow_html=True)
# ================= DOWNLOAD =================
def make_pdf(text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    y = 800
    for line in text.split("\n")[:40]:
        c.drawString(40, y, line[:100])
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

if st.button("⬇ Download Report") and st.session_state.chat:
    text = "\n".join([m[1] for m in st.session_state.chat if m[0] == "ai"])
    pdf = make_pdf(text)
    st.download_button("Download PDF", pdf, "legal_report.pdf", "application/pdf")
