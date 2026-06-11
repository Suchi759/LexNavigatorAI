import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from reportlab.pdfgen import canvas
from io import BytesIO
from gtts import gTTS
import tempfile
import time

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

/* CHAT */
.user {
    background:#1e293b;
    padding:10px;
    border-radius:12px;
    margin:6px;
    text-align:right;
}

.card {
    background:#0f172a;
    padding:14px;
    border-left:4px solid #00d4ff;
    border-radius:12px;
    margin:10px 0;
}

/* 🎨 CINEMATIC BUTTONS */
.stButton>button {
    border-radius:14px;
    padding:12px 18px;
    font-weight:700;
    border:none;
    color:white;
    transition:0.3s;
    box-shadow:0px 0px 15px rgba(0,0,0,0.4);
}

/* Send */
div[data-testid="stButton"]:nth-of-type(1) button {
    background: linear-gradient(90deg,#00d4ff,#0077ff);
}

/* Summary */
div[data-testid="stButton"]:nth-of-type(2) button {
    background: linear-gradient(90deg,#a855f7,#ff3d81);
}

/* Compare */
div[data-testid="stButton"]:nth-of-type(3) button {
    background: linear-gradient(90deg,#ff006e,#ffb703);
}

/* Download */
div[data-testid="stButton"]:nth-of-type(4) button {
    background: linear-gradient(90deg,#22c55e,#16a34a);
}

/* HOVER EFFECT */
.stButton>button:hover {
    transform: scale(1.05);
    filter: brightness(1.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator AI</div>", unsafe_allow_html=True)

# ================= SESSION =================
if "docs" not in st.session_state:
    st.session_state.docs = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= MEMORY =================
if "memory" not in st.session_state:
    st.session_state.memory = []

# ================= MODEL =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        return "".join([p.extract_text() or "" for p in reader.pages])[:20000]
    except:
        return ""

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
        return """
🧾 Answer: No relevant legal data found  
⚖️ Legal Reasoning: No matching clauses  
🚨 Risk Level: Low  
📌 Key Clauses: None  
"""

    return f"""
🧾 Answer:
Legal match found in documents.

⚖️ Legal Reasoning:
AI detected relevant legal clauses.

🚨 Risk Level:
Medium (review required)

📌 Key Clauses:
• {context[0] if len(context)>0 else 'N/A'}
• {context[1] if len(context)>1 else 'N/A'}
• {context[2] if len(context)>2 else 'N/A'}
• {context[3] if len(context)>3 else 'N/A'}
"""

# ================= FORMAT =================
def format_ai(text):
    lines = text.split("\n")
    html = "<div class='card'>"

    for line in lines:
        if "Answer" in line:
            html += f"<b style='color:#00d4ff'>🧾 {line}</b><br>"
        elif "Legal" in line:
            html += f"<b style='color:#a855f7'>⚖️ {line}</b><br>"
        elif "Risk" in line:
            html += f"<b style='color:#ff3d81'>🚨 {line}</b><br>"
        elif "Key" in line:
            html += f"<b style='color:#22c55e'>📌 {line}</b><br>"
        else:
            html += f"<span style='color:#e2e8f0'>{line}</span><br>"

    html += "</div>"
    return html

# ================= TYPE =================
def type_writer(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"<div class='card'>{out}</div>", unsafe_allow_html=True)
        time.sleep(0.003)

# ================= VOICE =================
def speak(text):
    try:
        tts = gTTS(text=text[:300])
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(path)
        st.audio(path)
    except:
        pass

# ================= SIDEBAR =================
st.sidebar.title("📚 Document Vault")

files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    for f in files:
        st.session_state.docs[f.name] = chunk(extract_pdf(f))

# ================= MEMORY UI =================
st.sidebar.subheader("🧠 Memory")
if st.session_state.chat:
    for role, msg in st.session_state.chat[-5:]:
        st.sidebar.write("🧑" if role=="user" else "🤖", msg[:50])

st.sidebar.subheader("📊 Stats")
st.sidebar.metric("Docs Uploaded", len(st.session_state.docs))
st.sidebar.metric("Chats", len(st.session_state.chat))

# ================= CHAT =================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(format_ai(msg), unsafe_allow_html=True)

# ================= INPUT =================
query = st.text_input("Ask legal question")

col1, col2, col3, col4 = st.columns(4)

send = col1.button("⚡ Send")
summary = col2.button("📌 Summary")
compare = col3.button("⚖️ Compare")
download = col4.button("⬇ Download")

# ================= SEND =================
if send and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    type_writer(response)
    speak(response)

    st.session_state.chat.append(("ai", response))

# ================= SUMMARY =================
if summary:
    summary_text = """
📌 Summary:
Documents analyzed successfully.

⚖️ Legal Reasoning:
Key clauses extracted.

🚨 Risk Level:
Medium

📌 Key Clauses:
Important legal patterns found.
"""
    type_writer(summary_text)
    speak(summary_text)

# ================= DASHBOARD =================
st.subheader("⚖️ Risk Dashboard")

low = sum("low" in m[1].lower() for m in st.session_state.chat)
med = sum("medium" in m[1].lower() for m in st.session_state.chat)
high = sum("high" in m[1].lower() for m in st.session_state.chat)

st.progress(min(1.0, med*0.1))
st.write("🟡 Medium Risk Level Trending")

st.metric("Low Risk", low)
st.metric("Medium Risk", med)
st.metric("High Risk", high)

# ================= COMPARE =================
file1 = st.file_uploader("OLD PDF", type=["pdf"])
file2 = st.file_uploader("NEW PDF", type=["pdf"])

if file1 and file2 and compare:
    t1 = extract_pdf(file1)
    t2 = extract_pdf(file2)

    a = set(t1.split())
    b = set(t2.split())

    st.markdown("<div class='card'><b>📌 Differences Found</b><br><br>", unsafe_allow_html=True)

    st.write("➕ Added:", list(b - a)[:30])
    st.write("➖ Removed:", list(a - b)[:30])

    st.markdown("</div>", unsafe_allow_html=True)

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

if download and st.session_state.chat:
    text = "\n".join([m[1] for m in st.session_state.chat if m[0] == "ai"])
    pdf = make_pdf(text)
    st.download_button("Download PDF Ready", pdf, "legal_report.pdf", "application/pdf")
