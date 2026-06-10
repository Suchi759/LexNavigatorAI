import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from reportlab.pdfgen import canvas
from io import BytesIO

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

/* CARDS */
.card {
    background:#0f172a;
    padding:14px;
    border-left:4px solid #00d4ff;
    border-radius:12px;
    margin:10px 0;
    line-height:1.6;
}

/* USER */
.user {
    background:#1e293b;
    padding:10px;
    border-radius:10px;
}

/* BUTTON STYLING */
.stButton>button {
    border-radius:12px;
    padding:10px 16px;
    font-weight:600;
    border:none;
    transition:0.3s;
}

/* SEND BUTTON */
div[data-testid="stButton"]:nth-of-type(1) button {
    background: linear-gradient(90deg,#00d4ff,#0077ff);
    color:white;
}

/* SUMMARY BUTTON */
div[data-testid="stButton"]:nth-of-type(2) button {
    background: linear-gradient(90deg,#a855f7,#7c3aed);
    color:white;
}

/* COMPARE BUTTON */
div[data-testid="stButton"]:nth-of-type(3) button {
    background: linear-gradient(90deg,#ff3d81,#ff006e);
    color:white;
}

/* DOWNLOAD BUTTON */
div[data-testid="stButton"]:nth-of-type(4) button {
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator </div>", unsafe_allow_html=True)

# ================= SESSION =================
if "docs" not in st.session_state:
    st.session_state.docs = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

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

# ================= AI ENGINE =================
def ask_ai(query):
    all_chunks = []
    for d in st.session_state.docs.values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    if not context:
        return """
🧾 Answer: No relevant legal data found
⚖️ Legal Reasoning: No matching clauses in uploaded documents
🚨 Risk Level: Low
📌 Key Clauses: None
"""

    return f"""
🧾 Answer:
The query matches your legal documents.

⚖️ Legal Reasoning:
Semantic search found relevant legal clauses.

🚨 Risk Level:
Medium (requires review)

📌 Key Clauses:
• {context[0] if len(context)>0 else 'N/A'}
• {context[1] if len(context)>1 else 'N/A'}
• {context[2] if len(context)>2 else 'N/A'}
• {context[3] if len(context)>3 else 'N/A'}
"""

# ================= FORMAT OUTPUT =================
def format_ai(text):
    sections = text.split("\n")
    html = "<div class='card'>"

    for line in sections:
        if "Answer" in line:
            html += f"<b style='color:#00d4ff'>🧾 {line}</b><br>"
        elif "Legal" in line:
            html += f"<b style='color:#a855f7'>⚖️ {line}</b><br>"
        elif "Risk" in line:
            html += f"<b style='color:#ff3d81'>🚨 {line}</b><br>"
        elif "Key" in line or "Clause" in line:
            html += f"<b style='color:#22c55e'>📌 {line}</b><br>"
        else:
            html += line + "<br>"

    html += "</div>"
    return html

# ================= SIDEBAR =================
st.sidebar.title("📚 Document Vault")

files = st.sidebar.file_uploader("📤 Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    for f in files:
        text = extract_pdf(f)
        st.session_state.docs[f.name] = chunk(text)

st.sidebar.write("Stored Docs:")
for name in st.session_state.docs:
    st.sidebar.write("📄", name)

# ================= CHAT =================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(format_ai(msg), unsafe_allow_html=True)

# ================= INPUT =================
query = st.text_input("Ask legal question")

col1, col2, col3, col4 = st.columns(4)

# ================= SEND =================
with col1:
    send = st.button("⚡ Send")

# ================= SUMMARY =================
with col2:
    summary = st.button("📌 Summary")

# ================= COMPARE =================
with col3:
    compare = st.button("⚖️ Compare")

# ================= DOWNLOAD =================
with col4:
    download = st.button("⬇ Download Report")

# ================= ACTIONS =================
if send and query:
    st.session_state.chat.append(("user", query))
    response = ask_ai(query)
    st.session_state.chat.append(("ai", response))
    st.markdown(format_ai(response), unsafe_allow_html=True)

# SUMMARY (OFFLINE)
if summary:
    all_text = "\n".join([" ".join(v) for v in st.session_state.docs.values()])[:3000]

    st.markdown("""
    <div class='card'>
    <b style='color:#00d4ff'>📌 Document Summary</b><br><br>
    """, unsafe_allow_html=True)

    for s in all_text.split(". ")[:6]:
        st.write("•", s)

    st.markdown("</div>", unsafe_allow_html=True)

# COMPARE (OFFLINE)
file1 = st.file_uploader("OLD PDF", type=["pdf"])
file2 = st.file_uploader("NEW PDF", type=["pdf"])

if file1 and file2 and compare:
    t1 = extract_pdf(file1)
    t2 = extract_pdf(file2)

    a = set(t1.split())
    b = set(t2.split())

    st.markdown("""
    <div class='card'>
    <b style='color:#ff3d81'>📌 Differences Found</b><br><br>
    """, unsafe_allow_html=True)

    st.write("➕ Added:", list(b - a)[:30])
    st.write("➖ Removed:", list(a - b)[:30])

    st.markdown("</div>", unsafe_allow_html=True)

# DOWNLOAD
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
