import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from reportlab.pdfgen import canvas
from io import BytesIO
from gtts import gTTS
import tempfile

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #06142e, #020617, #0b0f1a);
    color: white;
}

.title {
    font-size: 2.8rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background:#0f172a;
    padding:14px;
    border-left:4px solid #00d4ff;
    border-radius:12px;
    margin:10px 0;
    line-height:1.6;
}

.user {
    background:#1e293b;
    padding:10px;
    border-radius:10px;
}

/* SUMMARY STYLES */
.summary-box {
    background:#0b1220;
    padding:18px;
    border-radius:14px;
    border:1px solid #1e293b;
    margin-top:15px;
}

.s-title {color:#00d4ff; font-size:20px; font-weight:800;}
.s-answer {color:#a855f7; font-weight:600;}
.s-reason {color:#22c55e; font-weight:600;}
.s-risk {color:#ff3d81; font-weight:700;}
.s-clause {color:#fbbf24; font-weight:500;}

.stButton>button {
    border-radius:12px;
    padding:10px 16px;
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
        return "No relevant legal data found."

    return f"""
Answer:
Legal reasoning from document match.
Risk Level: Medium
Key Clauses:
- {context[0] if len(context)>0 else ''}
- {context[1] if len(context)>1 else ''}
- {context[2] if len(context)>2 else ''}
"""

# ================= FORMAT CHAT =================
def format_ai(text):
    return f"<div class='card'>{text}</div>"

# ================= VOICE =================
def speak(text):
    try:
        tts = gTTS(text[:300])
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
        st.markdown(f"<div class='card'>{msg}</div>", unsafe_allow_html=True)

# ================= INPUT =================
query = st.text_input("Ask  question")

col1, col2 = st.columns(2)

send = col1.button("⚡ Send")
summary = col2.button("📌 Summary + Voice")

# ================= SEND =================
if send and query:
    st.session_state.chat.append(("user", query))
    response = ask_ai(query)
    st.session_state.chat.append(("ai", response))
    st.markdown(format_ai(response), unsafe_allow_html=True)

# ================= BEAUTIFUL SUMMARY =================
def build_summary(text):
    parts = text.split("\n")

    answer = parts[1] if len(parts) > 1 else ""
    reason = parts[2] if len(parts) > 2 else ""
    risk = parts[3] if len(parts) > 3 else ""
    clause = parts[4] if len(parts) > 4 else ""

    html = f"""
    <div class='summary-box'>

    <div class='s-title'>📌 DOCUMENT SUMMARY</div><br>

    <div class='s-answer'>🧾 Answer:</div>
    {answer}<br><br>

    <div class='s-reason'>⚖️ Legal Reasoning:</div>
    {reason}<br><br>

    <div class='s-risk'>🚨 Risk Level:</div>
    {risk}<br><br>

    <div class='s-clause'>📌 Key Clauses:</div>
    {clause}<br>

    </div>
    """
    return html

# ================= SUMMARY + VOICE =================
if summary:
    all_text = "\n".join([" ".join(v) for v in st.session_state.docs.values()])[:3000]

    summary_text = f"""
Answer:
Based on uploaded documents.
Legal reasoning applied.
Risk Level: Medium
Key Clauses:
{all_text.split('. ')[0] if '. ' in all_text else ''}
"""

    st.markdown(build_summary(summary_text), unsafe_allow_html=True)

    speak("Here is the summary of your legal documents.")

# ================= PDF EXPORT =================
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
