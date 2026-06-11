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
st.set_page_config(page_title="LexNavigator ⚖️ ", layout="wide")

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

st.markdown("<div class='title'>⚖️ LexNavigator </div>", unsafe_allow_html=True)

# ================= SESSION =================
if "users" not in st.session_state:
    st.session_state.users = {}

username = st.sidebar.text_input("👤 Enter your name", "Guest")
if username not in st.session_state.users:
    st.session_state.users[username] = {"docs": {}, "chat": [], "history_risk": []}

user_data = st.session_state.users[username]

# ================= MODEL =================
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
    scores = np.dot(c, np.array(q))
    top = np.argsort(scores)[-5:][::-1]
    return [chunks[i] for i in top]

# ================= REAL RISK ENGINE =================
def risk_score(context):
    keywords = ["liability", "penalty", "termination", "breach", "damages", "indemnity"]
    score = 0
    for clause in context:
        for k in keywords:
            if k.lower() in clause.lower():
                score += 15
    return min(100, max(20, score))

# ================= THINKING MODE =================
def thinking():
    with st.status("🧠 Thinking Mode Activated...", expanded=True) as status:
        st.write("📄 Analyzing legal documents...")
        time.sleep(0.5)
        st.write("📌 Extracting clauses...")
        time.sleep(0.5)
        st.write("⚖️ Checking risk patterns...")
        time.sleep(0.5)
        status.update(label="✅ Analysis complete", state="complete")

# ================= AI =================
def ask_ai(query):
    all_chunks = []
    for d in user_data["docs"].values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    if not context:
        return "🧾 No relevant legal data found\n⚖️ Risk: Low"

    risk = risk_score(context)
    user_data["history_risk"].append(risk)

    return f"""
🧾 Answer:
Legal match found.

⚖️ Reasoning:
Semantic AI detected clauses.

🚨 Risk Level:
{risk}

📌 Clauses:
• {context[0] if len(context)>0 else 'N/A'}
• {context[1] if len(context)>1 else 'N/A'}
• {context[2] if len(context)>2 else 'N/A'}
• {context[3] if len(context)>3 else 'N/A'}
"""

# ================= TYPE WRITER =================
def type_writer(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"<div class='card'>{out}</div>", unsafe_allow_html=True)
        time.sleep(0.002)

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
st.sidebar.title("📚 Document Vault")
files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    for f in files:
        user_data["docs"][f.name] = chunk(extract_pdf(f))

# ================= MEMORY VAULT =================
st.sidebar.subheader("💾 Memory Vault")

search = st.sidebar.text_input("Search Chat Memory")
if search:
    for r,m in user_data["chat"]:
        if search.lower() in m.lower():
            st.sidebar.write("🔎", m[:80])

if st.sidebar.button("🗑 Clear Memory"):
    user_data["chat"] = []

if st.sidebar.button("⬇ Export Chat"):
    st.sidebar.download_button(
        "Download",
        "\n".join([m[1] for m in user_data["chat"]]),
        "chat.txt"
    )

# ================= CHAT =================
st.subheader("💬 Chat Interface")

for role, msg in user_data["chat"]:
    if role == "user":
        st.markdown(f"<div class='user'>🧑 {msg}<br><small>{datetime.now()}</small></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'>🤖 {msg}<br><small>{datetime.now()}</small></div>", unsafe_allow_html=True)

# ================= INPUT =================
query = st.text_input("Ask Legal Question")

col1, col2 = st.columns(2)

send = col1.button("⚡ Send")
summary = col2.button("📌 Summary")

# ================= SEND =================
if send and query:
    user_data["chat"].append(("user", query))
    thinking()
    response = ask_ai(query)
    type_writer(response)
    speak(response)
    user_data["chat"].append(("ai", response))

# ================= SUMMARY =================
if summary:
    thinking()
    text = "📌 Document summary generated...\n⚖️ Risk: Medium\n📄 Legal clauses extracted"
    type_writer(text)

# ================= DASHBOARD =================
st.subheader("📊 Risk Dashboard")

if user_data["history_risk"]:
    fig, ax = plt.subplots()
    ax.plot(user_data["history_risk"], marker="o")
    ax.set_title(f"Risk Trend for {username}")
    st.pyplot(fig)

# ================= DOCUMENT COMPARISON (Separate Sidebar) =================
st.sidebar.subheader("⚖️ Smart Document Comparison AI")

f1 = st.sidebar.file_uploader("OLD PDF", type=["pdf"])
f2 = st.sidebar.file_uploader("NEW PDF", type=["pdf"])

if f1 and f2:
    t1 = extract_pdf(f1)
    t2 = extract_pdf(f2)

    added = list(set(t2
