import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from gtts import gTTS
import tempfile
from reportlab.pdfgen import canvas
from io import BytesIO
import google.generativeai as genai

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= SAFE GEMINI (ONLY FOR SEND) =================
@st.cache_data(show_spinner=False)
def cached_gemini(prompt):
    return model.generate_content(prompt).text

def safe_gemini(prompt):
    try:
        return cached_gemini(prompt)
    except Exception as e:
        return "⚠️ Limit reached or API error."

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #06142e, #020617, #0b0f1a);
    color: white;
}
.title {
    font-size: 3rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.chat-user {
    background:#1e293b;
    padding:10px;
    border-radius:10px;
    margin:5px;
}
.chat-ai {
    background:#0f172a;
    border-left:3px solid #00d4ff;
    padding:10px;
    border-radius:10px;
    margin:5px;
}
.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:10px;
    padding:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator</div>", unsafe_allow_html=True)

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "docs" not in st.session_state:
    st.session_state.docs = {}

# ================= LANGUAGE =================
lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])

def lang_rule():
    if lang == "Hindi":
        return "Answer ONLY in Hindi."
    elif lang == "Telugu":
        return "Answer ONLY in Telugu."
    return "Answer ONLY in English."

# ================= PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t
        return text[:20000]
    except:
        return ""

def chunk(text):
    return text.split(". ")

# ================= EMBEDDING =================
def retrieve(query, chunks):
    if not chunks:
        return []
    q = embedder.encode(query)
    c = embedder.encode(chunks)
    q = np.array(q)
    scores = np.dot(c, q)
    top = np.argsort(scores)[-4:][::-1]
    return [chunks[i] for i in top]

# ================= TTS =================
def speak(text):
    try:
        tts = gTTS(text[:300])
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(path)
        st.audio(path)
    except:
        pass

# ================= FORMAT =================
def format_ai(text):
    text = text.replace("🧾", "<span style='color:#00d4ff'>🧾</span>")
    text = text.replace("⚖️", "<span style='color:#a855f7'>⚖️</span>")
    text = text.replace("🚨", "<span style='color:#ff3d81'>🚨</span>")
    text = text.replace("📌", "<span style='color:#22c55e'>📌</span>")
    return text

# ================= SIDEBAR =================
st.sidebar.title("📚 Document Vault")

files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    for f in files:
        reader = PdfReader(f)
        text = "".join([p.extract_text() or "" for p in reader.pages])
        st.session_state.docs[f.name] = chunk(text[:20000])

st.sidebar.write("Stored Docs:")
for name in st.session_state.docs:
    st.sidebar.write("📄", name)

st.sidebar.markdown("---")
st.sidebar.write("🌐 Language:", lang)

# ================= CHAT HISTORY =================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>{format_ai(msg)}</div>", unsafe_allow_html=True)

# ================= AI (ONLY API USE HERE) =================
def ask_ai(query):
    all_chunks = []
    for d in st.session_state.docs.values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    prompt = f"""
{lang_rule()}

You are a legal AI assistant.

Return STRICT FORMAT:
🧾 Answer: max 4 lines
⚖️ Legal Reasoning: max 3 lines
🚨 Risk Level: Low/Medium/High
📌 Key Clauses: max 4 bullets

RULES:
- Max 120 words
- Be concise

Context:
{chr(10).join(context)}

Question:
{query}
"""

    return safe_gemini(prompt)

# ================= INPUT =================
query = st.text_input("Ask legal question...")

if st.button("⚡ Send") and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    st.markdown(f"<div class='chat-ai'>{format_ai(response)}</div>", unsafe_allow_html=True)

    speak(response)

# ================= LOCAL SUMMARY (NO API) =================
if st.button("📌 Summary (Offline)"):
    all_text = "\n".join([" ".join(v) for v in st.session_state.docs.values()])[:3000]

    sentences = all_text.split(". ")[:6]

    st.write("📌 Summary:")
    for s in sentences:
        st.write("•", s)

# ================= LOCAL COMPARE (NO API) =================
file1 = st.file_uploader("OLD PDF", type=["pdf"])
file2 = st.file_uploader("NEW PDF", type=["pdf"])

if file1 and file2:
    t1 = extract_pdf(file1)
    t2 = extract_pdf(file2)

    if st.button("Compare (Offline)"):
        a = set(t1.split())
        b = set(t2.split())

        st.write("📌 Added:")
        st.write(list(b - a)[:30])

        st.write("📌 Removed:")
        st.write(list(a - b)[:30])

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
