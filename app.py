import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from gtts import gTTS
import tempfile
import time
from reportlab.pdfgen import canvas
from io import BytesIO

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
    font-size: 3rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CHAT */
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

/* BUTTON */
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

# ================= EMBEDDING SEARCH =================
def retrieve(query, chunks):
    if not chunks:
        return []

    q = embedder.encode([query])[0]
    c = embedder.encode(chunks)

    scores = np.dot(c, q)
    top = np.argsort(scores)[-4:][::-1]

    return [chunks[i] for i in top]

# ================= TEXT TO SPEECH =================
def speak(text):
    tts = gTTS(text[:300])
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(path)
    st.audio(path)

# ================= COLOR FORMAT =================
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
        text = extract_pdf(f)
        st.session_state.docs[f.name] = chunk(text)

st.sidebar.write("Stored Docs:")
for name in st.session_state.docs:
    st.sidebar.write("📄", name)

# ================= CHAT HISTORY =================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>{format_ai(msg)}</div>", unsafe_allow_html=True)

# ================= AI ENGINE =================
def ask_ai(query):
    all_chunks = []
    for d in st.session_state.docs.values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    prompt = f"""
You are a legal AI assistant.

Return STRICTLY in this format:

🧾 Answer: max 5-10 lines
⚖️ Legal Reasoning: max 5 lines
🚨 Risk Level: Low / Medium / High + 1 line reason
📌 Key Clauses: max 4 bullet points

RULES:
- Be extremely concise
- No long paragraphs
- Max 150 words total

Context:
{context}

Question:
{query}
"""

    try:
        return model.generate_content(prompt).text
    except:
        return "⚠️ Error"

# ================= INPUT =================
query = st.text_input("Ask legal question...")

if st.button("⚡ Send") and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    st.markdown(f"<div class='chat-ai'>{format_ai(response)}</div>", unsafe_allow_html=True)

    speak(response)

# ================= SUMMARY =================
if st.button("📌 Summary"):
    all_text = "\n".join([" ".join(v) for v in st.session_state.docs.values()])[:3000]
    prompt = "Summarize legal document in 5 bullet points:\n" + all_text

    try:
        st.write(model.generate_content(prompt).text)
    except:
        st.warning("Limit reached")

# ================= COMPARE =================
file1 = st.file_uploader("OLD PDF", type=["pdf"])
file2 = st.file_uploader("NEW PDF", type=["pdf"])

if file1 and file2:
    t1 = extract_pdf(file1)
    t2 = extract_pdf(file2)

    if st.button("Compare"):
        prompt = f"""
Compare OLD vs NEW contract:

OLD:
{t1[:2000]}

NEW:
{t2[:2000]}

Give only key differences in bullet points.
"""
        try:
            st.write(model.generate_content(prompt).text)
        except:
            st.warning("Too large")

# ================= PDF REPORT =================
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
