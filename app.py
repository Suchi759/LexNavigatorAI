import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")  # safer than 2.5
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PAGE =================
st.set_page_config(page_title="LexNavigator AI ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1220, #050814, #02030a);
    color: #e5e7eb;
}

h1 {
    text-align:center;
    font-size: 3rem;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.user {
    background:#1e293b;
    padding:10px;
    border-radius:12px;
    margin:6px;
}

.ai {
    background:linear-gradient(135deg,#0f172a,#1e293b);
    padding:10px;
    border-radius:12px;
    border-left:3px solid #00d4ff;
    margin:6px;
}

.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LexNavigator AI PRO (VOICE + STABLE)")

# ================= MEMORY =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "docs" not in st.session_state:
    st.session_state.docs = {}

# ================= PDF SAFE READ =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for p in reader.pages[:10]:  # LIMIT PAGES (FIX CRASH)
            t = p.extract_text()
            if t:
                text += t
        return text[:3000]  # LIMIT SIZE (IMPORTANT)
    except:
        return ""

def chunk(text):
    return [c for c in text.split(". ") if len(c) > 20][:30]  # LIMIT CHUNKS

# ================= SAFE RETRIEVAL =================
def retrieve(query, chunks):
    if not chunks:
        return []

    q = embedder.encode([query])[0]
    c = embedder.encode(chunks)

    scores = np.dot(c, q)
    top = np.argsort(scores)[-3:][::-1]  # LIMIT 3 ONLY

    return [chunks[i] for i in top]

# ================= VOICE INPUT =================
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return ""

# ================= VOICE OUTPUT =================
def speak(text):
    tts = gTTS(text)
    file_path = tempfile.mktemp(suffix=".mp3")
    tts.save(file_path)
    return file_path

# ================= AI SAFE CALL =================
def ask_ai(query):
    try:
        all_chunks = []
        for d in st.session_state.docs.values():
            all_chunks.extend(d)

        context = retrieve(query, all_chunks)
        context = "\n".join(context)[:1200]

        prompt = f"""
You are a legal AI assistant.

Give:
1. Answer
2. Legal reasoning
3. Risk level
4. Key clauses

Context:
{context}

Question:
{query}
"""

        time.sleep(1)  # prevent quota spam

        return model.generate_content(prompt).text

    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ================= SIDEBAR =================
st.sidebar.title("📂 Document Vault")

files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

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
        st.markdown(f"<div class='ai'>⚖️ {msg}</div>", unsafe_allow_html=True)

# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Ask legal question")

with col2:
    if st.button("🎤 Voice Input"):
        query = voice_input()
        st.write("You said:", query)

# ================= SEND =================
if st.button("⚡ Send") and query:

    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    st.markdown(f"<div class='ai'>{response}</div>", unsafe_allow_html=True)

    audio_file = speak(response)
    st.audio(audio_file)

# ================= FILE COMPARE =================
st.markdown("---")
st.subheader("📄 Compare Documents")

f1 = st.file_uploader("OLD PDF", type=["pdf"])
f2 = st.file_uploader("NEW PDF", type=["pdf"])

if f1 and f2:
    old = extract_pdf(f1)
    new = extract_pdf(f2)

    prompt = f"""
Compare legal documents:

OLD:
{old[:1200]}

NEW:
{new[:1200]}

Show:
- Changes
- Added clauses
- Removed clauses
- Risk
"""

    if st.button("Compare"):
        res = model.generate_content(prompt).text
        st.write(res)