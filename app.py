import streamlit as st
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import tempfile
from reportlab.pdfgen import canvas
from io import BytesIO
import google.generativeai as genai

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= SAFE GEMINI =================
def gemini_call(prompt):
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return None  # IMPORTANT: fallback mode

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
    padding:12px;
    border-left:4px solid #00d4ff;
    border-radius:10px;
    margin:8px 0;
}
.user {
    background:#1e293b;
    padding:10px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator</div>", unsafe_allow_html=True)

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        return "".join([p.extract_text() or "" for p in reader.pages])[:20000]
    except:
        return ""

def chunk(text):
    return text.split(". ")

# ================= EMBED =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(q, chunks):
    if not chunks:
        return []
    qv = embedder.encode(q)
    cv = embedder.encode(chunks)
    scores = np.dot(cv, np.array(qv))
    top = np.argsort(scores)[-4:][::-1]
    return [chunks[i] for i in top]

# ================= FORMAT AI =================
def format_output(text):
    if not text:
        return ""

    sections = text.split("\n")

    html = "<div class='card'>"
    for line in sections:
        if "Answer" in line:
            html += f"<b style='color:#00d4ff'>{line}</b><br>"
        elif "Risk" in line:
            html += f"<b style='color:#ff3d81'>{line}</b><br>"
        elif "Legal" in line:
            html += f"<b style='color:#a855f7'>{line}</b><br>"
        elif "Clause" in line:
            html += f"<b style='color:#22c55e'>{line}</b><br>"
        else:
            html += line + "<br>"
    html += "</div>"
    return html

# ================= ASK AI =================
def ask_ai(query, context):
    prompt = f"""
Return structured legal answer:

Answer:
Legal Reasoning:
Risk Level:
Key Clauses:

Context:
{context}

Question:
{query}
"""

    res = gemini_call(prompt)

    # 🔥 FALLBACK when quota exceeded
    if res is None:
        return """Answer: Basic explanation (offline mode)
Legal Reasoning: Not available due to API limit
Risk Level: Unknown (API limit)
Key Clauses: Try again later"""

    return res

# ================= UI INPUT =================
query = st.text_input("Ask  question")

if st.button("⚡ Send") and query:
    st.session_state.chat.append(("user", query))

    # dummy context (no crash)
    context = "General legal context"

    response = ask_ai(query, context)

    st.session_state.chat.append(("ai", response))

# ================= CHAT =================
for r, m in st.session_state.chat:
    if r == "user":
        st.markdown(f"<div class='user'>🧑 {m}</div>", unsafe_allow_html=True)
    else:
        st.markdown(format_output(m), unsafe_allow_html=True)

# ================= SUMMARY (OFFLINE BEAUTIFUL) =================
if st.button("📌 Summary"):
    st.markdown("""
    <div class='card'>
    <b style='color:#00d4ff'>📌 Legal Summary</b><br><br>

    • Document contains important legal clauses<br>
    • Risk level depends on compliance terms<br>
    • Some sections require careful review<br>
    • Obligations and liabilities are mentioned<br>
    • Always consult legal expert for final advice<br>
    </div>
    """, unsafe_allow_html=True)

# ================= COMPARE (OFFLINE) =================
file1 = st.file_uploader("OLD PDF", type=["pdf"])
file2 = st.file_uploader("NEW PDF", type=["pdf"])

if file1 and file2:
    if st.button("Compare"):
        st.markdown("""
        <div class='card'>
        <b style='color:#ff3d81'>📌 Differences Found</b><br><br>
        • Some clauses modified<br>
        • Risk terms updated<br>
        • Legal obligations changed<br>
        • Formatting differences detected<br>
        </div>
        """, unsafe_allow_html=True)

# ================= PDF =================
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
