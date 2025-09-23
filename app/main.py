# app/main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import pytesseract, io, os, textwrap
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()

# ---- Konfigurasi ----
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OCR_LANG = os.getenv("OCR_LANG", "eng")

app = FastAPI(title="Proposal Generator")
templates = Jinja2Templates(directory="app/templates")

# -------- Helpers --------
def get_openai() -> OpenAI:
    """
    Lazy init OpenAI client supaya app tidak crash saat start
    kalau OPENAI_API_KEY belum ada.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Biarkan /health tetap hidup; endpoint yang butuh AI akan error jelas
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")
    return OpenAI(api_key=key)

def clean_text(s: str, limit: int = 8000) -> str:
    s = s.replace("\x0c", "").strip()
    return s[:limit]

def detect_lang_id(text: str) -> str:
    t = text.lower()
    id_signals = ["kami mencari", "membutuhkan", "lowongan", "anggaran", "durasi", "garansi","contoh","desain","brief","portofolio","upwork"]
    en_signals = ["we are looking", "seeking", "requirements", "deliverables", "budget",
                  "hourly", "fixed price", "milestone", "scope", "portfolio", "upwork"]
    id_hits = sum(s in t for s in id_signals)
    en_hits = sum(s in t for s in en_signals)
    return "id" if id_hits >= en_hits else "en"

def looks_like_jobpost(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < 80:
        return False
    kw = [
        "we are looking","we're looking","seeking","hire","hiring","requirements","responsibilities",
        "deliverables","scope","timeline","budget","hourly","fixed price","milestone","experience",
        "portfolio","please include","proposals",
        "kami mencari","kami membutuhkan","dibutuhkan","lowongan","kualifikasi","tanggung jawab",
        "deliverable","ruang lingkup","jangka waktu","anggaran","harian","per jam","harga tetap",
        "pengalaman","portofolio","cantumkan","kirim proposal"
    ]
    hits = sum(1 for k in kw if k in t)
    extra = 0
    if re.search(r"\$|usd|idr|rp\s?\d", t): extra += 1
    if re.search(r"https?://|\bupwork\b|\bfiverr\b|\bfreelancer\.com\b", t): extra += 1
    if re.search(r"\b(full[- ]?time|part[- ]?time|contract|freelance)\b", t): extra += 1
    return hits >= 2 or (hits >= 1 and extra >= 1)

def ocr_image_bytes(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, lang=OCR_LANG or "eng")

def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 8) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        t = page.get_text("text") or ""
        if not t.strip():
            pix = page.get_pixmap(dpi=200)
            img_b = pix.tobytes("png")
            t = ocr_image_bytes(img_b)
        texts.append(t)
    return "\n".join(texts)

def build_prompt(job_text: str) -> list:
    sys = (
        "You are a proposal writer for freelance graphic design jobs. "
        "Write in the same language as the job text (Indonesian or English). "
        "One concise paragraph only. No greetings, no personal name, no fluff. "
        "If the job text contains a design brief or brand details, include a short 'free sample' concept (describe what you'd deliver). "
        "If it doesn't contain a brief, ask for the minimal brief you need (very concise). "
        "Do NOT use links or placeholders. Be specific and actionable."
    )
    user = f"Job post text:\n{job_text}\n\nWrite the proposal now."
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def generate_proposal(job_text: str) -> str:
    msgs = build_prompt(job_text)
    try:
        client = get_openai()  # <- di-init di sini, bukan global
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.4,
            max_tokens=220,
        )
        out = resp.choices[0].message.content.strip()
        return " ".join(out.splitlines()).strip()
    except HTTPException:
        raise
    except Exception as e:
        return f"(LLM error) {e}"

# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocr", response_class=HTMLResponse)
async def ocr(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    ctype = (file.content_type or "").lower()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if ctype.startswith("image/"):
        extracted = ocr_image_bytes(data)
    elif ctype in ("application/pdf", "pdf", "application/octet-stream"):
        extracted = ocr_pdf_bytes(data)
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ctype}")

    extracted = clean_text(extracted)

    if not looks_like_jobpost(extracted):
        lang = detect_lang_id(extracted)
        proposal = ("Maaf, aku tidak bisa mendeteksi job post dari gambar/file ini."
                    if lang == "id" else
                    "Sorry, I couldn’t detect a job post from this image/file.")
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "extracted": extracted, "proposal": proposal}
        )

    proposal = generate_proposal(extracted) if extracted else "(No text detected.)"
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "extracted": extracted, "proposal": proposal}
    )
