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
# (Windows) kalau perlu, set path tesseract.exe:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Proposal Generator")
templates = Jinja2Templates(directory="app/templates")

client = OpenAI()  # pakai OPENAI_API_KEY dari ENV

# -------- Utils --------
def clean_text(s: str, limit: int = 8000) -> str:
    s = s.replace("\x0c", "").strip()
    # batasi panjang agar prompt ke LLM tidak kebanyakan
    return s[:limit]

def detect_lang_id(text: str) -> str:
    """Deteksi kasar: 'id' atau 'en'."""
    t = text.lower()
    id_signals = ["kami mencari", "membutuhkan", "lowongan", "anggaran", "durasi", "garansi","contoh","desain","brief","portofolio","upwork"]
    en_signals = ["we are looking", "seeking", "requirements", "deliverables", "budget",
                  "hourly", "fixed price", "milestone", "scope", "portfolio", "upwork"]
    id_hits = sum(s in t for s in id_signals)
    en_hits = sum(s in t for s in en_signals)
    return "id" if id_hits >= en_hits else "en"

def looks_like_jobpost(text: str) -> bool:
    """
    Heuristik ringan untuk mendeteksi teks job post.
    Syarat: panjang minimal + adanya ≥2 kata kunci relevan.
    """
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < 80:  # terlalu pendek untuk job post
        return False

    # Keyword EN + ID
    kw = [
        # English
        "we are looking", "we're looking", "seeking", "hire", "hiring",
        "requirements", "responsibilities", "deliverables", "scope",
        "timeline", "budget", "hourly", "fixed price", "milestone",
        "experience", "portfolio", "please include", "proposals",
        # Indonesian
        "kami mencari", "kami membutuhkan", "dibutuhkan", "lowongan",
        "kualifikasi", "tanggung jawab", "deliverable", "ruang lingkup",
        "jangka waktu", "anggaran", "harian", "per jam", "harga tetap",
        "pengalaman", "portofolio", "cantumkan", "kirim proposal"
    ]
    hits = sum(1 for k in kw if k in t)

    # Sinyal tambahan: ada email/url/angka uang/format pekerjaan
    extra_signals = 0
    if re.search(r"\$|usd|idr|rp\s?\d", t):  # ada indikasi budget
        extra_signals += 1
    if re.search(r"https?://|\bupwork\b|\bfiverr\b|\bfreelancer\.com\b", t):
        extra_signals += 1
    if re.search(r"\b(full[- ]?time|part[- ]?time|contract|freelance)\b", t):
        extra_signals += 1

    return hits >= 2 or (hits >= 1 and extra_signals >= 1)

def ocr_image_bytes(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, lang=OCR_LANG or "eng")

def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 8) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        # 1) coba text extraction asli
        t = page.get_text("text") or ""
        if not t.strip():
            # 2) fallback: render jadi bitmap lalu OCR
            pix = page.get_pixmap(dpi=200)
            img_b = pix.tobytes("png")
            t = ocr_image_bytes(img_b)
        texts.append(t)
    return "\n".join(texts)

def build_prompt(job_text: str) -> list:
    """
    Prompt baru: 3 langkah (Penjelasan ID, Proposal EN <=110 words, Terjemahan ID),
    dengan format output yang HARUS persis:
    - Penjelasan Job Post:
    - Proposal:
    - Terjemahan Proposal:
    """
    system_msg = (
        "You are a precise assistant that reads Upwork job posts and returns exactly three sections. "
        "Follow the rules strictly, keep things concise and professional, and NEVER add extra headings or commentary."
    )

    user_msg = f"""
Your task is to perform three steps based on the Upwork job post I provide. Follow these rules carefully:

---
🔹 STEP 1: Jelaskan isi job post (dalam Bahasa Indonesia)
Gunakan bahasa Indonesia yang jelas, padat, dan mudah dimengerti.

---
🔹 STEP 2: Write the proposal (in English)
Write a proposal in English based on the job post. Follow these instructions:
1. Make the proposal concise and professional — min 2 short paragraphs, under 150 words.
2. Always start with: Hi,
3. Then create a natural, non-repetitive opening sentence based on brand visibility:
   - If the brand is NOT mentioned: write a sentence that expresses your willingness to create a free sample if the client can provide more details or share their brand — but avoid repeating the same sentence every time.
   - If the brand IS mentioned: write a sentence that says you’ve already created a design sample based on their brand — but phrase it differently each time.
4. After the opening, briefly mention relevant skills, offer, or how you collaborate.
5. End with: Best regards

---
🔹 STEP 3: Terjemahkan proposal ke dalam Bahasa Indonesia
Terjemahkan isi proposal dari Step 2 ke dalam Bahasa Indonesia. Gunakan bahasa yang tetap sopan, profesional, dan mudah dipahami klien lokal.

---
Format your output exactly like this (no extra text, no markdown, no emojis):

Penjelasan Job Post: [isi Step 1 - Bahasa Indonesia]
Proposal: [isi Step 2 - English]
Terjemahan Proposal: [isi Step 3 - Bahasa Indonesia]

---
Here is the job post:
{job_text}
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def parse_llm_output(raw: str) -> dict:
    """
    Parse LLM output into three parts. Return dict with keys:
    'explanation', 'proposal', 'translation'.
    Fallback: if parsing fails, put whole output into 'proposal' and leave others empty.
    """
    # normalize whitespace but keep line breaks for paragraphs
    text = raw.strip()

    # Try robust regex: capture after each heading up to next heading or end
    patterns = {
        'explanation': r"Penjelasan\s*Job\s*Post\s*:\s*(.*?)\s*(?=Proposal\s*:|Terjemahan\s*Proposal\s*:|$)",
        'proposal': r"Proposal\s*:\s*(.*?)\s*(?=Terjemahan\s*Proposal\s*:|Penjelasan\s*Job\s*Post\s*:|$)",
        'translation': r"Terjemahan\s*Proposal\s*:\s*(.*)$"
    }

    result = {'explanation': '', 'proposal': '', 'translation': ''}

    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            # strip and collapse multiple blank lines
            extracted = re.sub(r'\n{3,}', '\n\n', m.group(1).strip())
            result[k] = extracted

    # If nothing parsed, attempt simple split by lines with headings (defensive)
    if not any(result.values()):
        # attempt line-by-line splitting by headings
        parts = re.split(r"(Penjelasan\s*Job\s*Post\s*:|Proposal\s*:|Terjemahan\s*Proposal\s*:)", text, flags=re.IGNORECASE)
        # parts contains separators; reconstruct simple parse
        try:
            # naive mapping: find indexes of headings and following content
            joined = "".join(parts).strip()
            # fallback: put everything into 'proposal'
            result['proposal'] = text
        except Exception:
            result['proposal'] = text

    # final cleanup: ensure strings are not ridiculously long (safety)
    for k in result:
        if len(result[k]) > 20000:
            result[k] = result[k][:20000] + "\n\n...(truncated)"

    return result

def generate_proposal(job_text: str) -> dict:
    """
    Returns dict with keys: explanation, proposal, translation.
    If error occurs, returns those keys with an error message in 'proposal'.
    """
    msgs = build_prompt(job_text)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.5,
            max_tokens=600,
        )
        out = resp.choices[0].message.content.strip()
        # Keep original line breaks for better parsing
        parsed = parse_llm_output(out)
        # If proposal empty but the whole output is non-empty, put fallback
        if not parsed['proposal'] and out:
            # try to salvage by assuming second line/paragraph is proposal
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', out) if p.strip()]
            if len(paragraphs) >= 2:
                parsed['explanation'] = parsed['explanation'] or paragraphs[0]
                parsed['proposal'] = paragraphs[1]
                parsed['translation'] = parsed['translation'] or (paragraphs[2] if len(paragraphs) > 2 else "")
            else:
                parsed['proposal'] = out  # put entire text to proposal as last resort

        return parsed
    except Exception as e:
        err_msg = f"(LLM error) {e}"
        return {'explanation': '', 'proposal': err_msg, 'translation': ''}

# -------- Routes --------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocr", response_class=HTMLResponse)
async def ocr(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    ctype = (file.content_type or "").lower()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # OCR
    if ctype.startswith("image/"):
        extracted = ocr_image_bytes(data)
    elif ctype in ("application/pdf", "pdf", "application/octet-stream"):
        extracted = ocr_pdf_bytes(data)
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ctype}")

    extracted = clean_text(extracted)

    # === NEW: validasi apakah ini job post ===
    if not looks_like_jobpost(extracted):
        lang = detect_lang_id(extracted)
        if lang == "id":
            explanation = "Maaf, aku tidak bisa mendeteksi job post dari gambar/file ini."
            proposal = ""
            translation = ""
        else:
            explanation = "Sorry, I couldn’t detect a job post from this image/file."
            proposal = ""
            translation = ""
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "extracted": extracted,
                "explanation": explanation,
                "proposal": proposal,
                "translation": translation
            }
        )

    # Jika valid job post → generate proposal normal (mengembalikan dict)
    parsed = generate_proposal(extracted) if extracted else {'explanation': '', 'proposal': '(No text detected.)', 'translation': ''}

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "extracted": extracted,
            "explanation": parsed.get('explanation', ''),
            "proposal": parsed.get('proposal', ''),
            "translation": parsed.get('translation', '')
        }
    )
