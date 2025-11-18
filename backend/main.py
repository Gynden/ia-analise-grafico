import os
import io
import base64
import json
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY nao encontrada nas variaveis de ambiente!")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnaliseResponse(BaseModel):
    acao: str
    confianca: float
    justificativa: str


# -------------------------------------------------------------------------
# FUNCAO PARA PREPARAR IMAGEM
# -------------------------------------------------------------------------
def preparar_imagem(upload: UploadFile) -> str:
    """
    Converte a imagem enviada em JPEG comprimido e retorna em base64
    no formato data URL (data:image/jpeg;base64,...).
    """
    img = Image.open(upload.file).convert("RGB")

    max_width = 1280
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# -------------------------------------------------------------------------
# ROTA ROOT (TESTE)
# -------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "API de analise de grafico rodando"}


# -------------------------------------------------------------------------
# ROTA PRINCIPAL: /api/analisar
# --------------------------------
