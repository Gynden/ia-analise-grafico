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
# -------------------------------------------------------------------------
@app.post("/api/analisar", response_model=AnaliseResponse)
async def analisar_imagem(image: UploadFile = File(...)):
    if not image or not image.content_type or not image.content_type.startswith("image/"):
        logging.warning("Arquivo invalido recebido.")
        return AnaliseResponse(
            acao="NAO_OPERAR",
            confianca=0.0,
            justificativa="Arquivo invalido. Envie um print do grafico."
        )

    try:
        # Prepara imagem
        img_b64 = preparar_imagem(image)

        # Prompt do sistema (sem acentos pra evitar problema de encoding)
        sistema = """
Você é um trader profissional de opções binárias, especialista em M1.
Seu trabalho é analisar APENAS o gráfico enviado e decidir se vale a pena
entrar comprado (COMPRAR), vendido (VENDER) ou se é melhor NÃO OPERAR.

Contexto:
- Timeframe: 1 minuto (M1)
- Estilo: scalp rápido, operações curtas
- Objetivo: pegar movimentos com boa probabilidade, evitando entrada aleatória

Regras importantes:
1) Só marcar COMPRAR ou VENDER se o cenário estiver muito claro.
2) Se o mercado estiver lateral, confuso ou sem direção forte, responda NAO_OPERAR.
3) Dê preferência a:
   - Tendência forte bem definida (sequência de candles na mesma direção).
   - Pullback claro (correção contra a tendência batendo em região importante).
   - Rompimentos bem definidos (quebra de suporte/resistência com força).
4) Evite operar perto de regiões de briga (muita sombra, pavio grande, indecisão).
5) Você deve ser conservador: se estiver em dúvida, responda NAO_OPERAR.

Formato de resposta:
Responda APENAS um JSON neste formato exato:

{
  "acao": "COMPRAR" | "VENDER" | "NAO_OPERAR",
  "confianca": numero entre 0.0 e 1.0,
  "justificativa": "texto curto em portugues explicando o motivo"
}
"""

        usuario_texto = (
            "Analise esse grafico de M1 e siga as regras do manual. "
            "Lembre-se: seja conservador. Se o cenario nao estiver muito claro, responda NAO_OPERAR."
        )

        logging.info(f"[{datetime.utcnow()}] Iniciando analise na Groq...")

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=300,
            messages=[
                {"role": "system", "content": sistema},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usuario_texto},
                        {"type": "image_url", "image_url": {"url": img_b64}},
                    ],
                },
            ],
        )

        raw = completion.choices[0].message.content
        logging.info(f"Resposta bruta da Groq: {raw}")

        data = json.loads(raw)

        acao = str(data.get("acao", "NAO_OPERAR")).upper().strip()
        if acao not in ("COMPRAR", "VENDER", "NAO_OPERAR"):
            acao = "NAO_OPERAR"

        try:
            confianca = float(data.get("confianca", 0.0))
        except Exception:
            confianca = 0.0

        justificativa = str(data.get("justificativa", "")).strip()
        if not justificativa:
            justificativa = "Cenario indefinido. Melhor nao operar."

        logging.info(
            f"SAIDA FINAL -> acao={acao}, confianca={confianca:.2f}, "
            f"justificativa={justificativa[:120]}"
        )

        return AnaliseResponse(
            acao=acao,
            confianca=confianca,
            justificativa=justificativa
        )

    except Exception as e:
        logging.exception("Erro ao analisar grafico com a Groq")
        return AnaliseResponse(
            acao="NAO_OPERAR",
            confianca=0.0,
            justificativa="Erro ao analisar grafico. Tente novamente mais tarde."
        )
