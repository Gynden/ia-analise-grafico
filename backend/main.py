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

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO BÁSICA
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY não encontrada nas variáveis de ambiente!")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# MODELOS DE RESPOSTA
# -----------------------------------------------------------------------------
class AnaliseResponse(BaseModel):
    acao: str
    confianca: float
    justificativa: str


# -----------------------------------------------------------------------------
# FUNÇÃO PARA PREPARAR A IMAGEM
# -----------------------------------------------------------------------------
def preparar_imagem(upload: UploadFile) -> str:
    """
    Converte a imagem enviada em JPEG comprimido e retorna em base64
    no formato data URL (data:image/jpeg;base64,...).
    """
    img = Image.open(upload.file).convert("RGB")

    # Reduz um pouco o tamanho pra ficar leve no Render
    max_width = 1280
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# -----------------------------------------------------------------------------
# ROTA ROOT (SÓ PRA TESTE RÁPIDO)
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "API de análise de gráfico rodando"}


# -----------------------------------------------------------------------------
# ROTA PRINCIPAL: /api/analisar
# -----------------------------------------------------------------------------
@app.post("/api/analisar", response_model=AnaliseResponse)
async def analisar_imagem(image: UploadFile = File(...)):
    # Validação básica
    if not image or not image.content_type or not image.content_type.startswith("image/"):
        logging.warning("Arquivo inválido recebido.")
        return AnaliseResponse(
            acao="NAO_OPERAR",
            confianca=0.0,
            justificativa="Arquivo inválido. Envie um print do gráfico."
        )

    try:
        # Prepara imagem
        img_b64 = preparar_imagem(image)

        # Prompt focado em scalp M1
        sistema = (
            "Você é um trader profissional de opções binárias, especialista em M1. "
            "Seu trabalho é analisar APENAS o gráfico enviado e decidir se vale a pena "
            "entrar comprado (COMPRAR), vendido (VENDER) ou se é melhor NÃO OPERAR.\n\n"
            "Contexto:\n"
            "- Timeframe: 1 minuto (M1)\n"
            "- Estilo: scalp rápido, operações curtas\n"
            "- Objetivo: pegar movimentos com boa probabilidade, evitando entrada aleatória\n\n"
            "Regras importantes:\n"
            "1) Só marcar COMPRAR ou VENDER se o cenário estiver muito claro.\n"
            "2) Se o mercado estiver lateral, confuso ou sem direção forte, responda NAO_OPERAR.\n"
            "3) Dê preferência a:\n"
            "   - Tendência forte bem definida (sequência de candles na mesma direção).\n"
            "   - Pullback claro (correção contra a tendência batendo em região importante).\n"
            "   - Rompimentos bem definidos (quebra de suporte/resistência com força).\n"
            "4) Evite operar perto de regiões de briga (muita sombra, pavio grande, indecisão).\n"
            "5) Você deve ser conservador: se estiver em dúvida, responda NAO_OPERAR.\n\n"
            "Formato de resposta: responda APENAS um JSON com este formato exato:\n"
            "{\n"
            '  \"acao\": \"COMPRAR\" | \"VENDER\" | \"NAO_OPERAR\",\n'
            '  \"confianca\": número entre 0.0 e 1.0,\n'
            '  \"justificativa\": \"texto curto em português explicando o motivo\"\n"
            "}\n"
        )

        usuario_texto = (
            "Analise esse gráfico de M1 e siga as regras do seu manual. "
            "Lembre-se: seja conservador. Se o cenário não estiver muito claro, responda NAO_OPERAR."
        )

        logging.info(f"[{datetime.utcnow()}] Iniciando análise na Groq...")

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},  # força JSON
            temperature=0.2,  # mais conservador / menos aleatório
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
            justificativa = "Cenário indefinido. Melhor não operar."

        # Log simples pra gente analisar depois nos logs do Render
        logging.info(
            f"SAÍDA FINAL -> acao={acao}, confianca={confianca:.2f}, "
            f"justificativa={justificativa[:120]}"
        )

        return AnaliseResponse(
            acao=acao,
            confianca=confianca,
            justificativa=justificativa
        )

    except Exception as e:
        logging.exception("Erro ao analisar gráfico com a Groq")
        return AnaliseResponse(
            acao="NAO_OPERAR",
            confianca=0.0,
            justificativa="Erro ao analisar gráfico. Tente novamente mais tarde."
        )
