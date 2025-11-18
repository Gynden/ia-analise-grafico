import os
import base64
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.get("/")
async def root():
    return {"status": "online"}


# Aceita TODOS os métodos em /api/analisar e /api/analisar/
ALL_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]

@app.api_route("/api/analisar", methods=ALL_METHODS)
@app.api_route("/api/analisar/", methods=ALL_METHODS)
async def analisar_imagem(image: UploadFile | None = File(None)):
    """
    - Se vier imagem (POST com FormData), chama a Groq e retorna COMPRAR/VENDER/NAO_OPERAR.
    - Se NÃO vier imagem (GET, HEAD, OPTIONS, etc.), devolve um JSON padrão.
    Assim, nunca mais teremos 405 nesse caminho.
    """

    # Se não veio arquivo (GET, HEAD, OPTIONS...), responde algo padrão
    if image is None:
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": "Nenhuma imagem recebida. Envie um print do gráfico.",
        }

    try:
        img_bytes = await image.read()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Você é uma IA de análise técnica de gráficos. "
                                "Recebe um print de candles (timeframe 5 minutos) "
                                "e deve responder APENAS um JSON com:\n"
                                "{\n"
                                '  \"acao\": \"COMPRAR\" | \"VENDER\" | \"NAO_OPERAR\",\n'
                                '  \"confianca\": 0.0 a 1.0,\n'
                                '  \"justificativa\": \"texto curto em português\"\n'
                                "}\n\n"
                                "Prefira NAO_OPERAR quando o cenário estiver confuso, "
                                "lateral ou muito arriscado."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analise esse gráfico e me diga a melhor decisão agora.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_completion_tokens=256,
        )

        text = completion.choices[0].message.content
        data = json.loads(text)

        acao = data.get("acao", "NAO_OPERAR")
        confianca = float(data.get("confianca", 0.5))
        justificativa = data.get(
            "justificativa",
            "Mercado indefinido. IA recomenda não operar.",
        )

        return {
            "acao": acao,
            "confianca": confianca,
            "justificativa": justificativa,
        }

    except Exception as e:
        print("Erro Groq:", e)
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": "Erro ao analisar gráfico. Tente novamente mais tarde.",
        }
