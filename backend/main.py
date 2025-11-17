# backend/main.py
import os
import base64
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# --- APP FASTAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLIENTE GROQ (usa variável de ambiente GROQ_API_KEY) ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.get("/")
async def root():
    return {"status": "online"}


@app.post("/api/analisar")
async def analisar_imagem(image: UploadFile = File(...)):
    try:
        # lê bytes da imagem enviada
        img_bytes = await image.read()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        # chamada à Groq (modelo de visão)
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
                                # usamos base64 no formato que a Groq aceita
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_completion_tokens=256,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
        )

        # em JSON mode, o content já vem como string JSON
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
        # log pro Render
        print("Erro ao chamar Groq:", e)
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.5,
            "justificativa": "Erro ao analisar gráfico. Tente novamente mais tarde.",
        }
