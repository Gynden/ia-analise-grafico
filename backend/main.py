# backend/main.py
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

@app.post("/")
async def analisar_imagem(image: UploadFile = File(...)):
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
                                "Você é uma IA de análise técnica. "
                                "Recebe um print de gráfico de candles "
                                "e deve responder APENAS um JSON com "
                                "os campos: acao (COMPRAR, VENDER ou NAO_OPERAR), "
                                "confianca (0 a 1) e justificativa (frase curta em português)."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analise esse gráfico e me diga a melhor decisão.",
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
            max_completion_tokens=256,
        )

        msg = completion.choices[0].message
        text = getattr(msg, "content", msg)

        try:
            data = json.loads(text)
        except Exception:
            # fallback se a IA não respeitar o formato
            data = {
                "acao": "NAO_OPERAR",
                "confianca": 0.5,
                "justificativa": "Erro ao interpretar resposta da IA.",
            }

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
        print("Erro ao chamar Groq:", e)
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.5,
            "justificativa": "Erro ao analisar gráfico. Tente novamente mais tarde.",
        }
