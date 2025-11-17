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

@app.post("/api/analisar")
async def analisar_imagem(image: UploadFile = File(...)):
    img_bytes = await image.read()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    try:
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
                                "Responda APENAS JSON:\n"
                                "{"
                                '"acao": "COMPRAR" | "VENDER" | "NAO_OPERAR",'
                                '"confianca": 0.0 a 1.0,'
                                '"justificativa": "texto curto em português"'
                                "}"
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analise esse gráfico de candles."},
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

        return {
            "acao": data.get("acao", "NAO_OPERAR"),
            "confianca": float(data.get("confianca", 0.5)),
            "justificativa": data.get(
                "justificativa",
                "Mercado indefinido. IA recomenda não operar.",
            ),
        }

    except Exception as e:
        print("Erro Groq:", e)
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.5,
            "justificativa": "Erro ao analisar gráfico. Tente novamente mais tarde.",
        }
