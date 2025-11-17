import os
import base64
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# cliente OpenAI usando variável de ambiente
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
async def root():
    return {"status": "online"}

@app.post("/api/analisar")
async def analisar(image: UploadFile = File(...)):
    # lê bytes da imagem enviada pelo front
    img_bytes = await image.read()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    # instruções pra IA (em pt-BR, focando em COMPRAR / VENDER / NÃO OPERAR)
    prompt = """
Você é um analista técnico de gráficos de trading.

O usuário tirou um print de um gráfico de candles (timeframe 5 minutos).
Seu trabalho é dizer se, nesse exato momento, a melhor decisão é:

- "COMPRAR" (se o contexto favorece alta / continuidade de alta)
- "VENDER" (se o contexto favorece baixa / continuidade de baixa)
- "NAO_OPERAR" (se o cenário estiver confuso, lateral, sem sinal claro ou muito arriscado)

Considere:
- Tendência (alta, baixa ou lateral)
- Regiões de suporte e resistência
- Força dos candles recentes (corpos grandes/pequenos, pavios)
- Possíveis rompimentos ou rejeições importantes

Responda SEMPRE em JSON válido, exatamente nesse formato:

{
  "acao": "COMPRAR" | "VENDER" | "NAO_OPERAR",
  "confianca": 0.0 a 1.0,
  "justificativa": "texto curto em português explicando o motivo"
}

Regras importantes:
- Se o gráfico estiver lateral ou muito indeciso, escolha "NAO_OPERAR" com confiança <= 0.6.
- Evite ser agressivo demais. Prefira "NAO_OPERAR" quando não houver confluência clara.
"""

    try:
        # chamada à IA com imagem + prompt
        response = client.responses.create(
            model="gpt-4.1-mini",  # pode trocar por gpt-4.1 se quiser algo mais forte
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_output_tokens=200,
        )

        # SDK novo tem atalho pra pegar o texto inteiro
        raw = response.output_text
        data = json.loads(raw)

        # segurança: garante campos mínimos
        acao = data.get("acao", "NAO_OPERAR")
        confianca = float(data.get("confianca", 0.0))
        justificativa = data.get(
            "justificativa",
            "Não foi possível analisar o gráfico com clareza."
        )

        return {
            "acao": acao,
            "confianca": confianca,
            "justificativa": justificativa,
        }

    except Exception as e:
        # log simples no servidor
        print("Erro na IA:", e)
        # resposta fallback pro front não quebrar
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": "Erro ao analisar gráfico. Tente novamente mais tarde.",
        }
