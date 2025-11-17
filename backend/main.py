from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online"}

@app.post("/api/analisar")
async def analisar(image: UploadFile = File(...)):
    conteudo = await image.read()

    # TODO: aqui depois você conecta na IA de visão de verdade.
    # Por enquanto, vamos devolver algo fixo só para testar o app.

    resposta = {
        "acao": "VENDER",        # COMPRAR, VENDER ou NAO_OPERAR
        "confianca": 0.70,       # 0 a 1
        "justificativa": (
            "Exemplo demo: tendência de baixa, rejeição em resistência e "
            "candles fortes de venda. IA sugere venda."
        )
    }
    return resposta
