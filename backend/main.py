import os
import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

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

ALL_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]

# Arquivo simples para logar os sinais (um JSON por linha)
LOG_PATH = Path("sinais_log.jsonl")


def log_signal(entry: dict) -> None:
    """Registra cada sinal em um arquivo JSONL (um JSON por linha)."""
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Só loga no console, não quebra a API
        print("Erro ao salvar log de sinal:", e)


async def _processar_imagem(image: UploadFile | None):
    """
    Lógica central de análise:
    - Se não vier imagem, devolve uma resposta padrão.
    - Se vier, chama a Groq, interpreta o JSON e devolve acao/confianca/justificativa.
    - Aplica regra conservadora em cima da confiança.
    - Registra o sinal em arquivo de log.
    """
    agora = datetime.now(timezone.utc).isoformat()
    sinal_id = str(uuid.uuid4())

    if image is None:
        entrada = {
            "id": sinal_id,
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": "Nenhuma imagem recebida. Envie um print do gráfico.",
            "timeframe": "M1",
            "created_at": agora,
            "erro": "sem_imagem",
        }
        log_signal(entrada)
        return entrada

    try:
        img_bytes = await image.read()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        # PROMPT AJUSTADO: trader conservador em M1 (scalp)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            temperature=0.15,
            max_completion_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Você é uma IA de análise técnica focada em SCALP no timeframe M1 (1 minuto). "
                                "Você recebe apenas um print de um gráfico de candles em M1 e deve ajudar o trader humano "
                                "a decidir se vale a pena entrar NA PRÓXIMA VELA.\n\n"
                                "REGRAS IMPORTANTES (PERFIL SUPER CONSERVADOR):\n"
                                "- Seu objetivo é PROTEGER o capital. Não force sinais.\n"
                                "- SE TIVER QUALQUER DÚVIDA, escolha sempre 'NAO_OPERAR'.\n"
                                "- Só sugira 'COMPRAR' quando houver:\n"
                                "  * Tendência de ALTA bem clara (topos e fundos ascendentes) E\n"
                                "  * Preço fazendo PULLBACK em região de SUPORTE ou média relevante, "
                                "    com candles mostrando rejeição da queda (pavio para baixo, corpo respeitando a zona).\n"
                                "- Só sugira 'VENDER' quando houver:\n"
                                "  * Tendência de BAIXA bem clara (topos e fundos descendentes) E\n"
                                "  * Preço fazendo PULLBACK em região de RESISTÊNCIA ou média relevante, "
                                "    com candles mostrando rejeição da alta (pavio para cima, corpo respeitando a zona).\n"
                                "- Se o gráfico estiver lateral, sem direção clara, com muita sombra, muitos pavios, "
                                "ou muito ruído -> 'NAO_OPERAR'.\n"
                                "- Se tiver acabado de aparecer um candle MUITO grande (explosivo), evite operar "
                                "na sequência imediata -> 'NAO_OPERAR'.\n"
                                "- Se o preço estiver “no meio do caminho”, longe de suporte/resistência óbvios, "
                                "sem contexto claro -> 'NAO_OPERAR'.\n\n"
                                "TIMEFRAME:\n"
                                "- Considere que o gráfico é M1 (1 minuto) e a decisão é apenas para a PRÓXIMA VELA.\n\n"
                                "FORMATO DA RESPOSTA (OBRIGATÓRIO):\n"
                                "{\n"
                                '  \"acao\": \"COMPRAR\" | \"VENDER\" | \"NAO_OPERAR\",\n'
                                '  \"confianca\": número entre 0.0 e 1.0 (onde 1.0 é certeza absoluta),\n'
                                '  \"justificativa\": frase curta em português explicando a leitura do gráfico\n'
                                "}\n\n"
                                "Nunca invente valores de preço, horário ou indicador específico; descreva apenas "
                                "o que você enxerga visualmente no gráfico. Se não for possível ter um cenário claro, "
                                "devolva 'NAO_OPERAR' com confiança baixa (por exemplo 0.3–0.5)."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analise esse gráfico de candles em M1 e diga qual é a melhor decisão agora "
                                "para a PRÓXIMA VELA: COMPRAR, VENDER ou NAO_OPERAR, seguindo as regras conservadoras."
                            ),
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
        )

        text = completion.choices[0].message.content
        data = json.loads(text)

        acao = data.get("acao", "NAO_OPERAR")
        confianca = float(data.get("confianca", 0.5))
        justificativa = data.get(
            "justificativa",
            "Mercado indefinido. IA recomenda não operar.",
        )

        # CAMADA EXTRA DE SEGURANÇA:
        # se confiança for baixa, força NAO_OPERAR
        LIMIAR_CONFIANCA = 0.70  # M1 é mais sujo, então vamos ser mais rígidos

        acao_original = acao
        if confianca < LIMIAR_CONFIANCA:
            acao = "NAO_OPERAR"

        resposta = {
            "id": sinal_id,
            "acao": acao,
            "confianca": confianca,
            "justificativa": justificativa,
            "timeframe": "M1",
            "created_at": agora,
            "acao_original_modelo": acao_original,
            "limiar_confianca": LIMIAR_CONFIANCA,
        }

        log_signal(resposta)
        return resposta

    except Exception as e:
        print("Erro Groq:", e)
        erro_resp = {
            "id": sinal_id,
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": "Erro ao analisar gráfico. Tente novamente mais tarde.",
            "timeframe": "M1",
            "created_at": agora,
            "erro": str(e),
        }
        log_signal(erro_resp)
        return erro_resp


# Raiz (/) usa a MESMA lógica da IA – qualquer método
@app.api_route("/", methods=ALL_METHODS)
async def analisar_root(image: UploadFile | None = File(None)):
    return await _processar_imagem(image)


# /api/analisar e /api/analisar/ também usam a mesma lógica
@app.api_route("/api/analisar", methods=ALL_METHODS)
@app.api_route("/api/analisar/", methods=ALL_METHODS)
async def analisar_imagem(image: UploadFile | None = File(None)):
    return await _processar_imagem(image)
