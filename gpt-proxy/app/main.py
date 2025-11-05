import os, logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# Logging claro a stdout (Cloud Run lo captura)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

# Prueba también con gpt-4.1-mini si tu key no tiene visión en 4o-mini
MODEL = os.getenv("MODEL", "gpt-4o-mini")
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="1.3")

class InferenceIn(BaseModel):
    text: str
    image_b64: str | None = None
    mime: str | None = None  # "image/jpeg" o "image/png"

class InferenceOut(BaseModel):
    model: str
    output: str
    branch: str | None = None  # opcional, para debug rápido

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
async def infer(payload: InferenceIn, request: Request):
    try:
        # modo debug simple: /infer?debug=1 añade "branch" en la respuesta
        debug = request.query_params.get("debug") == "1"

        messages_content = [{"type": "text", "text": payload.text}]
        branch = "text"

        if payload.image_b64:
            mime = payload.mime or "image/jpeg"
            data_url = f"data:{mime};base64,{payload.image_b64}"
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
            branch = "vision"
            log.info(f"[VISION] mime={mime} b64_len={len(payload.image_b64)}")

        else:
            log.info("[TEXT] sin imagen (no se recibió image_b64)")

        # Usamos SIEMPRE chat.completions (más robusto para visión)
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.2,
        )
        out = chat.choices[0].message.content

        if debug:
            return {"model": MODEL, "output": out, "branch": branch}
        return {"model": MODEL, "output": out}

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference error") from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
