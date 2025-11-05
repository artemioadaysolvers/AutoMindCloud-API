import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import logging

# Configurar logs
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Validar clave API
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")  # ✅ el mismo que te funciona
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="2.1")

class InferenceIn(BaseModel):
    text: str
    image_b64: str | None = None
    mime: str | None = None  # ejemplo: image/jpeg

class InferenceOut(BaseModel):
    model: str
    output: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        # Si hay imagen, usar exactamente el formato que probaste en Colab
        if payload.image_b64:
            mime = payload.mime or "image/jpeg"
            data_url = f"data:{mime};base64,{payload.image_b64}"
            log.info(f"[VISION] usando modelo {MODEL}, mime={mime}, len={len(payload.image_b64)}")

            resp = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": payload.text},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
            )

        else:
            log.info(f"[TEXT] usando modelo {MODEL}")
            resp = client.responses.create(
                model=MODEL,
                input=payload.text
            )

        out = resp.output_text
        return {"model": MODEL, "output": out}

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference error") from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
