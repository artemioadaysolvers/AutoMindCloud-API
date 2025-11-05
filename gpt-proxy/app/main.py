import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import logging

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4o-mini")  # modelo con visión
client = OpenAI()  # usa OPENAI_API_KEY del entorno

app = FastAPI(title="GPT Proxy", version="1.0")
log = logging.getLogger("uvicorn.error")

class InferenceIn(BaseModel):
    text: str
    image_b64: str | None = None  # imagen opcional en base64 (sin data URL)
    mime: str | None = None       # opcional: "image/jpeg" o "image/png"

class InferenceOut(BaseModel):
    model: str
    output: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        if payload.image_b64:
            # Construimos data URL con el MIME correcto (default: JPEG)
            mime = payload.mime or "image/jpeg"
            data_url = f"data:{mime};base64,{payload.image_b64}"
            log.info("Rama VISIÓN: enviando imagen + texto a chat.completions")

            # ✅ Camino robusto para visión: Chat Completions con image_url
            chat = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": payload.text},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},  # <- clave: objeto con {"url": ...}
                            },
                        ],
                    }
                ],
                temperature=0.2,
            )
            out = chat.choices[0].message.content
            return {"model": MODEL, "output": out}

        # Rama solo texto
        log.info("Rama TEXTO: sin imagen, usando responses")
        resp = client.responses.create(model=MODEL, input=payload.text)
        out = resp.output_text
        return {"model": MODEL, "output": out}

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference error") from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
