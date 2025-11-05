# server.py
import os, base64, binascii
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import uvicorn

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
MAX_REQ_BYTES = 32 * 1024 * 1024  # límite práctico Cloud Run

client = OpenAI()
app = FastAPI(title="GPT Proxy", version="1.2")

class InferenceIn(BaseModel):
    text: str = Field(..., description="Prompt para el modelo")
    image_b64: Optional[str] = Field(
        None, description="Imagen en base64 (sin prefijo data:)"
    )
    mime: Optional[str] = Field("image/jpeg", description="image/jpeg|image/png|image/webp")

class InferenceOut(BaseModel):
    model: str
    output: str
    debug: dict

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        has_img = bool(payload.image_b64 and payload.image_b64.strip())
        content = [{"type": "input_text", "text": payload.text}]
        debug = {
            "has_image_b64": has_img,
            "mime": payload.mime,
            "b64_prefix": (payload.image_b64[:16] if has_img else None),
            "approx_bytes": None,
            "decoded_len": None,
        }

        if has_img:
            # Estimar tamaño (base64 ~ 4/3 binario)
            approx_bytes = int(len(payload.image_b64) * 0.75)
            debug["approx_bytes"] = approx_bytes
            if approx_bytes > MAX_REQ_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Imagen demasiado grande para esta vía (~>32MiB). Usa GCS + URL."
                )

            # Decodificar para validar que realmente llegó imagen
            try:
                img_bytes = base64.b64decode(payload.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="image_b64 inválido (no es base64).")

            debug["decoded_len"] = len(img_bytes)

            if payload.mime not in {"image/jpeg", "image/png", "image/webp"}:
                payload.mime = "image/jpeg"

            data_url = f"data:{payload.mime};base64,{payload.image_b64}"

            # Añadimos SIEMPRE el bloque de imagen si llegó base64
            content.append({"type": "input_image", "image_url": data_url})

        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )
        out = resp.output_text
        return {"model": MODEL, "output": out, "debug": debug}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Inference error")
        
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
