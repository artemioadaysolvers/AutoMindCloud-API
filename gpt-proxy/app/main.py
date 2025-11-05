# server.py
import os, base64, binascii
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import uvicorn

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada.")
MODEL = os.getenv("MODEL", "gpt-4.1-mini")  # modelo con visión
MAX_REQ_BYTES = 32 * 1024 * 1024  # Límite práctico HTTP/1 en Cloud Run (~32MiB)

client = OpenAI()
app = FastAPI(title="GPT Proxy", version="1.1")

# --- Schemas ---
class InferenceIn(BaseModel):
    text: str = Field(..., description="Prompt para el modelo")
    image_b64: Optional[str] = Field(None, description="Imagen en base64 (sin prefijo data:)")
    mime: Optional[str] = Field("image/jpeg", description="Mime type, ej: image/jpeg, image/png")

class InferenceOut(BaseModel):
    model: str
    output: str

# --- Health ---
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

# --- Infer ---
@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        # Validaciones mínimas
        if payload.image_b64 is None:
            # Solo texto -> también permitido
            content = [{"type": "input_text", "text": payload.text}]
        else:
            # Estimar tamaño para cortar solicitudes demasiado grandes
            # Aproximación: base64 ~ 4/3 del binario ⇒ bin_size ≈ len(b64) * 3/4
            approx_bytes = int(len(payload.image_b64) * 0.75)
            if approx_bytes > MAX_REQ_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Imagen demasiado grande para esta vía (≈>32MiB)."
                           " Sube la imagen a Cloud Storage y envía la URL."
                )

            # Decodificar base64 de forma segura
            try:
                img_bytes = base64.b64decode(payload.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="image_b64 no es base64 válido.")

            if payload.mime not in {"image/jpeg", "image/png", "image/webp"}:
                # no es obligatorio, pero ayuda a los modelos
                payload.mime = "image/jpeg"

            # Construir data URL para la Responses API
            data_url = f"data:{payload.mime};base64,{payload.image_b64}"

            content = [
                {"type": "input_text", "text": payload.text},
                {"type": "input_image", "image_url": data_url},
            ]

        # Llamar a la Responses API (texto/imágenes)
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )
        out = resp.output_text
        return {"model": MODEL, "output": out}

    except HTTPException:
        raise
    except Exception as e:
        # Log interno opcional: print(e)
        raise HTTPException(status_code=500, detail="Inference error") from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
