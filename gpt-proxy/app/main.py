# main.py
import os, base64, binascii
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
MAX_REQ_BYTES = 32 * 1024 * 1024  # ~32 MiB

client = OpenAI()
app = FastAPI(title="GPT Proxy", version="1.6")

class InferenceIn(BaseModel):
    text: str = Field(..., description="Prompt para el modelo")
    image_b64: Optional[str] = Field(None, description="Imagen en base64 (SIN prefijo data:)")
    mime: Optional[str] = Field(None, description="image/jpeg|image/png|image/webp")

class InferenceOut(BaseModel):
    model: str
    output: str
    image_b64: Optional[str] = None
    debug: Dict[str, Any]

@app.get("/")
def root():
    return {"ok": True, "service": "GPT Proxy", "version": "1.6"}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/echo")
async def echo(req: Request):
    headers = dict(req.headers)
    try:
        body = await req.json()
    except Exception:
        body = (await req.body()).decode("utf-8", errors="replace")
    return {"headers": headers, "body": body}

@app.post("/infer", response_model=InferenceOut, response_model_exclude_none=False)
def infer(payload: InferenceIn):
    try:
        has_img = bool(payload.image_b64 and payload.image_b64.strip())
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": payload.text}]

        debug = {
            "has_image_b64": has_img,
            "mime": payload.mime,
            "b64_prefix": (payload.image_b64[:20] if has_img else None),
            "approx_bytes": None,
            "decoded_len": None,
        }

        if has_img:
            approx_bytes = int(len(payload.image_b64) * 0.75)
            debug["approx_bytes"] = approx_bytes
            if approx_bytes > MAX_REQ_BYTES:
                raise HTTPException(status_code=413, detail="Imagen demasiado grande (~>32 MiB).")

            try:
                img_bytes = base64.b64decode(payload.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="image_b64 inválido (no es base64).")

            debug["decoded_len"] = len(img_bytes)

            # Detectar MIME si no viene
            if not payload.mime:
                import imghdr
                fmt = imghdr.what(None, h=img_bytes)
                payload.mime = f"image/{fmt}" if fmt else "application/octet-stream"

            # Data URL para Responses API
            data_url = f"data:{payload.mime};base64,{payload.image_b64}"
            content.append({"type": "input_image", "image_url": data_url})

        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )

        return {
            "model": MODEL,
            "output": resp.output_text,
            "image_b64": payload.image_b64 if has_img else None,  # clave siempre presente
            "debug": debug,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
