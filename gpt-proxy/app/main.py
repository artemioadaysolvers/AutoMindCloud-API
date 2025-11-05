import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4o-mini")  # modelo con visión
client = OpenAI()  # usa OPENAI_API_KEY del entorno

app = FastAPI(title="GPT Proxy", version="1.0")

class InferenceIn(BaseModel):
    text: str
    image_b64: str | None = None  # <-- NUEVO

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
            # Asumimos JPEG por defecto; cambia a image/png si corresponde
            data_url = f"data:image/jpeg;base64,{payload.image_b64}"
            resp = client.responses.create(
                model=MODEL,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": payload.text},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
            )
        else:
            resp = client.responses.create(model=MODEL, input=payload.text)

        return {"model": MODEL, "output": resp.output_text}
    except Exception:
        raise HTTPException(status_code=500, detail="Inference error")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
