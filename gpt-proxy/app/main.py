import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# Lee API key desde el entorno (Cloud Run la inyecta desde Secret Manager)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4o-mini")  # modelo por defecto
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="1.1")

class InferenceIn(BaseModel):
    text: str

class InferenceOut(BaseModel):
    model: str
    output: str
    bonus: str  # ← nuevo campo para "banana"

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        # Llamada al modelo de OpenAI
        resp = client.responses.create(
            model=MODEL,
            input=payload.text
        )
        out = resp.output_text

        # Devuelve dos valores: el output y la palabra "banana"
        return {"model": MODEL, "output": out, "bonus": "banana"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
