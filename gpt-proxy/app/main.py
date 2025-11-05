import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# Lee API key desde env (Cloud Run la inyecta desde Secret Manager)
# NO la pongas en el código ni en el repo
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4o-mini")  # puedes fijar "gpt-3.5-turbo" si quieres
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="1.0")

class InferenceIn(BaseModel):
    text: str

class InferenceOut(BaseModel):
    model: str
    output: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        resp = client.responses.create(
            model=MODEL,
            input=payload.text
        )
        out = resp.output_text
        return {"model": MODEL, "output": out}
    except Exception as e:
        # No exponemos detalles internos
        raise HTTPException(status_code=500, detail="Inference error") from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
