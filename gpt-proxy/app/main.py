import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import base64
from io import BytesIO
from PIL import Image

# Lee API key desde env (Cloud Run la inyecta desde Secret Manager)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4-vision-preview")  # Usar modelo con visión
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="1.0")

class InferenceIn(BaseModel):
    text: str
    image_b64: str  # Nuevo campo para la imagen en base64
    mime: str = "image/png"  # MIME type por defecto

class InferenceOut(BaseModel):
    model: str
    output: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

def process_image_base64(image_b64: str, mime: str):
    """Convierte base64 a datos que OpenAI puede procesar"""
    try:
        # Decodificar base64
        image_data = base64.b64decode(image_b64)
        
        # Para modelos de visión, podemos enviar la imagen como base64 directamente
        # o procesarla con PIL si necesitamos validación
        image = Image.open(BytesIO(image_data))
        
        # Validar que es una imagen válida
        image.verify()
        
        # Regresar al formato original para enviar a OpenAI
        return image_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        # Preparar el mensaje con la imagen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": payload.text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{payload.mime};base64,{payload.image_b64}"
                        }
                    }
                ]
            }
        ]
        
        # Llamar a OpenAI con el modelo de visión
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1000
        )
        
        out = response.choices[0].message.content
        return {"model": MODEL, "output": out}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference error") from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
