import os
os.environ["USE_TF"] = "0"  # avoid loading TensorFlow/Keras


from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ✅ Load academic embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed_text(req: EmbedRequest):
    vectors = model.encode(req.texts, normalize_embeddings=True).tolist()
    return {"embeddings": vectors}
