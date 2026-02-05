from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the Hugging Face transformer model for embeddings
embedder = pipeline("feature-extraction", model="sentence-transformers/all-mpnet-base-v2")

class TextQuery(BaseModel):
    query: str

@app.post("/embed")
async def generate_embedding(request: TextQuery):
    try:
        # Generate the embedding from the query text
        embedding = embedder(request.query)[0][0]
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)