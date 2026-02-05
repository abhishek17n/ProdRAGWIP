from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the BERT model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

class TextQuery(BaseModel):
    query: str

@app.post("/embed")
async def generate_embedding(request: TextQuery):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.query, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate embeddings from the model
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)