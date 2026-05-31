from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from .inference import get_predictions

app = FastAPI(title="Tamil Next Word Prediction API")

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    current_sentence: str
    language: str = "ta"
    partial_word: str = ""  # The word currently being typed (after last space)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Tamil Prediction API is running"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    sentence = request.current_sentence.strip()
    language = request.language.lower()
    partial_word = request.partial_word.strip()
    
    # Input validation
    if not sentence:
        raise HTTPException(status_code=400, detail="Input sentence cannot be empty")
    
    if len(sentence) > 100:
        raise HTTPException(status_code=400, detail="Input sentence exceeds maximum length of 100 characters")
    
    if language not in ['tamil', 'en', 'ta']:
        raise HTTPException(status_code=400, detail="Language must be 'tamil', 'ta', or 'en'")
    
    # Normalize language code
    language = 'ta' if language in ['tamil', 'ta'] else 'en'
    
    print(f"Received prediction request for: {sentence} (Language: {language}, Partial: '{partial_word}')")
    
    try:
        # Run inference
        results = get_predictions(sentence, language, partial_word)
        
        # Calculate total latency
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        # Add latency to response (could clarify per model if needed, but overall is fine for now)
        # Note: get_predictions runs sequentially, so we might want to split timing if precise per-model latency is needed.
        # For now, let's just add the global API latency.
        
        return {
            "status": "success",
            "latency_ms": latency_ms,
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
