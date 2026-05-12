from fastapi import FastAPI, HTTPException
from app.confluence import ConfluenceClient
import uvicorn

app = FastAPI(title="Pulsar Backend - Requirement Intelligence")
conf_client = ConfluenceClient()

@app.get("/")
def read_root():
    return {"status": "Pulsar Backend is running"}

@app.post("/analyze/{page_id}")
async def analyze_requirements(page_id: str):
    try:
        # 1. Fetch & Chunk (из твоего confluence.py)
        chunks = conf_client.fetch_page_content(page_id)
        
        # 2. Здесь мы подготовим данные для твоего Internal AI
        # Нам нужно передать список топиков, чтобы AI нашел связи
        topics = list(set([c['topic'] for c in chunks]))
        
        return {
            "page_id": page_id,
            "detected_topics": topics,
            "chunks_count": len(chunks),
            "message": "Data ready for Internal AI dependency analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)