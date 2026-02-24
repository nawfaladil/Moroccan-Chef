from fastapi import FastAPI
from src.rag.orchestrate import Orchestrator

app = FastAPI()

responds = Orchestrator()

@app.get("/health")
async def health():
    return {"message": "heartbeat"}

@app.get("/generate/{query_text}")
async def generate(query_text: str):
    return responds.respond(query_text)
