from fastapi import FastAPI
from pydantic import BaseModel
from agents.graph import run_agent

app = FastAPI(title="Personal AI Assistant")

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat/")
async def chat(req: ChatRequest):
    """Endpoint to chat with the AI assistant."""
    response = run_agent(req.prompt)
    return {"response": response}
