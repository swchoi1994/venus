"""FastAPI server with OpenAI-compatible API"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import asyncio
from venus.core import LLM, SamplingParams

app = FastAPI(title="Venus Inference Server", version="0.1.0")

# Global LLM instance
llm: Optional[LLM] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert messages to prompt
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )
    
    # Generate response
    response = llm.chat(messages_dict, sampling_params)
    
    # Return OpenAI-compatible response
    return {
        "id": "chatcmpl-venus",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": llm.model_name if llm else "none",
            "object": "model",
            "created": 1234567890,
            "owned_by": "venus"
        }]
    }

def run(model: str, host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Run the API server"""
    global llm
    
    print(f"🚀 Starting Venus server with {model}")
    llm = LLM(model=model, **kwargs)
    
    uvicorn.run(app, host=host, port=port)

def main():
    """CLI entry point"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    run(args.model, args.host, args.port)

if __name__ == "__main__":
    main()