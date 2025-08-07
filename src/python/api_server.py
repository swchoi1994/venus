#!/usr/bin/env python3
"""FastAPI OpenAI-compatible API server for Venus Inference Engine"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, AsyncGenerator
import uvicorn
import ctypes
import platform
import json
import uuid
import time
import asyncio
from pathlib import Path

# Load the C library
def load_venus_library():
    system = platform.system()
    if system == "Darwin":
        lib_path = "./libvenus.dylib"
    elif system == "Windows":
        lib_path = "./venus.dll"
    else:
        lib_path = "./libvenus.so"
    
    try:
        return ctypes.CDLL(lib_path)
    except OSError:
        # Try without prefix
        try:
            return ctypes.CDLL(lib_path.replace("./", ""))
        except OSError:
            raise RuntimeError(f"Failed to load Venus library from {lib_path}")

# Initialize library
lib = load_venus_library()

# Define C structures
class GenerationConfig(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("max_tokens", ctypes.c_int),
        ("seed", ctypes.c_int),
        ("repetition_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
    ]

# Define C function signatures
lib.create_engine.argtypes = [ctypes.c_char_p]
lib.create_engine.restype = ctypes.c_void_p

lib.free_engine.argtypes = [ctypes.c_void_p]
lib.free_engine.restype = None

lib.generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(GenerationConfig)]
lib.generate.restype = ctypes.c_char_p

lib.create_tokenizer.argtypes = [ctypes.c_char_p]
lib.create_tokenizer.restype = ctypes.c_void_p

lib.free_tokenizer.argtypes = [ctypes.c_void_p]
lib.free_tokenizer.restype = None

lib.tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.tokenize.restype = ctypes.POINTER(ctypes.c_int)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

# Venus Engine wrapper
class VenusEngine:
    def __init__(self, model_path: str):
        self.engine = lib.create_engine(model_path.encode('utf-8'))
        if not self.engine:
            raise RuntimeError(f"Failed to load model from {model_path}")
        self.tokenizer = lib.create_tokenizer(model_path.encode('utf-8'))
        self.model_name = Path(model_path).stem
    
    def __del__(self):
        if hasattr(self, 'engine') and self.engine:
            lib.free_engine(self.engine)
        if hasattr(self, 'tokenizer') and self.tokenizer:
            lib.free_tokenizer(self.tokenizer)
    
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        result = lib.generate(self.engine, prompt.encode('utf-8'), ctypes.byref(config))
        if result:
            text = result.decode('utf-8')
            # Free the C string
            ctypes.c_void_p.from_address(ctypes.addressof(ctypes.c_char_p(result)))
            return text
        return ""
    
    def count_tokens(self, text: str) -> int:
        n_tokens = ctypes.c_int()
        tokens_ptr = lib.tokenize(self.tokenizer, text.encode('utf-8'), ctypes.byref(n_tokens))
        if tokens_ptr:
            # Free the tokens array
            ctypes.c_void_p.from_address(ctypes.addressof(tokens_ptr))
        return n_tokens.value

# Engine manager
class EngineManager:
    def __init__(self):
        self.engines = {}
    
    def load_model(self, model_name: str, model_path: str):
        try:
            self.engines[model_name] = VenusEngine(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def get_engine(self, model_name: str) -> Optional[VenusEngine]:
        return self.engines.get(model_name)
    
    def list_models(self) -> List[str]:
        return list(self.engines.keys())

# Initialize FastAPI app
app = FastAPI(title="Venus Inference Engine API")
engine_manager = EngineManager()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    return {"message": "Venus Inference Engine API Server"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/v1/models")
async def list_models():
    models = engine_manager.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "venus"
            }
            for model in models
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Get engine
    engine = engine_manager.get_engine(request.model)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    # Format messages into prompt
    prompt = format_chat_prompt(request.messages)
    
    # Create generation config
    config = GenerationConfig(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        seed=-1,
        repetition_penalty=1.1,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )
    
    if request.stream:
        return StreamingResponse(
            generate_stream(engine, prompt, config, request.model),
            media_type="text/event-stream"
        )
    else:
        # Generate response
        response_text = engine.generate(prompt, config)
        
        # Count tokens
        prompt_tokens = engine.count_tokens(prompt)
        completion_tokens = engine.count_tokens(response_text)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )

@app.post("/v1/completions")
async def completions(request: dict):
    # TODO: Implement completions endpoint
    raise HTTPException(status_code=501, detail="Completions endpoint not yet implemented")

# Helper functions
def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into a single prompt string"""
    return "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

async def generate_stream(engine: VenusEngine, prompt: str, config: GenerationConfig, model: str) -> AsyncGenerator[str, None]:
    """Generate streaming response"""
    # Generate the full response first (TODO: implement actual streaming)
    response_text = engine.generate(prompt, config)
    
    # Send initial chunk
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    
    # Send content in chunks
    chunk_size = 10
    for i in range(0, len(response_text), chunk_size):
        chunk_text = response_text[i:i+chunk_size]
        content_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"
        await asyncio.sleep(0.02)  # Small delay for streaming effect
    
    # Send final chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# Load models on startup
@app.on_event("startup")
async def startup_event():
    # TODO: Scan models directory and load models
    # For now, try to load a demo model if it exists
    demo_model_path = "./models/demo_model.bin"
    if Path(demo_model_path).exists():
        try:
            engine_manager.load_model("demo-model", demo_model_path)
            print(f"Loaded model: demo-model")
        except Exception as e:
            print(f"Failed to load demo model: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Venus Inference Engine API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-dir", default="./models", help="Directory containing models")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)