#!/usr/bin/env python3
"""FastAPI OpenAI-compatible API server for Venus Inference Engine"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import uvicorn
import ctypes
import platform
import json
import uuid
import time
import asyncio
import os
from pathlib import Path
import base64
import io
import hashlib

try:
    import torch  # type: ignore
    from transformers import AutoModelForVision2Seq, AutoProcessor, TextIteratorStreamer  # type: ignore
    from PIL import Image  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRANSFORMERS_AVAILABLE = False

MODEL_DIR = Path(os.environ.get("VENUS_MODEL_DIR", "./models")).resolve()
manifest_models: Dict[str, Dict[str, Any]] = {}
default_model_name: Optional[str] = None
MAX_IMAGE_SIDE = int(os.environ.get("VLM_MAX_IMAGE_SIDE", "640"))

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
class Attachment(BaseModel):
    kind: str
    data: str
    mime_type: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    attachments: Optional[List[Attachment]] = None

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
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.engine = lib.create_engine(model_path.encode('utf-8'))
        if not self.engine:
            raise RuntimeError(f"Failed to load model from {model_path}")
        self.tokenizer = lib.create_tokenizer(self.tokenizer_path.encode('utf-8'))
        if not self.tokenizer:
            lib.free_engine(self.engine)
            raise RuntimeError(f"Failed to load tokenizer from {self.tokenizer_path}")
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
        self.vlm_engines = {}
    
    def load_model(self, model_name: str, model_path: str, tokenizer_path: Optional[str] = None):
        try:
            self.engines[model_name] = VenusEngine(model_path, tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def get_engine(self, model_name: str) -> Optional[VenusEngine]:
        return self.engines.get(model_name)
    
    def list_models(self) -> List[str]:
        return list(self.engines.keys()) + list(self.vlm_engines.keys())

    # HF VLM engines
    def load_vlm_model(self, model_name: str, model_dir: str):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/Pillow not available to load VLM model")
        self.vlm_engines[model_name] = HFVLMEngine(model_dir)

    def get_vlm_engine(self, model_name: str):
        return self.vlm_engines.get(model_name)


def resolve_artifact_path(base_dir: Path, artifact_path: Optional[str]) -> Optional[Path]:
    if not artifact_path:
        return None
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def load_models_from_manifest(model_dir: Path) -> bool:
    global manifest_models, default_model_name

    manifest_path = model_dir / "deployment.json"
    if not manifest_path.exists():
        print(f"[startup] deployment manifest not found at {manifest_path}")
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print(f"[startup] failed to parse deployment manifest {manifest_path}: {exc}")
        return False

    default_model_name = manifest.get("default_model")
    models_cfg = manifest.get("models", {})

    loaded_models: Dict[str, Dict[str, Any]] = {}
    loaded_any = False

    for model_name, cfg in models_cfg.items():
        model_kind = cfg.get("model_kind") or "llm"
        if model_kind == "vlm":
            # Prefer explicit hf_model_dir
            hf_dir = cfg.get("hf_model_dir")
            hf_path = resolve_artifact_path(model_dir, hf_dir) if hf_dir else None
            # Fallback: try a sibling directory with config.json
            if not hf_path:
                candidate = model_dir / f"{model_name}-fp16"
                if candidate.exists() and (candidate / "config.json").exists():
                    hf_path = candidate
                else:
                    # Try model_name directory
                    direct = model_dir / model_name
                    if direct.exists() and (direct / "config.json").exists():
                        hf_path = direct
            if not hf_path or not hf_path.exists():
                print(f"[startup] skipping {model_name}: hf_model_dir not found (set 'hf_model_dir' in deployment.json)")
                continue
            try:
                engine_manager.load_vlm_model(model_name, str(hf_path))
                loaded_models[model_name] = dict(cfg)
                loaded_any = True
                print(f"[startup] loaded VLM model {model_name} from {hf_path}")
            except Exception as exc:
                print(f"[startup] failed to load VLM model {model_name}: {exc}")
            continue

        # Default LLM via Venus C engine
        model_path = resolve_artifact_path(model_dir, cfg.get("model_path"))
        tokenizer_path = resolve_artifact_path(model_dir, cfg.get("tokenizer_path"))

        if not model_path or not model_path.exists():
            print(f"[startup] skipping {model_name}: model_path {model_path} does not exist")
            continue

        if tokenizer_path and not tokenizer_path.exists():
            print(f"[startup] tokenizer_path {tokenizer_path} for {model_name} not found; using model_path instead")
            tokenizer_path = None

        try:
            engine_manager.load_model(
                model_name,
                str(model_path),
                str(tokenizer_path) if tokenizer_path else None,
            )
            loaded_models[model_name] = dict(cfg)
            loaded_any = True
            print(f"[startup] loaded model {model_name} from {model_path}")
        except Exception as exc:
            print(f"[startup] failed to load model {model_name}: {exc}")

    manifest_models = loaded_models

    if not loaded_any:
        print(f"[startup] no models were loaded from {manifest_path}")

    return loaded_any

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
    data = []
    for model in models:
        manifest_entry = manifest_models.get(model, {})
        model_info = {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "venus",
        }
        if manifest_entry.get("metadata") is not None:
            model_info["metadata"] = manifest_entry["metadata"]
        if manifest_entry.get("model_kind") is not None:
            model_info["model_kind"] = manifest_entry["model_kind"]
        data.append(model_info)

    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Prefer VLM engine if exists
    vlm_engine = engine_manager.get_vlm_engine(request.model)
    if vlm_engine is not None:
        if request.stream:
            async def vlm_event_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4()}"
                initial_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(initial_chunk)}\n\n"

                async for piece in vlm_engine.stream_generate(request.messages, {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "max_tokens": request.max_tokens,
                }):
                    content_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"

                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(vlm_event_stream(), media_type="text/event-stream")

        text_response, usage = vlm_engine.generate(request.messages, {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens,
        })
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": text_response},
                "finish_reason": "stop",
            }],
            usage=usage,
        )

    # Otherwise, use Venus C engine
    engine = engine_manager.get_engine(request.model)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    prompt = format_chat_prompt(request.messages)

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

    response_text = engine.generate(prompt, config)
    prompt_tokens = engine.count_tokens(prompt)
    completion_tokens = engine.count_tokens(response_text)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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


def extract_first_image(messages: List[ChatMessage]) -> Optional[Image.Image]:
    for msg in reversed(messages):
        if msg.attachments:
            for att in msg.attachments:
                if att.kind == "image" and att.data:
                    try:
                        raw = base64.b64decode(att.data)
                        return Image.open(io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        continue
    return None


def extract_images_and_hashes(messages: List[ChatMessage]) -> (List[Image.Image], List[str]):
    images: List[Image.Image] = []
    hashes: List[str] = []
    for msg in messages:
        if not msg.attachments:
            continue
        for att in msg.attachments:
            if att.kind == "image" and att.data:
                try:
                    raw = base64.b64decode(att.data)
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    images.append(img)
                    hashes.append(hashlib.sha1(raw).hexdigest())
                except Exception:
                    continue
    return images, hashes


# HF VLM implementation
class HFVLMEngine:
    def __init__(self, model_dir: str):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available")
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        # Select device and dtype for speed
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_dir, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self._image_cache: Dict[str, Image.Image] = {}
        # Prefer channels_last for MPS speedups
        try:
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def generate(self, messages: List[ChatMessage], gen_cfg: Dict[str, Any]):
        # Build chat with image placeholders using apply_chat_template
        chat: List[Dict[str, Any]] = []
        for m in messages:
            content_items: List[Dict[str, Any]] = []
            if m.attachments:
                for att in m.attachments:
                    if att.kind == "image":
                        content_items.append({"type": "image"})
            if m.content:
                content_items.append({"type": "text", "text": m.content})
            if content_items:
                role = m.role if m.role in {"system", "user", "assistant"} else "user"
                chat.append({"role": role, "content": content_items})

        prompt_text = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        images, hashes = extract_images_and_hashes(messages)
        # Cache decoded PIL to avoid repeated decode cost across turns
        cache_hits = 0
        cached_images: List[Image.Image] = []
        for img, h in zip(images, hashes):
            if h in self._image_cache:
                cached_images.append(self._image_cache[h])
                cache_hits += 1
            else:
                self._image_cache[h] = img
                cached_images.append(img)

        # Resize large images to max side for speed
        resized: List[Image.Image] = []
        for img in cached_images:
            w, h = img.size
            s = max(w, h)
            if s > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / float(s)
                resized.append(img.resize((int(w * scale), int(h * scale))))
            else:
                resized.append(img)

        inputs = self.processor(text=prompt_text, images=(resized or None), return_tensors="pt")
        try:
            inputs = inputs.to(self.device)  # type: ignore[attr-defined]
        except Exception:
            # Manually move tensors if BatchFeature doesn't expose .to()
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_tokens", 256)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
            )
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        # Heuristic: strip echoed prompt
        if text.startswith(prompt_text):
            text = text[len(prompt_text):].lstrip()
        # If template fragments leaked, keep content after last 'assistant' marker
        marker = "\nassistant\n"
        if marker in text:
            text = text.split(marker)[-1].lstrip()
        # Rough token accounting without a tokenizer
        ptoks = len(prompt_text.split())
        usage = {
            "prompt_tokens": ptoks,
            "completion_tokens": len(text.split()),
            "total_tokens": ptoks + len(text.split()),
        }
        return text, usage

    async def stream_generate(self, messages: List[ChatMessage], gen_cfg: Dict[str, Any]):
        chat: List[Dict[str, Any]] = []
        for m in messages:
            content_items: List[Dict[str, Any]] = []
            if m.attachments:
                for att in m.attachments:
                    if att.kind == "image":
                        content_items.append({"type": "image"})
            if m.content:
                content_items.append({"type": "text", "text": m.content})
            if content_items:
                role = m.role if m.role in {"system", "user", "assistant"} else "user"
                chat.append({"role": role, "content": content_items})

        prompt_text = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        images, hashes = extract_images_and_hashes(messages)
        cached_images: List[Image.Image] = []
        for img, h in zip(images, hashes):
            cached_images.append(self._image_cache.get(h, img))

        resized: List[Image.Image] = []
        for img in cached_images:
            w, h = img.size
            s = max(w, h)
            if s > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / float(s)
                resized.append(img.resize((int(w * scale), int(h * scale))))
            else:
                resized.append(img)

        inputs = self.processor(text=prompt_text, images=(resized or None), return_tensors="pt")
        try:
            inputs = inputs.to(self.device)
        except Exception:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_tokens", 256)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            streamer=streamer,
        )

        import threading

        def _worker():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = threading.Thread(target=_worker)
        thread.start()

        for token in streamer:
            yield token

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
    model_dir = getattr(app.state, "model_dir", MODEL_DIR)
    print(f"[startup] using model directory {model_dir}")

    loaded = load_models_from_manifest(model_dir)
    if loaded:
        return

    # Fallback to legacy demo model loading if manifest missing
    demo_model_path = model_dir / "demo_model.bin"
    if demo_model_path.exists():
        try:
            engine_manager.load_model("demo-model", str(demo_model_path))
            print(f"[startup] Loaded fallback model: demo-model")
        except Exception as e:
            print(f"[startup] Failed to load demo model: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Venus Inference Engine API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-dir", default="./models", help="Directory containing models")
    
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir).resolve()
    app.state.model_dir = MODEL_DIR
    print(f"[server] Starting Venus API server with model directory {MODEL_DIR}")

    uvicorn.run(app, host=args.host, port=args.port)
