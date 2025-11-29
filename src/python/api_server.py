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
from contextlib import contextmanager

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
TORCH_COMPILE_POLICY = os.environ.get("VENUS_TORCH_COMPILE", "").strip().lower()
DISABLE_WARMUP = os.environ.get("VENUS_DISABLE_WARMUP", "").strip().lower() in {"1", "true", "yes"}


def configure_torch_runtime() -> None:
    if not TRANSFORMERS_AVAILABLE:
        return
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.set_float32_matmul_precision("high")
        threads_env = os.environ.get("VENUS_TORCH_THREADS")
        if threads_env:
            threads = max(1, int(threads_env))
            torch.set_num_threads(threads)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, min(threads, os.cpu_count() or threads)))
    except Exception as exc:  # pragma: no cover - best effort tuning
        print(f"[startup] torch runtime tuning skipped: {exc}")


def _select_attn_implementation() -> Optional[str]:
    if not TRANSFORMERS_AVAILABLE:
        return None
    impl = "sdpa"
    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore

        if torch.cuda.is_available() and is_flash_attn_2_available():
            impl = "flash_attention_2"
    except Exception:
        pass
    return impl


def _should_compile_model() -> bool:
    if not TRANSFORMERS_AVAILABLE or not hasattr(torch, "compile"):
        return False
    if TORCH_COMPILE_POLICY in {"0", "false", "no", "off"}:
        return False
    if TORCH_COMPILE_POLICY in {"1", "true", "yes", "on"}:
        return True
    # Default: enable when CUDA available
    return TORCH_COMPILE_POLICY == "auto" and torch.cuda.is_available()


def _maybe_compile_model(model: "torch.nn.Module", label: str) -> "torch.nn.Module":
    if not _should_compile_model():
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)  # type: ignore[arg-type]
        print(f"[startup] torch.compile enabled for {label}")
        return compiled
    except Exception as exc:
        print(f"[startup] torch.compile disabled for {label}: {exc}")
        return model


@contextmanager
def _inference_context(device: "torch.device", dtype: "torch.dtype"):
    if not TRANSFORMERS_AVAILABLE:
        yield
        return
    with torch.inference_mode():
        if device.type == "cuda":
            target_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
            with torch.autocast("cuda", dtype=target_dtype):
                yield
        elif device.type == "mps":
            with torch.autocast("mps", dtype=torch.float16):
                yield
        else:
            yield


configure_torch_runtime()

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
    venus_options: Optional[Dict[str, Any]] = None

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
        self.hf_text_engines = {}
    
    def load_model(self, model_name: str, model_path: str, tokenizer_path: Optional[str] = None):
        try:
            self.engines[model_name] = VenusEngine(model_path, tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def get_engine(self, model_name: str) -> Optional[VenusEngine]:
        return self.engines.get(model_name)
    
    def list_models(self) -> List[str]:
        return list(self.engines.keys()) + list(self.vlm_engines.keys()) + list(self.hf_text_engines.keys())

    # HF VLM engines
    def load_vlm_model(self, model_name: str, model_dir: str):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/Pillow not available to load VLM model")
        try:
            self.vlm_engines[model_name] = HFVLMEngine(model_dir)
        except ImportError:
            # Fallback to CausalLM for VLM models that might be text-only
            self.load_hf_text_model(model_name, model_dir)

    def get_vlm_engine(self, model_name: str):
        return self.vlm_engines.get(model_name)

    def load_hf_text_model(self, model_name: str, model_dir: str):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available")
        self.hf_text_engines[model_name] = HFTxtEngine(model_dir)

    def get_hf_text_engine(self, model_name: str):
        return self.hf_text_engines.get(model_name)


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

        # LLM: prefer HF text if hf_model_dir is present; otherwise Venus C engine
        hf_txt_dir = cfg.get("hf_model_dir")
        if hf_txt_dir:
            hf_txt_path = resolve_artifact_path(model_dir, hf_txt_dir)
            if not hf_txt_path or not hf_txt_path.exists():
                print(f"[startup] skipping {model_name}: hf_model_dir {hf_txt_path} not found")
                continue
            try:
                engine_manager.load_hf_text_model(model_name, str(hf_txt_path))
                loaded_models[model_name] = dict(cfg)
                loaded_any = True
                print(f"[startup] loaded HF text LLM {model_name} from {hf_txt_path}")
            except Exception as exc:
                print(f"[startup] failed to load HF text LLM {model_name}: {exc}")
            continue

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
        # For simplicity, recursion is disabled in streaming mode
        # Optional per-request VLM max image side
        max_side_opt = None
        if request.venus_options and isinstance(request.venus_options.get("max_image_side"), (int, float, str)):
            try:
                max_side_opt = int(request.venus_options.get("max_image_side"))
            except Exception:
                max_side_opt = None
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
                }, max_image_side=max_side_opt):
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

        # Optional recursive controller for VLM via request.venus_options or manifest
        model_cfg = manifest_models.get(request.model, {})
        rec_cfg = None
        if request.venus_options and isinstance(request.venus_options.get("recursive_reasoning"), dict):
            rec_cfg = request.venus_options.get("recursive_reasoning")
        elif isinstance(model_cfg.get("recursive_reasoning"), dict):
            rec_cfg = model_cfg.get("recursive_reasoning")

        def _vlm_generate_once(msgs: List[ChatMessage]):
            return vlm_engine.generate(msgs, {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "max_tokens": request.max_tokens,
            }, max_image_side=max_side_opt)

        text_response: str
        usage: Dict[str, Any]

        if rec_cfg and rec_cfg.get("enabled"):
            max_depth = int(rec_cfg.get("max_depth", 3))
            beam_width = int(rec_cfg.get("beam_width", 1))
            # Simple recursive loop: append assistant hypotheses as new messages
            working_messages: List[ChatMessage] = list(request.messages)
            best_text: str = ""
            best_usage: Dict[str, Any] = {}
            for _ in range(max_depth):
                candidates: List[tuple[str, Dict[str, Any]]] = []
                for _b in range(max(1, beam_width)):
                    t, u = _vlm_generate_once(working_messages)
                    candidates.append((t, u))
                # Trivial selection: pick the longest candidate (proxy for completeness)
                text, u = max(candidates, key=lambda x: len(x[0]) if x[0] else 0)
                working_messages = working_messages + [ChatMessage(role="assistant", content=text)]
                best_text, best_usage = text, u
            text_response, usage = best_text, best_usage
        else:
            text_response, usage = _vlm_generate_once(request.messages)
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

    # Otherwise, try HF text engine, then Venus C engine
    hf_text_engine = engine_manager.get_hf_text_engine(request.model)
    engine = engine_manager.get_engine(request.model)
    if not hf_text_engine and not engine:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

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

    # Inspect BitNet config and surface a notice for now (no accelerated kernels yet)
    model_cfg = manifest_models.get(request.model, {})
    bit_cfg = None
    if request.venus_options and isinstance(request.venus_options.get("bitnet_b1_58"), dict):
        bit_cfg = request.venus_options.get("bitnet_b1_58")
    elif isinstance(model_cfg.get("bitnet_b1_58"), dict):
        bit_cfg = model_cfg.get("bitnet_b1_58")
    if bit_cfg and bit_cfg.get("enabled"):
        print(f"[warn] bitnet_b1_58 enabled for {request.model} but accelerated kernels are not yet available; using standard decode.")

    # Optional recursive controller for LLM via request.venus_options or manifest
    rec_cfg = None
    if request.venus_options and isinstance(request.venus_options.get("recursive_reasoning"), dict):
        rec_cfg = request.venus_options.get("recursive_reasoning")
    elif isinstance(model_cfg.get("recursive_reasoning"), dict):
        rec_cfg = model_cfg.get("recursive_reasoning")

    response_text: str
    prompt_tokens: int
    completion_tokens: int

    if hf_text_engine:
        # Hugging Face text engine path (uses message list)
        if rec_cfg and rec_cfg.get("enabled") and not request.stream:
            max_depth = int(rec_cfg.get("max_depth", 3))
            beam_width = int(rec_cfg.get("beam_width", 1))
            working_messages: List[ChatMessage] = list(request.messages)
            best_text: str = ""
            for _ in range(max_depth):
                candidates: List[str] = []
                for _b in range(max(1, beam_width)):
                    t = hf_text_engine.generate(working_messages, config)
                    candidates.append(t)
                best_text = max(candidates, key=lambda x: len(x) if x else 0)
                working_messages.append(ChatMessage(role="assistant", content=best_text))
            response_text = best_text
        else:
            response_text = hf_text_engine.generate(request.messages, config)
        
        prompt_text = hf_text_engine.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = hf_text_engine.count_tokens(prompt_text)
        completion_tokens = hf_text_engine.count_tokens(response_text)
    
    elif engine: # Venus C engine path (uses prompt string)
        prompt = format_chat_prompt(request.messages)

        if request.stream:
            return StreamingResponse(
                generate_stream(engine, prompt, config, request.model),
                media_type="text/event-stream"
            )

        def _generate_once_llm(p: str) -> str:
            return engine.generate(p, config)

        if rec_cfg and rec_cfg.get("enabled"):
            max_depth = int(rec_cfg.get("max_depth", 3))
            beam_width = int(rec_cfg.get("beam_width", 1))
            working_prompt = prompt
            best_text = ""
            for _ in range(max_depth):
                candidates: List[str] = []
                for _b in range(max(1, beam_width)):
                    t = _generate_once_llm(working_prompt)
                    candidates.append(t)
                best_text = max(candidates, key=lambda x: len(x) if x else 0)
                working_prompt = working_prompt + f"\nassistant: {best_text}"
            response_text = best_text
        else:
            response_text = _generate_once_llm(prompt)
        
        prompt_tokens = engine.count_tokens(prompt)
        completion_tokens = engine.count_tokens(response_text)
    else:
        raise HTTPException(status_code=500, detail="No valid engine found for model")


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
        self._attn_impl = _select_attn_implementation()
        self._model_label = f"vlm:{Path(model_dir).name}"
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
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=self._attn_impl,
        ).to(self.device)
        # Prefer channels_last for image-heavy workloads where supported
        try:
            base_model = base_model.to(memory_format=torch.channels_last)  # type: ignore[arg-type]
        except Exception:
            pass
        self.dtype = next(base_model.parameters()).dtype
        self.model = _maybe_compile_model(base_model, self._model_label)
        self.model.eval()
        self._image_cache: Dict[str, Image.Image] = {}
        # Cache of resized images keyed by (sha1, side_cap)
        self._resized_cache: Dict[str, Dict[int, Image.Image]] = {}

    def generate(self, messages: List[ChatMessage], gen_cfg: Dict[str, Any], *, max_image_side: Optional[int] = None):
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        print(f"[TIMER] apply_chat_template: {(t1 - t0) * 1000:.2f} ms")

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
        side_cap = int(max_image_side) if max_image_side else MAX_IMAGE_SIDE
        for img, h in zip(cached_images, hashes):
            # Check per-hash resized cache
            cached_by_side = self._resized_cache.get(h)
            if cached_by_side and side_cap in cached_by_side:
                resized.append(cached_by_side[side_cap])
                continue
            w, ih = img.size
            s = max(w, ih)
            if s > side_cap:
                scale = side_cap / float(s)
                rimg = img.resize((int(w * scale), int(ih * scale)))
            else:
                rimg = img
            resized.append(rimg)
            # Update cache
            if h not in self._resized_cache:
                self._resized_cache[h] = {}
            self._resized_cache[h][side_cap] = rimg
        t2 = time.perf_counter()
        print(f"[TIMER] image processing: {(t2 - t1) * 1000:.2f} ms")

        inputs = self.processor(text=prompt_text, images=(resized or None), return_tensors="pt")
        t3 = time.perf_counter()
        print(f"[TIMER] processor call: {(t3 - t2) * 1000:.2f} ms")
        try:
            inputs = inputs.to(self.device, self.dtype)  # type: ignore[attr-defined]
        except Exception:
            # Manually move tensors if BatchFeature doesn't expose .to()
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                        inputs[k] = v.to(self.device, dtype=self.dtype)
                    else:
                        inputs[k] = v.to(self.device)
        t4 = time.perf_counter()
        print(f"[TIMER] inputs.to(device): {(t4 - t3) * 1000:.2f} ms")
        with _inference_context(self.device, self.dtype):
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_tokens", 256)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
            )
        t5 = time.perf_counter()
        print(f"[TIMER] model.generate: {(t5 - t4) * 1000:.2f} ms")
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
        t6 = time.perf_counter()
        print(f"[TIMER] decode and usage: {(t6 - t5) * 1000:.2f} ms")
        return text, usage

    async def stream_generate(self, messages: List[ChatMessage], gen_cfg: Dict[str, Any], *, max_image_side: Optional[int] = None):
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        print(f"[TIMER] stream apply_chat_template: {(t1 - t0) * 1000:.2f} ms")

        images, hashes = extract_images_and_hashes(messages)
        cached_images: List[Image.Image] = []
        for img, h in zip(images, hashes):
            cached_images.append(self._image_cache.get(h, img))

        resized: List[Image.Image] = []
        side_cap = int(max_image_side) if max_image_side else MAX_IMAGE_SIDE
        for img, h in zip(cached_images, hashes):
            cached_by_side = self._resized_cache.get(h)
            if cached_by_side and side_cap in cached_by_side:
                resized.append(cached_by_side[side_cap])
                continue
            w, ih = img.size
            s = max(w, ih)
            if s > side_cap:
                scale = side_cap / float(s)
                rimg = img.resize((int(w * scale), int(ih * scale)))
            else:
                rimg = img
            resized.append(rimg)
            if h not in self._resized_cache:
                self._resized_cache[h] = {}
            self._resized_cache[h][side_cap] = rimg
        t2 = time.perf_counter()
        print(f"[TIMER] stream image processing: {(t2 - t1) * 1000:.2f} ms")

        inputs = self.processor(text=prompt_text, images=(resized or None), return_tensors="pt")
        t3 = time.perf_counter()
        print(f"[TIMER] stream processor call: {(t3 - t2) * 1000:.2f} ms")
        try:
            inputs = inputs.to(self.device, self.dtype)
        except Exception:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                        inputs[k] = v.to(self.device, dtype=self.dtype)
                    else:
                        inputs[k] = v.to(self.device)
        t4 = time.perf_counter()
        print(f"[TIMER] stream inputs.to(device): {(t4 - t3) * 1000:.2f} ms")

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
            with _inference_context(self.device, self.dtype):
                t5 = time.perf_counter()
                self.model.generate(**gen_kwargs)
                t6 = time.perf_counter()
                print(f"[TIMER] stream model.generate: {(t6 - t5) * 1000:.2f} ms")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        for token in streamer:
            yield token


class HFTxtEngine:
    def __init__(self, model_dir: str):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available")
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Explicitly trust remote code for all text models
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Device and dtype
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            dtype = torch.float16  # bfloat16 is not fully supported on MPS
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32
        self._attn_impl = _select_attn_implementation()
        # Load tokenizer and model (low_cpu_mem_usage to help large checkpoints)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.tokenizer, "padding_side", None):
            self.tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=self._attn_impl,
        ).to(self.device)
        self.dtype = next(base_model.parameters()).dtype
        self.model = _maybe_compile_model(base_model, f"llm:{Path(model_dir).name}")
        self.model.eval()
        self._warm_start_done = False
        if not DISABLE_WARMUP:
            self._run_warmup()

    def _run_warmup(self) -> None:
        prompt = "Warmup request."
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            try:
                inputs = inputs.to(self.device)  # type: ignore[attr-defined]
            except Exception:
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to(self.device)
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )
            t0 = time.perf_counter()
            with _inference_context(self.device, self.dtype):
                self.model.generate(**gen_kwargs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()
            self._warm_start_done = True
            print(f"[warmup] hf text engine ready on {self.device} ({(t1 - t0) * 1000:.1f} ms)")
        except Exception as exc:  # pragma: no cover - warmup best effort
            print(f"[warmup] skipped for HF text engine: {exc}")

    def generate(self, messages: List[ChatMessage], gen_cfg: GenerationConfig) -> str:
        # Convert Pydantic models to dicts for the template
        chat_dicts = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            chat_dicts, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            inputs = inputs.to(self.device)  # type: ignore[attr-defined]
        except Exception:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)
        input_ids_len = inputs["input_ids"].shape[-1]
        
        do_sample = bool(gen_cfg.temperature and gen_cfg.temperature > 0.0)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(gen_cfg.max_tokens),
            do_sample=do_sample,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=float(gen_cfg.temperature),
                top_p=float(gen_cfg.top_p),
                top_k=int(gen_cfg.top_k),
            )
        with _inference_context(self.device, self.dtype):
            output = self.model.generate(**gen_kwargs)
        
        generated_tokens = output[0][len(inputs["input_ids"][0]) :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


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
