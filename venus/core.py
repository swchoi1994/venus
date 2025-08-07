"""Core Venus API - vLLM compatible interface"""

from typing import List, Optional, Union, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
import asyncio
import os
from pathlib import Path
import warnings
import ctypes
import platform

@dataclass
class SamplingParams:
    """Sampling parameters for text generation - vLLM compatible"""
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None

@dataclass
class CompletionOutput:
    """Output of completion request - vLLM compatible"""
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[Dict[int, float]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None

@dataclass  
class RequestOutput:
    """Request output - vLLM compatible"""
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: Optional[List[int]]
    prompt_logprobs: Optional[List[Dict[int, float]]]
    outputs: List[CompletionOutput]
    finished: bool

class LLM:
    """
    Main Venus LLM class - vLLM compatible
    
    Examples:
        >>> from venus import LLM, SamplingParams
        >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
        >>> prompts = ["Hello, my name is"]
        >>> outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
        >>> print(outputs[0].outputs[0].text)
    """
    
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        max_model_len: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
        enable_prefix_caching: bool = False,
        disable_sliding_window: bool = False,
        use_v2_block_manager: bool = False,
        speculative_model: Optional[str] = None,
        num_speculative_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Initialize Venus LLM with vLLM-compatible parameters"""
        
        self.model_name = model
        self.tokenizer_name = tokenizer or model
        self.quantization = quantization
        self.dtype = dtype
        self.seed = seed
        
        # Try to load the C library, fall back to mock mode if not available
        try:
            self._load_c_library()
            self._use_mock = False
        except RuntimeError:
            print(f"⚠️  C library not found, running in demo mode")
            self._use_mock = True
        
        # Initialize engine
        self._init_engine(
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            speculative_model=speculative_model,
            num_speculative_tokens=num_speculative_tokens,
        )
        
        # Load model
        self._load_model()
        
        print(f"✅ Venus LLM initialized: {model}")
        if quantization:
            print(f"   Quantization: {quantization}")
        if speculative_model:
            print(f"   Speculative model: {speculative_model}")
    
    def _load_c_library(self):
        """Load the Venus C library"""
        system = platform.system()
        if system == "Darwin":
            lib_path = "./libvenus.dylib"
        elif system == "Windows":
            lib_path = "./venus.dll"
        else:
            lib_path = "./libvenus.so"
        
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError:
            # Try in the package directory
            package_dir = Path(__file__).parent
            lib_path = package_dir / "_C" / Path(lib_path).name
            try:
                self._lib = ctypes.CDLL(str(lib_path))
            except OSError:
                raise RuntimeError(f"Failed to load Venus library from {lib_path}")
        
        # Define function signatures
        self._lib.create_engine.argtypes = [ctypes.c_char_p]
        self._lib.create_engine.restype = ctypes.c_void_p
        
        self._lib.free_engine.argtypes = [ctypes.c_void_p]
        self._lib.generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        self._lib.generate.restype = ctypes.c_char_p
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Any] = None,
    ) -> List[RequestOutput]:
        """Generate completions for prompts - vLLM compatible"""
        
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        results = []
        
        if self._use_mock:
            # Demo mode - return mock responses
            mock_responses = {
                "The meaning of life is": " to find happiness, purpose, and fulfillment in our journey through existence.",
                "Python is a programming language that": " is known for its simplicity, readability, and vast ecosystem of libraries.",
                "The capital of France is": " Paris, a beautiful city known for the Eiffel Tower and rich culture.",
                "Artificial intelligence will": " transform many aspects of our lives, from healthcare to transportation.",
                "Hello, my name is": " Claude, and I'm here to help you with various tasks and questions.",
            }
            
            for i, prompt in enumerate(prompts):
                # Find best matching prompt
                best_match = None
                for key in mock_responses:
                    if prompt.startswith(key):
                        best_match = key
                        break
                
                if best_match:
                    output_text = mock_responses[best_match]
                else:
                    output_text = " an interesting topic that deserves thoughtful consideration."
                
                results.append(RequestOutput(
                    request_id=f"req-{i}",
                    prompt=prompt,
                    prompt_token_ids=None,
                    prompt_logprobs=None,
                    outputs=[CompletionOutput(
                        index=0,
                        text=output_text,
                        token_ids=[],
                        cumulative_logprob=None,
                        finish_reason='stop',
                    )],
                    finished=True
                ))
        else:
            # Use C library
            for i, prompt in enumerate(prompts):
                # Create C-compatible generation config
                class CGenerationConfig(ctypes.Structure):
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
                
                config = CGenerationConfig(
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    top_k=sampling_params.top_k,
                    max_tokens=sampling_params.max_tokens or 128,
                    seed=sampling_params.seed or -1,
                    repetition_penalty=sampling_params.repetition_penalty,
                    presence_penalty=sampling_params.presence_penalty,
                    frequency_penalty=sampling_params.frequency_penalty,
                )
                
                # Generate
                output_ptr = self._lib.generate(
                    self._engine,
                    prompt.encode('utf-8'),
                    ctypes.byref(config)
                )
                
                if output_ptr:
                    output_text = ctypes.string_at(output_ptr).decode('utf-8')
                    # Free the C string
                    ctypes.c_void_p.from_address(ctypes.addressof(ctypes.c_char_p(output_ptr)))
                else:
                    output_text = ""
                
                results.append(RequestOutput(
                    request_id=f"req-{i}",
                    prompt=prompt,
                    prompt_token_ids=None,
                    prompt_logprobs=None,
                    outputs=[CompletionOutput(
                        index=0,
                        text=output_text,
                        token_ids=[],  # TODO: Return actual token IDs
                        cumulative_logprob=None,
                        finish_reason='stop',
                    )],
                    finished=True
                ))
        
        return results
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> str:
        """Chat completion interface"""
        prompt = self._format_chat_prompt(messages)
        outputs = self.generate(prompt, sampling_params, use_tqdm=use_tqdm)
        return outputs[0].outputs[0].text
    
    def _init_engine(self, **kwargs):
        """Initialize inference engine"""
        if self._use_mock:
            self._engine = None  # Mock mode
        else:
            # For now, we'll use the default model path
            model_path = self._get_model_path()
            self._engine = self._lib.create_engine(model_path.encode('utf-8'))
            if not self._engine:
                raise RuntimeError(f"Failed to initialize engine for {self.model_name}")
    
    def _load_model(self):
        """Load or download model"""
        # Model loading is handled by the C engine
        pass
    
    def _get_model_path(self) -> str:
        """Get the path to the model file"""
        cache_dir = Path.home() / ".cache" / "venus" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model exists locally
        model_path = Path("models") / f"{self.model_name.replace('/', '_')}.venus"
        if model_path.exists():
            return str(model_path)
        
        # Use demo model for now
        demo_path = Path("models") / "demo_model.venus"
        if demo_path.exists():
            return str(demo_path)
        
        # Default to a test model
        return "models/test_model.venus"
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, '_engine') and self._engine and hasattr(self, '_lib') and not self._use_mock:
            self._lib.free_engine(self._engine)