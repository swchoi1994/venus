"""
Venus - Ultra-Fast Universal LLM Inference Engine
As easy as vLLM, as fast as Groq
"""

from venus.core import LLM, SamplingParams, CompletionOutput
from venus.models import ModelRegistry, get_model_list

__version__ = "0.1.0"
__all__ = [
    "LLM",
    "SamplingParams", 
    "CompletionOutput",
    "ModelRegistry",
    "get_model_list",
]

# Auto-detect platform and load optimized backend
import platform
import warnings

def _get_platform_info():
    """Get current platform information"""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

# Print platform info on import (can be disabled)
import os
if os.environ.get("VENUS_QUIET") != "1":
    info = _get_platform_info()
    print(f"Venus {__version__} on {info['system']} {info['machine']}")