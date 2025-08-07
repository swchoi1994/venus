"""Model registry and management"""

SUPPORTED_MODELS = {
    "llama": ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"],
    "qwen": ["Qwen/Qwen2-7B", "Qwen/Qwen2-14B", "Qwen/Qwen2.5-0.5B"],
    "mistral": ["mistralai/Mistral-7B-v0.1"],
    "phi": ["microsoft/phi-2"],
}

class ModelRegistry:
    """Model registry for Venus"""
    
    def __init__(self):
        self.models = {}
    
    def register(self, name: str, model_class):
        """Register a model"""
        self.models[name] = model_class
    
    def get(self, name: str):
        """Get a model class"""
        return self.models.get(name)
    
    def list(self):
        """List all models"""
        return list(self.models.keys())

def get_model_list():
    """Get list of supported models"""
    all_models = []
    for models in SUPPORTED_MODELS.values():
        all_models.extend(models)
    return all_models