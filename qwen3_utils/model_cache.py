"""
Model caching system for Qwen3-TTS.
Prevents redundant model loading and manages VRAM efficiently.
"""

import gc
import torch
from typing import Dict, Optional, Any


class Qwen3TTSModelCache:
    """
    Global cache for Qwen3-TTS models.
    Uses singleton pattern to ensure only one cache exists.
    """

    _instance: Optional['Qwen3TTSModelCache'] = None
    _cache: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_cache_key(
        cls,
        model_type: str,
        dtype: str,
        attention: str,
        device: str
    ) -> str:
        """Generate a unique cache key for the model configuration."""
        return f"{model_type}_{dtype}_{attention}_{device}"

    @classmethod
    def get(
        cls,
        model_type: str,
        dtype: str,
        attention: str,
        device: str
    ) -> Optional[Any]:
        """Get a cached model if it exists."""
        cache_key = cls.get_cache_key(model_type, dtype, attention, device)
        return cls._cache.get(cache_key)

    @classmethod
    def set(
        cls,
        model_type: str,
        dtype: str,
        attention: str,
        device: str,
        model: Any
    ) -> None:
        """Cache a model."""
        cache_key = cls.get_cache_key(model_type, dtype, attention, device)
        cls._cache[cache_key] = model
        print(f"💾 Cached model: {cache_key}")

    @classmethod
    def clear(cls, force: bool = False) -> None:
        """
        Clear the model cache and free VRAM.

        Args:
            force: If True, clear all cached models regardless of usage
        """
        if not cls._cache:
            return

        print(f"🗑️ Clearing model cache ({len(cls._cache)} models)...")

        for key, model in list(cls._cache.items()):
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    # Move model to CPU first to free GPU memory
                    model.model.to('cpu')
                del model
            except Exception as e:
                print(f"⚠️ Warning: Error clearing model {key}: {e}")

        cls._cache.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✅ Model cache cleared")

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            "count": len(cls._cache),
            "keys": list(cls._cache.keys())
        }


def cleanup_vram():
    """
    Clean up VRAM using ComfyUI's memory management.
    """
    import comfy.model_management as mm

    print("🗑️ Cleaning up VRAM...")

    # Unload all models from VRAM
    mm.unload_all_models()

    # Clear ComfyUI's internal cache
    mm.soft_empty_cache()

    # Python garbage collection
    gc.collect()

    # Clear CUDA caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("✅ VRAM cleanup complete")
