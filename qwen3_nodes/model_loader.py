"""
Qwen3-TTS Model Loader Node for ComfyUI.
Loads and caches the Qwen3-TTS model for use by other nodes.
"""

import torch
import os
from typing import Tuple, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_utils.constants import (
    MODEL_CONFIGS,
    DTYPE_OPTIONS,
    ATTENTION_OPTIONS,
    DEVICE_OPTIONS,
)
from qwen3_utils.model_cache import Qwen3TTSModelCache


class Qwen3TTSModelLoader:
    """
    Model loader node for Qwen3-TTS.
    Loads the model with specified configuration and caches it for reuse.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list(MODEL_CONFIGS.keys()), {
                    "default": "CustomVoice-1.7B",
                    "tooltip": "Select the Qwen3-TTS model variant"
                }),
                "dtype": (DTYPE_OPTIONS, {
                    "default": "bfloat16",
                    "tooltip": "Data type for model weights"
                }),
                "attention": (ATTENTION_OPTIONS, {
                    "default": "sdpa",
                    "tooltip": "Attention mechanism (flash_attention_2 requires Ampere+ GPU)"
                }),
                "device": (DEVICE_OPTIONS, {
                    "default": "auto",
                    "tooltip": "Device to load model on"
                }),
            },
        }

    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/Qwen3-TTS"

    def load_model(
        self,
        model_type: str,
        dtype: str,
        attention: str,
        device: str
    ) -> Tuple[Any]:
        """
        Load the Qwen3-TTS model with caching.

        Returns:
            Tuple containing the loaded model wrapper
        """
        import comfy.model_management as mm

        # Check for cancellation
        mm.throw_exception_if_processing_interrupted()

        print("=" * 60)
        print("🎤 Qwen3-TTS Model Loader")
        print("=" * 60)
        print(f"📦 Model type: {model_type}")
        print(f"🔢 Dtype: {dtype}")
        print(f"⚡ Attention: {attention}")
        print(f"💻 Device: {device}")

        # Check if model is already cached
        cached_model = Qwen3TTSModelCache.get(model_type, dtype, attention, device)
        if cached_model is not None:
            print("✅ Using cached model")
            print("=" * 60)
            return (cached_model,)

        # Validate device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔄 Auto-detected device: {device}")

        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")
            device = "cpu"

        # Get model configuration
        config = MODEL_CONFIGS[model_type]
        model_id = config["model_id"]

        print(f"📥 Loading model from: {model_id}")
        print(f"📝 {config['description']}")

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        try:
            # Import qwen-tts library
            from qwen_tts import Qwen3TTSModel

            print("⏳ Loading model (this may take a few minutes on first run)...")

            # Check for cancellation before loading
            mm.throw_exception_if_processing_interrupted()

            # Load the model
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation=attention,
                device_map=device if device != "cpu" else None,
            )

            # If device is cpu, move model explicitly
            if device == "cpu":
                model = model.to("cpu")

            # Create model wrapper
            model_wrapper = Qwen3TTSModelWrapper(
                model=model,
                model_type=model_type,
                dtype=dtype,
                attention=attention,
                device=device,
            )

            # Cache the model
            Qwen3TTSModelCache.set(model_type, dtype, attention, device, model_wrapper)

            print("✅ Model loaded successfully")
            print("=" * 60)

            return (model_wrapper,)

        except ImportError as e:
            raise ImportError(
                f"Failed to import qwen-tts library.\n"
                f"Please install it with: pip install qwen-tts\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen3-TTS model.\n"
                f"Model: {model_id}\n"
                f"Error: {e}"
            )


class Qwen3TTSModelWrapper:
    """
    Wrapper class for Qwen3-TTS model.
    Holds the model and its configuration for easy access.
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        dtype: str,
        attention: str,
        device: str
    ):
        self.model = model
        self.model_type = model_type
        self.dtype = dtype
        self.attention = attention
        self.device = device

    def __repr__(self):
        return f"Qwen3TTSModelWrapper(type={self.model_type}, dtype={self.dtype}, device={self.device})"


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3TTS_ModelLoader": Qwen3TTSModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS_ModelLoader": "Qwen3-TTS Model Loader",
}
