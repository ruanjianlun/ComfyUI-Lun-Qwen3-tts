"""
Qwen3-TTS Voice Design Node for ComfyUI.
Creates custom voices from natural language descriptions.
"""

import torch
import numpy as np
import random
from typing import Tuple, Any, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_utils.constants import (
    SUPPORTED_LANGUAGES,
    SAMPLE_RATE,
)
from qwen3_utils.model_cache import Qwen3TTSModelCache

# Type checking import to avoid circular dependency
if TYPE_CHECKING:
    from .model_loader import Qwen3TTSModelWrapper


class Qwen3TTSVoiceDesign:
    """
    Voice Design node for Qwen3-TTS.
    Creates custom voices from natural language descriptions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "你好,这是我的声音。",
                    "dynamicPrompts": False,
                    "tooltip": "Text to synthesize"
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "A warm and friendly female voice with a slight British accent, speaking at a moderate pace.",
                    "dynamicPrompts": False,
                    "tooltip": "Natural language description of the desired voice"
                }),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {
                    "default": "Auto",
                    "tooltip": "Language for synthesis"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature"
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p sampling"
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 100,
                    "max": 16000,
                    "step": 100,
                    "tooltip": "Maximum tokens to generate"
                }),
                "keep_model_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM for faster subsequent generations"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/Qwen3-TTS"

    def generate_speech(
        self,
        model,  # Qwen3TTSModelWrapper
        text: str,
        voice_description: str,
        language: str = "Auto",
        seed: int = 0,
        temperature: float = 0.9,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        keep_model_in_vram: bool = True,
    ) -> Tuple[dict]:
        """
        Generate speech with a custom voice designed from text description.

        Returns:
            Tuple containing audio dictionary for ComfyUI
        """
        import comfy.model_management as mm

        # Check for cancellation
        mm.throw_exception_if_processing_interrupted()

        # Handle seed
        if seed == 0:
            actual_seed = random.randint(1, 0xffffffffffffffff)
        else:
            actual_seed = seed

        print("=" * 60)
        print("🎨 Qwen3-TTS Voice Design Generation")
        print("=" * 60)
        print(f"🎲 Seed: {actual_seed}")
        print(f"🌐 Language: {language}")
        print(f"🎭 Voice: {voice_description[:60]}{'...' if len(voice_description) > 60 else ''}")
        print(f"📝 Text: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"🌡️ Temperature: {temperature}, Top-p: {top_p}")
        print(f"📏 Max tokens: {max_new_tokens}")

        # Set seed
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        try:
            # Check for cancellation
            mm.throw_exception_if_processing_interrupted()

            print("🎵 Generating speech with voice design...")

            # Generate using the model
            # Note: generate_voice_design requires 'instruct' parameter (not 'voice_description')
            # Returns: Tuple[List[np.ndarray], int] - (list of audio arrays, sample_rate)
            audio_list, output_sr = model.model.generate_voice_design(
                text=text,
                instruct=voice_description,
                language=language if language != "Auto" else None,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

            # Check for cancellation
            mm.throw_exception_if_processing_interrupted()

            # Get first audio from the list
            audio_np = audio_list[0] if audio_list else np.array([])

            # Ensure float32 and normalize
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            elif audio_np.dtype == np.int32:
                audio_np = audio_np.astype(np.float32) / 2147483648.0

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_np).float()

            # Add batch and channel dimensions: [samples] -> [1, 1, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": output_sr,
            }

            duration = audio_tensor.shape[-1] / output_sr
            print(f"✅ Generated {duration:.2f}s of audio at {output_sr}Hz")
            print("=" * 60)

            # Handle VRAM management
            if not keep_model_in_vram:
                print("🗑️ Offloading model from VRAM...")
                Qwen3TTSModelCache.clear(force=True)

            return (audio_output,)

        except InterruptedError:
            print("\n🛑 Generation cancelled by user")
            raise
        except Exception as e:
            print(f"\n❌ Generation failed: {str(e)}")
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3TTS_VoiceDesign": Qwen3TTSVoiceDesign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS_VoiceDesign": "Qwen3-TTS Voice Design",
}
