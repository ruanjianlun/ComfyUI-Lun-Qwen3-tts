"""
Qwen3-TTS Voice Clone Node for ComfyUI.
Clones voices from reference audio samples.
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


class Qwen3TTSVoiceClone:
    """
    Voice Clone node for Qwen3-TTS.
    Clones voices from reference audio samples (3+ seconds recommended).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "你好,这是克隆后的声音。",
                    "dynamicPrompts": False,
                    "tooltip": "Text to synthesize"
                }),
                "ref_audio": ("AUDIO", {
                    "tooltip": "Reference audio for voice cloning (3+ seconds recommended)"
                }),
                "ref_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Text spoken in the reference audio (improves quality)"
                }),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {
                    "default": "Auto",
                    "tooltip": "Language for synthesis"
                }),
                "x_vector_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use only speaker embedding (faster, less accurate)"
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
        ref_audio: dict,
        ref_text: str,
        language: str = "Auto",
        x_vector_only: bool = False,
        seed: int = 0,
        temperature: float = 0.9,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        keep_model_in_vram: bool = True,
    ) -> Tuple[dict]:
        """
        Generate speech by cloning a voice from reference audio.

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
        print("🎙️ Qwen3-TTS Voice Clone Generation")
        print("=" * 60)
        print(f"🎲 Seed: {actual_seed}")
        print(f"🌐 Language: {language}")
        print(f"📝 Text: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"🎤 Reference text: {ref_text[:40]}{'...' if len(ref_text) > 40 else ''}")
        print(f"🔌 X-vector only: {x_vector_only}")
        print(f"🌡️ Temperature: {temperature}, Top-p: {top_p}")
        print(f"📏 Max tokens: {max_new_tokens}")

        # Set seed
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        try:
            # Check for cancellation
            mm.throw_exception_if_processing_interrupted()

            # Process reference audio
            ref_waveform = ref_audio.get("waveform")
            ref_sample_rate = ref_audio.get("sample_rate", SAMPLE_RATE)

            if ref_waveform is None:
                raise ValueError("Reference audio has no waveform data")

            # Convert to torch tensor if needed
            if isinstance(ref_waveform, list):
                ref_waveform = np.array(ref_waveform)
            if isinstance(ref_waveform, np.ndarray):
                ref_waveform = torch.from_numpy(ref_waveform).float()

            # Ensure we have a tensor
            if not isinstance(ref_waveform, torch.Tensor):
                raise ValueError(f"Unsupported waveform type: {type(ref_waveform)}")

            # Resample if needed
            if ref_sample_rate != SAMPLE_RATE:
                print(f"🔄 Resampling reference audio from {ref_sample_rate}Hz to {SAMPLE_RATE}Hz...")
                try:
                    import torchaudio.transforms as T
                    resampler = T.Resample(ref_sample_rate, SAMPLE_RATE)
                    ref_waveform = resampler(ref_waveform)
                except ImportError:
                    print("⚠️ torchaudio not available, skipping resampling")

            # Remove batch and channel dimensions if present
            if ref_waveform.dim() == 3:
                ref_waveform = ref_waveform.squeeze(0).squeeze(0)
            elif ref_waveform.dim() == 2:
                ref_waveform = ref_waveform.squeeze(0)

            # Convert to numpy array (1D float32)
            ref_audio_np = ref_waveform.cpu().numpy().astype(np.float32).flatten()

            # Debug: verify it's a numpy array
            print(f"DEBUG: ref_audio_np type={type(ref_audio_np)}, dtype={ref_audio_np.dtype}, shape={ref_audio_np.shape}")

            ref_duration = len(ref_audio_np) / SAMPLE_RATE
            print(f"🎤 Reference audio duration: {ref_duration:.2f}s")

            if ref_duration < 3.0:
                print("⚠️ Warning: Reference audio is less than 3 seconds. Quality may be reduced.")

            # Check for cancellation
            mm.throw_exception_if_processing_interrupted()

            print("🎵 Generating speech with cloned voice...")

            # When x_vector_only=False (ICL mode), ref_text is required
            # If ref_text is empty, automatically use x_vector_only=True
            effective_x_vector_only = x_vector_only
            if not x_vector_only and not ref_text:
                print("⚠️ ref_text is required for ICL mode. Using x_vector_only=True instead.")
                effective_x_vector_only = True

            # Generate using the model
            # Returns: Tuple[List[np.ndarray], int] - (list of audio arrays, sample_rate)
            # ref_audio needs to be a list of tuples [(numpy_array, sample_rate), ...]
            # Each audio_array must be numpy array, NOT list
            audio_list, output_sr = model.model.generate_voice_clone(
                text=text,
                ref_audio=[(ref_audio_np, ref_sample_rate)],
                ref_text=ref_text if ref_text else None,
                language=language if language != "Auto" else None,
                x_vector_only_mode=effective_x_vector_only,
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
    "Qwen3TTS_VoiceClone": Qwen3TTSVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS_VoiceClone": "Qwen3-TTS Voice Clone",
}
