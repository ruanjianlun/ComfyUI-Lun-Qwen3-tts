"""
ComfyUI-Lun-Qwen3-tts - Qwen3-TTS Text-to-Speech custom node
"""

import os
import sys

# Ensure current directory is in path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import node mappings from submodules
try:
    from qwen3_nodes.model_loader import NODE_CLASS_MAPPINGS as _loader_mappings
    from qwen3_nodes.model_loader import NODE_DISPLAY_NAME_MAPPINGS as _loader_display
    from qwen3_nodes.custom_voice import NODE_CLASS_MAPPINGS as _voice_mappings
    from qwen3_nodes.custom_voice import NODE_DISPLAY_NAME_MAPPINGS as _voice_display
    from qwen3_nodes.voice_design import NODE_CLASS_MAPPINGS as _design_mappings
    from qwen3_nodes.voice_design import NODE_DISPLAY_NAME_MAPPINGS as _design_display
    from qwen3_nodes.voice_clone import NODE_CLASS_MAPPINGS as _clone_mappings
    from qwen3_nodes.voice_clone import NODE_DISPLAY_NAME_MAPPINGS as _clone_display

    # Combine all mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_CLASS_MAPPINGS.update(_loader_mappings)
    NODE_CLASS_MAPPINGS.update(_voice_mappings)
    NODE_CLASS_MAPPINGS.update(_design_mappings)
    NODE_CLASS_MAPPINGS.update(_clone_mappings)

    NODE_DISPLAY_NAME_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS.update(_loader_display)
    NODE_DISPLAY_NAME_MAPPINGS.update(_voice_display)
    NODE_DISPLAY_NAME_MAPPINGS.update(_design_display)
    NODE_DISPLAY_NAME_MAPPINGS.update(_clone_display)

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

    print("=" * 60)
    print("ComfyUI-Lun-Qwen3-tts loaded successfully!")
    print("Available nodes:")
    for name in NODE_CLASS_MAPPINGS.keys():
        print(f"  - {name}")
    print("=" * 60)

except Exception as e:
    import traceback
    print(f"[ComfyUI-Lun-Qwen3-tts] Error loading nodes:")
    traceback.print_exc()

    # Provide empty mappings to prevent ComfyUI crash
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
