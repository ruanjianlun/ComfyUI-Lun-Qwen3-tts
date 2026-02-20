"""
Constants for Qwen3-TTS nodes.
"""

# Available pre-defined speakers for Custom Voice mode
CUSTOM_VOICE_SPEAKERS = [
    "Vivian",      # 拽拽的、可爱的小暴躁 (女性)
    "Serena",      # 温柔小姐姐 (女性)
    "Uncle_Fu",    # 田叔 - 沙哑烟嗓 (男性)
    "Dylan",       # 北京-晓东 - 北京胡同少年 (男性)
    "Eric",        # 四川-程川 - 跳脱市井的四川男子 (男性)
    "Ryan",        # 甜茶 - 戏感炸裂 (男性)
    "Aiden",       # 艾登 - 精通厨艺的美语大男孩 (男性)
    "Ono_Anna",    # 小野杏 - 鬼灵精怪的青梅竹马 (女性)
    "Sohee",       # 素熙 - 温柔开朗的韩国欧尼 (女性)
]

# Supported languages
SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Russian",
    "Italian",
    "Spanish",
    "Portuguese",
]

# Model configurations
MODEL_CONFIGS = {
    "CustomVoice-1.7B": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "description": "Pre-defined speakers (9 voices)",
    },
    "VoiceDesign-1.7B": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "description": "Voice design from text description",
    },
    "Base-1.7B": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "description": "Base model for voice cloning",
    },
}

# Audio output settings
SAMPLE_RATE = 24000  # Qwen3-TTS outputs 24kHz audio

# Dtype options
DTYPE_OPTIONS = ["bfloat16", "float16", "float32"]

# Attention mechanism options
ATTENTION_OPTIONS = ["flash_attention_2", "sdpa", "eager"]

# Device options
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
