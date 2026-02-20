# ComfyUI-Lun-Qwen3-tts

A ComfyUI custom node for **Qwen3-TTS** - Alibaba Cloud's state-of-the-art text-to-speech model.

## Features

- **Custom Voice**: Generate speech using 9 pre-defined speakers with different characteristics
- **Voice Design**: Create custom voices from natural language descriptions
- **Voice Clone**: Clone voices from reference audio samples (3+ seconds recommended)

## Installation

### Method 1: ComfyUI Manager (Recommended)

Search for "Qwen3-TTS" in ComfyUI Manager and install.

### Method 2: Manual Installation

1. Clone this repository to your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-Lun-Qwen3-tts.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using the ComfyUI portable Python:
```bash
# Windows
path\to\ComfyUI\python_embeded\python.exe -m pip install -r requirements.txt
```

3. Restart ComfyUI

## Nodes

### 1. Qwen3-TTS Model Loader

Loads the Qwen3-TTS model with configurable settings.

**Inputs:**
- `model_type`: Select model variant (CustomVoice-1.7B, VoiceDesign-1.7B, Base-1.7B)
- `dtype`: Data type (bfloat16, float16, float32)
- `attention`: Attention mechanism (flash_attention_2, sdpa, eager)
- `device`: Device (auto, cuda, cpu)

**Outputs:**
- `model`: Loaded model for use by other nodes

### 2. Qwen3-TTS Custom Voice

Generates speech using pre-defined speakers.

**Inputs:**
- `model`: Model from loader
- `text`: Text to synthesize
- `speaker`: Speaker selection (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
- `language`: Language (Auto, Chinese, English, Japanese, Korean)
- `instruct`: Optional emotion/style instruction
- `seed`: Random seed (0 = random)
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling
- `max_new_tokens`: Maximum tokens to generate
- `keep_model_in_vram`: Keep model in VRAM

**Outputs:**
- `audio`: Generated audio

### 3. Qwen3-TTS Voice Design

Creates custom voices from text descriptions.

**Inputs:**
- `model`: Model from loader (use VoiceDesign-1.7B)
- `text`: Text to synthesize
- `voice_description`: Natural language description of desired voice
- Other parameters same as Custom Voice

**Outputs:**
- `audio`: Generated audio

### 4. Qwen3-TTS Voice Clone

Clones voices from reference audio.

**Inputs:**
- `model`: Model from loader (use Base-1.7B)
- `text`: Text to synthesize
- `ref_audio`: Reference audio (3+ seconds recommended)
- `ref_text`: Text spoken in reference audio
- `x_vector_only`: Use only speaker embedding
- Other parameters same as Custom Voice

**Outputs:**
- `audio`: Generated audio

## Available Speakers

| Speaker | Description |
|---------|-------------|
| Vivian | Chinese female - warm and friendly |
| Serena | Chinese female - gentle and soft |
| Uncle_Fu | Chinese male - deep and mature |
| Dylan | Chinese male - Beijing dialect |
| Eric | Chinese male - Sichuan dialect |
| Ryan | English male - dramatic |
| Aiden | English male - natural American |
| Ono_Anna | Japanese female |
| Sohee | Korean female |

## Example Workflows

Example workflows are provided in the `workflows/` directory:

- `basic_custom_voice.json` - Basic TTS with pre-defined speaker
- `voice_design.json` - Create custom voice from description
- `voice_clone.json` - Clone voice from reference audio

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended) or CPU
- ~8GB VRAM for 1.7B model

## License

MIT License

## Credits

- Qwen3-TTS by Alibaba Cloud: https://github.com/QwenLM/Qwen3-TTS
- ComfyUI by ComfyAnonymous: https://github.com/comfyanonymous/ComfyUI

## Support

For issues and feature requests, please open an issue on GitHub.
