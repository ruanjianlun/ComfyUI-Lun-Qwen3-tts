#!/usr/bin/env python3
"""
API-format workflow tests for Qwen3-TTS.
ComfyUI API expects a different format than the UI workflow JSON.
"""

import json
import time
import requests
import uuid
import sys

COMFYUI_URL = "http://127.0.0.1:8188"
TIMEOUT = 600

def queue_prompt(prompt_workflow):
    """Queue a workflow to ComfyUI."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": prompt_workflow, "client_id": client_id}

    response = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Failed to queue prompt: {response.text}")

    return response.json().get("prompt_id"), client_id

def wait_for_completion(prompt_id, timeout=TIMEOUT):
    """Wait for workflow execution."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    if "outputs" in prompt_data:
                        return True, prompt_data
                    if prompt_data.get("status", {}).get("status_str") == "error":
                        return False, prompt_data
            time.sleep(2)
        except requests.RequestException:
            time.sleep(2)

    return False, {"error": "Timeout"}


# API-format workflow for Basic Custom Voice
BASIC_CUSTOM_VOICE_WORKFLOW = {
    "1": {
        "class_type": "Qwen3TTS_ModelLoader",
        "inputs": {
            "model_type": "CustomVoice-1.7B",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "device": "auto"
        }
    },
    "2": {
        "class_type": "Qwen3TTS_CustomVoice",
        "inputs": {
            "model": ["1", 0],
            "text": "你好,欢迎使用千问语音合成。",
            "speaker": "Vivian",
            "language": "Auto",
            "instruct": "",
            "seed": 42,
            "temperature": 0.9,
            "top_p": 1.0,
            "max_new_tokens": 2048,
            "keep_model_in_vram": True
        }
    },
    "3": {
        "class_type": "SaveAudio",
        "inputs": {
            "audio": ["2", 0],
            "filename_prefix": "qwen3_tts_test"
        }
    }
}

# API-format workflow for Voice Design
VOICE_DESIGN_WORKFLOW = {
    "1": {
        "class_type": "Qwen3TTS_ModelLoader",
        "inputs": {
            "model_type": "VoiceDesign-1.7B",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "device": "auto"
        }
    },
    "2": {
        "class_type": "Qwen3TTS_VoiceDesign",
        "inputs": {
            "model": ["1", 0],
            "text": "Hello, this is a test of voice design.",
            "voice_description": "A warm and friendly female voice speaking clearly.",
            "language": "Auto",
            "seed": 42,
            "temperature": 0.9,
            "top_p": 1.0,
            "max_new_tokens": 2048,
            "keep_model_in_vram": True
        }
    },
    "3": {
        "class_type": "SaveAudio",
        "inputs": {
            "audio": ["2", 0],
            "filename_prefix": "qwen3_voice_design_test"
        }
    }
}


def test_workflow(name, workflow):
    """Test a single workflow."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

    try:
        print("Queueing workflow...")
        prompt_id, client_id = queue_prompt(workflow)
        print(f"Prompt ID: {prompt_id}")

        print("Waiting for execution...")
        success, result = wait_for_completion(prompt_id)

        if success:
            print("SUCCESS!")
            outputs = result.get("outputs", {})
            for node_id, output in outputs.items():
                if "audio" in output:
                    print(f"  Generated audio: {output['audio']}")
            return True
        else:
            print(f"FAILED: {result}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("="*60)
    print("Qwen3-TTS API Workflow Tests")
    print("="*60)

    # Check connection
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=10)
        if response.status_code != 200:
            print("ERROR: Cannot connect to ComfyUI")
            return 1
        print("ComfyUI is running")
    except Exception as e:
        print(f"ERROR: Cannot connect to ComfyUI: {e}")
        return 1

    # Run tests
    results = {}

    # Test 1: Basic Custom Voice
    results["Basic Custom Voice"] = test_workflow(
        "Basic Custom Voice",
        BASIC_CUSTOM_VOICE_WORKFLOW
    )

    # Test 2: Voice Design
    results["Voice Design"] = test_workflow(
        "Voice Design",
        VOICE_DESIGN_WORKFLOW
    )

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)

    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
