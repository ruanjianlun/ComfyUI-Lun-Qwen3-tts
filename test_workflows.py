#!/usr/bin/env python3
"""
Automated test script for Qwen3-TTS ComfyUI workflows.
Tests all three workflow types via ComfyUI REST API.
"""

import json
import time
import requests
import uuid
import sys
import os

# ComfyUI API settings
COMFYUI_URL = "http://127.0.0.1:8188"
TIMEOUT = 600  # 10 minutes timeout for model loading and generation

def queue_prompt(prompt_workflow):
    """Queue a workflow to ComfyUI and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {
        "prompt": prompt_workflow,
        "client_id": client_id
    }

    response = requests.post(
        f"{COMFYUI_URL}/prompt",
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"Failed to queue prompt: {response.text}")

    return response.json().get("prompt_id"), client_id


def wait_for_completion(prompt_id, timeout=TIMEOUT):
    """Wait for workflow execution to complete."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{COMFYUI_URL}/history/{prompt_id}",
                timeout=30
            )

            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    prompt_data = history[prompt_id]

                    # Check if execution is complete
                    if "outputs" in prompt_data:
                        return True, prompt_data

                    # Check for errors
                    if prompt_data.get("status", {}).get("status_str") == "error":
                        return False, prompt_data

            time.sleep(2)

        except requests.RequestException as e:
            print(f"  Warning: Request error: {e}")
            time.sleep(2)

    return False, {"error": "Timeout"}


def load_workflow(workflow_path):
    """Load workflow JSON file."""
    with open(workflow_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_basic_custom_voice():
    """Test the basic custom voice workflow."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Custom Voice Workflow")
    print("=" * 60)

    workflow_path = os.path.join(
        os.path.dirname(__file__),
        "example_workflows",
        "basic_custom_voice.json"
    )

    if not os.path.exists(workflow_path):
        print(f"  ERROR: Workflow file not found: {workflow_path}")
        return False

    try:
        workflow = load_workflow(workflow_path)
        print(f"  Loaded workflow from: {workflow_path}")
        print(f"  Model: CustomVoice-1.7B")
        print(f"  Speaker: Vivian")
        print(f"  Text: 你好,欢迎使用千问语音合成...")

        print("  Queueing workflow...")
        prompt_id, client_id = queue_prompt(workflow)
        print(f"  Prompt ID: {prompt_id}")

        print("  Waiting for execution (this may take a while for model loading)...")
        success, result = wait_for_completion(prompt_id)

        if success:
            print("  SUCCESS: Workflow completed!")
            outputs = result.get("outputs", {})
            for node_id, output in outputs.items():
                if "audio" in output:
                    audio_info = output["audio"]
                    print(f"  Generated audio: {audio_info}")
            return True
        else:
            print(f"  FAILED: {result}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_voice_design():
    """Test the voice design workflow."""
    print("\n" + "=" * 60)
    print("Test 2: Voice Design Workflow")
    print("=" * 60)

    workflow_path = os.path.join(
        os.path.dirname(__file__),
        "example_workflows",
        "voice_design.json"
    )

    if not os.path.exists(workflow_path):
        print(f"  ERROR: Workflow file not found: {workflow_path}")
        return False

    try:
        workflow = load_workflow(workflow_path)
        print(f"  Loaded workflow from: {workflow_path}")
        print(f"  Model: VoiceDesign-1.7B")
        print(f"  Voice Description: A warm and friendly female voice...")

        print("  Queueing workflow...")
        prompt_id, client_id = queue_prompt(workflow)
        print(f"  Prompt ID: {prompt_id}")

        print("  Waiting for execution...")
        success, result = wait_for_completion(prompt_id)

        if success:
            print("  SUCCESS: Workflow completed!")
            outputs = result.get("outputs", {})
            for node_id, output in outputs.items():
                if "audio" in output:
                    audio_info = output["audio"]
                    print(f"  Generated audio: {audio_info}")
            return True
        else:
            print(f"  FAILED: {result}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_voice_clone():
    """Test the voice clone workflow."""
    print("\n" + "=" * 60)
    print("Test 3: Voice Clone Workflow")
    print("=" * 60)

    workflow_path = os.path.join(
        os.path.dirname(__file__),
        "example_workflows",
        "voice_clone.json"
    )

    if not os.path.exists(workflow_path):
        print(f"  ERROR: Workflow file not found: {workflow_path}")
        return False

    # Check for reference audio
    # Note: Voice clone needs a reference audio file
    print("  NOTE: Voice clone workflow requires a reference audio file.")
    print("  This test will queue the workflow but may fail if no audio is provided.")

    try:
        workflow = load_workflow(workflow_path)
        print(f"  Loaded workflow from: {workflow_path}")
        print(f"  Model: Base-1.7B")

        print("  Queueing workflow...")
        prompt_id, client_id = queue_prompt(workflow)
        print(f"  Prompt ID: {prompt_id}")

        print("  Waiting for execution...")
        success, result = wait_for_completion(prompt_id)

        if success:
            print("  SUCCESS: Workflow completed!")
            outputs = result.get("outputs", {})
            for node_id, output in outputs.items():
                if "audio" in output:
                    audio_info = output["audio"]
                    print(f"  Generated audio: {audio_info}")
            return True
        else:
            print(f"  FAILED (may need reference audio): {result}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_comfyui_connection():
    """Check if ComfyUI is running and accessible."""
    print("Checking ComfyUI connection...")

    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=10)
        if response.status_code == 200:
            print(f"  ComfyUI is running at {COMFYUI_URL}")
            return True
        else:
            print(f"  ComfyUI returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"  Cannot connect to ComfyUI: {e}")
        return False


def check_node_registration():
    """Check if Qwen3-TTS nodes are registered in ComfyUI."""
    print("\nChecking node registration...")

    try:
        response = requests.get(f"{COMFYUI_URL}/object_info", timeout=30)
        if response.status_code == 200:
            object_info = response.json()

            required_nodes = [
                "Qwen3TTS_ModelLoader",
                "Qwen3TTS_CustomVoice",
                "Qwen3TTS_VoiceDesign",
                "Qwen3TTS_VoiceClone"
            ]

            all_found = True
            for node_name in required_nodes:
                if node_name in object_info:
                    print(f"  [OK] {node_name}")
                else:
                    print(f"  [MISSING] {node_name}")
                    all_found = False

            return all_found
        else:
            print(f"  Failed to get object_info: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"  Error checking nodes: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Qwen3-TTS ComfyUI Workflow Tests")
    print("=" * 60)

    # Check connection
    if not check_comfyui_connection():
        print("\nERROR: Cannot connect to ComfyUI. Make sure it's running.")
        sys.exit(1)

    # Check node registration
    nodes_ok = check_node_registration()
    if not nodes_ok:
        print("\nWARNING: Some nodes are not registered. Tests may fail.")

    # Run tests
    results = {
        "basic_custom_voice": False,
        "voice_design": False,
        "voice_clone": False
    }

    # Test 1: Basic Custom Voice
    results["basic_custom_voice"] = test_basic_custom_voice()

    # Test 2: Voice Design
    results["voice_design"] = test_voice_design()

    # Test 3: Voice Clone (requires reference audio)
    results["voice_clone"] = test_voice_clone()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
