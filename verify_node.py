#!/usr/bin/env python3
"""
Quick verification script for Qwen3-TTS nodes.
Run this to check if nodes are properly registered in ComfyUI.
"""

import requests
import sys

COMFYUI_URL = "http://127.0.0.1:8188"

def main():
    print("=" * 60)
    print("Qwen3-TTS Node Verification")
    print("=" * 60)

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

    # Check node registration
    try:
        response = requests.get(f"{COMFYUI_URL}/object_info", timeout=30)
        if response.status_code != 200:
            print("ERROR: Failed to get node info")
            return 1

        object_info = response.json()

        required_nodes = [
            "Qwen3TTS_ModelLoader",
            "Qwen3TTS_CustomVoice",
            "Qwen3TTS_VoiceDesign",
            "Qwen3TTS_VoiceClone"
        ]

        print()
        print("Node Status:")
        print("-" * 40)

        all_found = True
        for node in required_nodes:
            if node in object_info:
                print(f"  [OK] {node}")
            else:
                print(f"  [MISSING] {node}")
                all_found = False

        print("-" * 40)

        if all_found:
            print("\nAll Qwen3-TTS nodes are registered!")
            print("You can now use the workflows.")
            return 0
        else:
            print("\nSome nodes are missing!")
            print("\nPlease restart ComfyUI:")
            print("  1. Close ComfyUI completely")
            print("  2. Start ComfyUI again")
            print("  3. Run this script again to verify")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
