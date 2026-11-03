"""EdgeMind local LLM downloader and inference smoke test.

Model: OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov
Downloads to the models/qwen-1.5b folder next to this script.

Run:
    python local_llm.py
    python local_llm.py --download-only
    python local_llm.py --device CPU --prompt "Say hello"
"""

import argparse
import sys
from pathlib import Path

MODEL_ID = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "qwen-1.5b"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and test EdgeMind's local Qwen model")
    parser.add_argument("--download-only", action="store_true", help="Download model but skip inference")
    parser.add_argument("--device", choices=["AUTO", "CPU", "GPU", "NPU"], default="AUTO")
    parser.add_argument("--prompt", default="Explain what an AI agent is in two sentences.")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing package: huggingface_hub. Run: python -m pip install huggingface_hub")
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: {MODEL_ID}")
    print(f"Local directory: {MODEL_DIR}")
    print("Downloading or checking existing model files...")
    try:
        model_path = snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    except Exception as exc:
        print(f"Model download failed: {exc}")
        return 1

    print("Model files are ready.")
    if args.download_only:
        return 0

    try:
        import openvino as ov
        import openvino_genai as genai
    except ImportError:
        print("Missing OpenVINO packages. Run: python -m pip install openvino openvino-genai")
        return 1

    devices = ov.Core().available_devices
    print("Available devices:", ", ".join(devices))
    device = ("GPU" if "GPU" in devices else "CPU") if args.device == "AUTO" else args.device
    if device not in devices:
        print(f"Requested device '{device}' not detected. Choose one of: {', '.join(devices)}")
        return 1

    print(f"Running on: {device}")
    try:
        pipe = genai.LLMPipeline(str(model_path), device)
        result = pipe.generate(args.prompt, max_new_tokens=120)
        print("\nModel response:\n", result)
    except Exception as exc:
        print(f"Inference failed on {device}: {exc}")
        return 1
    print("\nEdgeMind local inference test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
