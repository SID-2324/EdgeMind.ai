"""OPTIONAL: explicitly download a local OpenVINO Qwen model. No download on normal runs."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/qwen-1.5b')
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit('Install huggingface_hub: python -m pip install huggingface_hub') from exc
    target = Path(args.model_dir).expanduser().resolve()
    print('Downloading OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov to', target)
    snapshot_download(repo_id='OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov', local_dir=str(target))
    print('Ready:', target)


if __name__ == '__main__':
    main()
