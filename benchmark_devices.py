"""Standalone, local CPU/GPU OpenVINO Qwen benchmark (does not use cloud)."""
import argparse
from edgemind.benchmark.benchmarks import run_benchmarks, save_report
from edgemind.local_models import LLAMA, QWEN


def main():
    parser = argparse.ArgumentParser(description='Benchmark actual local Qwen cold-start latency')
    parser.add_argument('--model-path', required=True, help='Existing Qwen OpenVINO model directory')
    parser.add_argument('--model', choices=['llama', 'qwen'], default='qwen')
    parser.add_argument('--devices', nargs='+', default=['GPU', 'CPU'], choices=['GPU', 'CPU'])
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output')
    args = parser.parse_args()
    model = LLAMA if args.model == 'llama' else QWEN
    output = args.output or model.benchmark_file
    report = run_benchmarks(model_path=args.model_path, model_id=model.model_id, devices=args.devices,
                            runs=args.runs, warmups=args.warmups,
                            max_new_tokens=args.max_new_tokens)
    save_report(report, output)
    print('Benchmark report saved to:', output)
    for device, measurements in report['devices'].items():
        print(device, 'status:', measurements['status'],
              'median cold adapter time (ms):', measurements.get('median_adapter_total_ms'))
    if not any(item['status'] == 'ok' for item in report['devices'].values()):
        print('No eligible measurements recorded. Check GPU/CPU availability, RAM and model folder.')


if __name__ == '__main__':
    main()
