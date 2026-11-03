"""Terminal-only Layer 1 adapter. Does not leak CLI concerns into Layer 2."""
import argparse
import json
from .service import run_task
from .contracts import mock_hardware_profile, mock_model_registry

SAMPLE_DOCUMENTS = [
    {'source_id': 'sample/privacy', 'text': 'Private documents must remain on the local device.'},
    {'source_id': 'sample/compute', 'text': 'GPU accelerates supported language models; NPU can run small neural networks.'},
    {'source_id': 'sample/offline', 'text': 'Offline requests cannot call a cloud model.'},
]


def main():
    parser = argparse.ArgumentParser(description='EdgeMind Layer 2 terminal demonstration')
    parser.add_argument('--query', help='Task text; prompted if omitted')
    parser.add_argument('--privacy', choices=['local_only', 'cloud_allowed'], default='local_only')
    parser.add_argument('--network', choices=['online', 'offline'], default='online')
    parser.add_argument('--complexity', choices=['simple', 'complex'], default='simple')
    parser.add_argument('--task-type', choices=['text', 'code', 'classification'],
                        default='text', help='Task category for capability routing')
    parser.add_argument('--demo-docs', action='store_true', help='Use three fake sample documents')
    parser.add_argument('--rag-mode', choices=['bm25', 'hybrid'], default='bm25',
                        help='BM25 baseline or local ONNX semantic+BM25 hybrid')
    parser.add_argument('--rerank', action='store_true',
                        help='Run local cross-encoder reranker after hybrid retrieval')
    parser.add_argument('--rag-candidate-k', type=int, default=10,
                        help='Hybrid candidates to consider before optional reranking')
    parser.add_argument('--rag-top-k', type=int, default=3, help='Maximum retrieved passages')
    parser.add_argument('--rag-chunk-words', type=int, default=120, help='Words per document chunk')
    parser.add_argument('--rag-overlap-words', type=int, default=20, help='Overlapping words between chunks')
    parser.add_argument('--simulate-failure', action='store_true', help='Trigger one recoverable mock timeout')
    parser.add_argument('--simulate-permanent-failure', action='store_true', help='Trigger non-retryable mock failure')
    parser.add_argument('--simulate-invalid-result', action='store_true', help='Trigger malformed mock response')
    parser.add_argument('--ram-mb', type=int, default=4096,
                        help='SIMULATED available RAM in MB (not total installed RAM)')
    parser.add_argument('--gpu-off', action='store_true', help='SIMULATE unavailable GPU')
    parser.add_argument('--npu-off', action='store_true', help='SIMULATE unavailable NPU')
    parser.add_argument('--cpu-off', action='store_true', help='SIMULATE unavailable CPU')
    parser.add_argument('--strategy', choices=['balanced', 'fastest', 'cheapest', 'highest_quality'],
                        default='balanced', help='Simulation-only optimization policy')
    parser.add_argument('--max-latency-ms', type=int, help='Hard estimated latency limit in ms')
    parser.add_argument('--max-cost-usd', type=float, help='Hard estimated request cost budget in USD')
    parser.add_argument('--min-quality', type=float, help='Minimum estimated task quality from 0 to 1')
    parser.add_argument('--gpu-latency-ms', type=int, help='Override MOCK GPU latency for experiments')
    parser.add_argument('--real-local', action='store_true',
                        help='Opt in to REAL LOCAL OpenVINO Qwen inference (no cloud calls)')
    parser.add_argument('--model-path', default='models/qwen-1.5b',
                        help='Existing local OpenVINO Qwen model directory')
    parser.add_argument('--local-device', choices=['GPU', 'CPU', 'auto'], default='GPU',
                        help='Actual OpenVINO device, or auto from matching local benchmarks')
    parser.add_argument('--benchmarks', default='benchmarks_local.json',
                        help='Local benchmark file for --local-device auto')
    parser.add_argument('--benchmark-max-age-hours', type=int, default=168,
                        help='Reject older benchmark readings (default 7 days)')
    parser.add_argument('--max-new-tokens', type=int, default=160,
                        help='Maximum generation tokens for real local model (1..512)')
    parser.add_argument('--profile', action='store_true',
                        help='Measure observed real-local loading, generation, and process RAM')
    parser.add_argument('--metrics-out',
                        help='Optionally write safe numeric performance fields to a JSON file (no prompt/answer)')
    parser.add_argument('--json', action='store_true', help='Print complete state in JSON')
    args = parser.parse_args()
    query = args.query if args.query is not None else input('EdgeMind > ').strip()
    if not 1 <= args.max_new_tokens <= 512:
        parser.error('--max-new-tokens must be between 1 and 512')
    if args.benchmark_max_age_hours < 1 or args.benchmark_max_age_hours > 8760:
        parser.error('--benchmark-max-age-hours must be 1..8760')
    if args.local_device == 'auto' and not args.real_local:
        parser.error('--local-device auto requires --real-local')
    if args.profile and not args.real_local:
        parser.error('--profile requires --real-local')
    if args.metrics_out and not args.profile:
        parser.error('--metrics-out requires --profile')
    if args.real_local:
        if args.simulate_failure or args.simulate_permanent_failure or args.simulate_invalid_result:
            parser.error('--real-local cannot be combined with simulated failure flags')
        if args.gpu_off or args.npu_off or args.cpu_off or args.gpu_latency_ms is not None or args.ram_mb != 4096:
            parser.error('--real-local does not accept simulated hardware controls')
        if args.task_type == 'classification':
            parser.error('Local Qwen adapter supports only text and code tasks')
        if args.min_quality is not None or args.max_latency_ms is not None:
            parser.error('Quality/latency estimates are not measured; remove these constraints in --real-local mode')
    if args.rerank and args.rag_mode != 'hybrid':
        parser.error('--rerank requires --rag-mode hybrid')
    if args.rag_candidate_k < args.rag_top_k:
        parser.error('--rag-candidate-k must be >= --rag-top-k')
    if args.rag_top_k < 1 or not 1 <= args.rag_chunk_words <= 2000 or not 0 <= args.rag_overlap_words < args.rag_chunk_words:
        parser.error('Invalid RAG settings: top-k positive, chunk-words 1..2000, overlap lower than chunk size')
    if args.ram_mb < 0:
        parser.error('--ram-mb must be nonnegative')
    if args.gpu_latency_ms is not None and args.gpu_latency_ms <= 0:
        parser.error('--gpu-latency-ms must be positive')
    if args.max_latency_ms is not None and args.max_latency_ms <= 0:
        parser.error('--max-latency-ms must be positive')
    if args.max_cost_usd is not None and args.max_cost_usd < 0:
        parser.error('--max-cost-usd must be nonnegative')
    if args.min_quality is not None and not 0 <= args.min_quality <= 1:
        parser.error('--min-quality must be between 0 and 1')
    registry = mock_model_registry()
    if args.gpu_latency_ms is not None:
        for model in registry:
            if model['provider'] == 'local' and model['device'] == 'GPU':
                model['estimated_latency_ms'] = args.gpu_latency_ms
    preferences = {'strategy': args.strategy}
    if args.max_latency_ms is not None:
        preferences['max_latency_ms'] = args.max_latency_ms
    if args.max_cost_usd is not None:
        preferences['max_cost_usd'] = args.max_cost_usd
    if args.min_quality is not None:
        preferences['min_quality_score'] = args.min_quality
    hardware = mock_hardware_profile()
    hardware['available_memory_mb'] = args.ram_mb
    for flag, device in ((args.gpu_off, 'GPU'), (args.npu_off, 'NPU'),
                         (args.cpu_off, 'CPU')):
        if flag:
            hardware['devices'][device] = False
    result = run_task(
        query=query,
        execution_mode='local_qwen' if args.real_local else 'mock',
        local_model_path=args.model_path if args.real_local else None,
        local_device=args.local_device,
        max_new_tokens=args.max_new_tokens,
        profile_local=args.profile,
        benchmark_file=args.benchmarks if args.real_local and args.local_device == 'auto' else None,
        benchmark_max_age_hours=args.benchmark_max_age_hours,
        privacy_mode=args.privacy,
        network=args.network,
        complexity=args.complexity,
        task_type=args.task_type,
        documents=SAMPLE_DOCUMENTS if args.demo_docs else [],
        simulate_failure=args.simulate_failure,
        simulate_permanent_failure=args.simulate_permanent_failure,
        simulate_invalid_result=args.simulate_invalid_result,
        hardware_profile=None if args.real_local else hardware,
        available_models=None if args.real_local else registry,
        routing_preferences=preferences,
        rag_settings={'top_k': args.rag_top_k, 'chunk_words': args.rag_chunk_words,
                      'overlap_words': args.rag_overlap_words, 'mode': args.rag_mode,
                      'enable_reranker': args.rerank, 'candidate_k': args.rag_candidate_k},
    )
    if args.metrics_out:
        from pathlib import Path
        report = {
            'status': result.get('status'),
            'model_id': result.get('decision', {}).get('model_id'),
            'device': result.get('decision', {}).get('device'),
            'task_type': args.task_type,
            'max_new_tokens': args.max_new_tokens,
            'performance': result.get('performance', {}),
        }
        Path(args.metrics_out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print('\n===== EDGEMIND / ' + ('REAL LOCAL QWEN' if args.real_local else 'MOCK DEMO') + ' =====')
    for entry in result['trace']:
        print('  ' + entry)
    print('\nStatus:', result['status'])
    print('Hardware:', result['hardware_profile'] if args.real_local else hardware)
    print('Routing strategy:', args.strategy)
    if args.real_local and args.local_device == 'auto':
        print('Benchmark validation:', result.get('benchmark_diagnostics'))
        print('Selected observed cold-start benchmark (ms):', result['decision'].get('observed_median_adapter_total_ms'))
        print('Routing source:', result['decision'].get('latency_source'))
    else:
        print('Router estimates (UNMEASURED):', result['decision'].get('estimates'))
    if args.profile:
        print('Observed performance (NOT routing estimates):', json.dumps(result.get('performance', {}), indent=2))
        if args.metrics_out:
            print('Numeric-only metrics saved to:', args.metrics_out)
    print('RAG diagnostics:', result.get('rag_diagnostics'))
    print('Planner steps:', result.get('plan_steps'))
    print('Planner diagnostics:', result.get('plan_diagnostics'))
    print('Actual planned execution:', result.get('executed_steps'))
    print('Per-step results:', result.get('step_results'))
    print('Retrieved passage sources:', [item['source_id'] for item in result.get('evidence', [])])
    print('Decision:', result['decision'])
    print('Verification:', result.get('verification'))
    print('Verification history:', result.get('verification_history'))
    print('Recovery reason:', result.get('recovery_reason', 'none'))
    print('Result:', result['tool_result']['content'] or '(none)')
    if args.real_local:
        print('REAL LOCAL inference attempted. No cloud API call or generated-code execution.')
        print('Only auto-mode latency was calibrated from local cold-start observations; quality unverified.')
    else:
        print('WARNING: All GENERATIVE responses and routing capabilities are simulated.')
        print('Hybrid RAG, when enabled, uses real local embedding/reranking models.')
        print('No GPU/NPU generation and no cloud API call occurred in mock mode.')


if __name__ == '__main__':
    main()
