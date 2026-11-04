"""Local test runner: python run_all_tests.py --all (no installs/downloads)."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def group(test):
    name = test.id()
    if 'test_live_hardware' in name or 'test_live_llama' in name:
        return 'hardware'
    if 'test_stage11_llama' in name:
        return 'integration'
    if 'test_live_retrieval' in name:
        return 'retrieval'
    if 'test_browser_ui' in name:
        return 'browser'
    if 'test_cli_smoke' in name:
        return 'smoke'
    if 'web_api' in name or 'ApiRegressions' in name:
        return 'api'
    if any(word in name for word in ('workflow', 'handoff', 'web_api', 'ApiRegressions', 'WorkflowRegressions')):
        return 'integration'
    if 'test_audit_regressions' in name:
        return 'regression'
    if 'stage9_benchmarks' in name:
        return 'benchmark'
    return 'unit'


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def record(self, test, status, detail=None):
        item = {'id': test.id(), 'group': group(test), 'status': status,
                'seconds': round(time.perf_counter() - getattr(self, 'started', time.perf_counter()), 3)}
        if detail:
            item['detail'] = detail
        if hasattr(test, 'observed_metrics'):
            item['observed_metrics'] = test.observed_metrics
        if hasattr(test, 'evidence'):
            item['evidence'] = test.evidence
        self.records.append(item)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, 'PASS')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, 'FAIL', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, 'ERROR', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, 'BLOCKED' if reason.startswith('BLOCKED:') else 'SKIP', reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            self.record(subtest, 'FAIL', self._exc_info_to_string(err, test))


def write_reports(result, selected):
    counts = {status: sum(row['status'] == status for row in result.records)
              for status in ('PASS', 'FAIL', 'ERROR', 'SKIP', 'BLOCKED')}
    required = {'unit', 'regression', 'integration', 'api', 'browser', 'hardware', 'retrieval', 'benchmark', 'smoke'}
    verified = (required.issubset(selected) and result.wasSuccessful()
                and not counts['SKIP'] and not counts['BLOCKED'])
    limitations = ['Real cloud inference is NOT IMPLEMENTED; cloud routing is simulated.',
                   'Verifier checks structure, not factual correctness.',
                   'Passing tests do not establish public deployment readiness.']
    if not verified:
        limitations.append('Stage 11 verification incomplete: inspect nonpassing outcomes or run --all.')
    summary = {
        'generated_at': datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        'selected_groups': sorted(selected), 'tests_executed': result.testsRun,
        'passed': counts['PASS'], 'failed': counts['FAIL'], 'errors': counts['ERROR'],
        'skipped': counts['SKIP'], 'blocked': counts['BLOCKED'],
        'stage10_fully_verified': verified, 'stage11_ready': verified,
        'stage11_llama_verified': verified,
        'cloud_inference': 'NOT IMPLEMENTED', 'outcomes': result.records,
        'limitations': limitations}
    summary['dependency_versions'] = {}
    for package in ('playwright', 'fastembed', 'onnxruntime', 'openvino', 'openvino-genai', 'fastapi', 'langgraph'):
        try:
            summary['dependency_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            summary['dependency_versions'][package] = 'not installed'
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    baseline_file = reports / 'STAGE10_BASELINE.json'
    if baseline_file.is_file():
        baseline_ids = {row['id'] for row in json.loads(baseline_file.read_text(encoding='utf-8'))['outcomes']}
        current = {row['id']: row['status'] for row in result.records}
        summary['stage10_baseline_regression'] = {
            'total': len(baseline_ids), 'passed': sum(current.get(key) == 'PASS' for key in baseline_ids),
            'missing_or_nonpassing': sorted(key for key in baseline_ids if current.get(key) != 'PASS')}
    performance = [{'id': row['id'], 'metrics': row['observed_metrics']}
                   for row in result.records if row.get('observed_metrics')]
    (reports / 'MODEL_PERFORMANCE.json').write_text(json.dumps(performance, indent=2), encoding='utf-8')
    (reports / 'TEST_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    features = [
        ('Planner', ('planner', 'execution', 'workflow')),
        ('Router', ('router', 'hardware', 'optimization', 'privacy_matrix')),
        ('RAG algorithms (injected semantic adapters)', ('rag', 'hybrid', 'numpy', 'RetrievalRegressions')),
        ('Coordinator', ('execution', 'local_adapter', 'workflow')),
        ('Verifier', ('safety', 'workflow', 'local_adapter')),
        ('Recovery', ('safety', 'workflow', 'timeout')),
        ('FastAPI', ('web_api', 'ApiRegressions')),
        ('Web UI browser interactions', ('test_browser_ui',)),
        ('Local Qwen CPU/GPU', ('test_live_hardware',)),
        ('Llama CPU/GPU, code and RAG', ('test_live_llama', 'test_stage11_llama')),
        ('Real FastEmbed semantic retrieval and cross-encoder', ('test_live_retrieval',)),
        ('Adaptive routing and benchmark validation', ('benchmarks',)),
    ]
    lines = ['# EdgeMind test report', '',
             f"Generated: {summary['generated_at']}", '', '## Executive summary', '',
             f"Executed {result.testsRun} test cases: {counts['PASS']} passed, {counts['FAIL']} failed, "
             f"{counts['ERROR']} errors, {counts['SKIP']} skipped, {counts['BLOCKED']} blocked.", '',
             '193 existing tests passed before changes. New regressions reproduced four failures '
             '(one diagnostic leak and three invalid-setting cases) before fixes.', '',
             ('Stage 11 Llama local integration verified alongside Stage 10 regressions.'
              if verified else 'Stage 11 verification incomplete; see selected groups and nonpassing outcomes.'),
             'This local demonstration is not approved for public deployment.', '',
             '## Feature validation matrix', '', '| Feature | Status | Evidence | Notes |',
             '|---|---|---|---|']
    for feature, tokens in features:
        rows = [row for row in result.records if any(token in row['id'] for token in tokens)]
        status = ('NOT TESTED' if not rows else 'FAIL' if any(r['status'] in ('FAIL', 'ERROR') for r in rows)
                  else 'BLOCKED' if any(r['status'] == 'BLOCKED' for r in rows)
                  else 'NOT TESTED' if any(r['status'] == 'SKIP' for r in rows) else 'PASS')
        passed = sum(row['status'] == 'PASS' for row in rows)
        lines.append(f'| {feature} | {status} | {passed}/{len(rows)} matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |')
    lines += ['| Real cloud inference | NOT IMPLEMENTED | No authenticated adapter in coordinator | Cloud routing is simulated |', '',
              '## Audit and fixes', '',
              '- Feature inventory and architecture: [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md).',
              '- Fixed returned OpenVINO exception text leaking through API error, verifier explanation, and trace. '
              'The API now replaces failed real-adapter diagnostics with safe text.',
              '- Fixed hybrid retrieval accepting invalid top-k, candidate-k and similarity with empty documents. '
              'Settings are validated before the empty-corpus branch.',
              '- Failed real requests now state that no verified result was produced.',
              '- Added concurrent API state isolation, validation, full HTTP document/code smoke, persistent timeout '
              'exhaustion and privacy matrix regressions.',
              '- Added a runner with quick, integration, browser, hardware and all groups; JSON records include '
              'individual outcomes and observed hardware metrics when successful.', '',
              '- Follow-up: installed Playwright/Chromium after user approval; browser tests start an actual '
              'loopback Uvicorn process and submit mock, retrieval and real CPU/GPU Qwen tasks. '
              'Browser requests outside the local server are blocked and treated as failures.',
              '- Follow-up: actual BAAI embedding cache loaded successfully offline. The configured Xenova '
              'cross-encoder was missing; its weights were downloaded with explicit user approval. '
              'Subsequent tests require local_files_only and offline mode. Existing Qwen weights were preserved.',
              '- Added EDGEMIND_RETRIEVAL_LOCAL_ONLY=1 for explicit no-download retrieval model loading.', '',
              '- Stage 11: Llama 3.2 1B INT4 added to the fixed allowlist, shared OpenVINO adapter, '
              'service, coordinator and verifier. Legacy local_qwen mode and model paths are preserved.',
              '- Llama is preselected in the local-model UI. Real mode remains opt-in. CPU/GPU and '
              'model-specific Auto routing are available. Qwen observations cannot satisfy Llama identity checks.',
              '- Llama download explicitly approved; existing Qwen weights were preserved. Official license, '
              'acceptable-use policy and NOTICE accompany Built with Llama attribution.',
              '- Original 223-test Stage 10 baseline preserved in STAGE10_BASELINE.json; '
              'per-ID regression comparison is in TEST_SUMMARY.json. Model timing/RSS evidence is '
              'also exported to MODEL_PERFORMANCE.json.', '',
              '- Initial Stage 11 full run blocked three browser cases under low RAM after hybrid model loading. '
              'The browser fixture now clears its parent preflight model caches and restarts only its own '
              'loopback server before real-model scenarios. Application memory safeguards are unchanged.', '',
              '## Environment and exact errors', '',
              '- Sandbox launch of the existing venv failed: `The file cannot be accessed by the system.` '
              'Running the same existing environment with approved escalation succeeded; venv was preserved.',
              '- Default Python 3.14 failed workflow imports: `ModuleNotFoundError: No module named langgraph`. '
              'Use the existing project venv (Python 3.13). Only the approved browser tooling was installed.', '',
              '- Initial runner report generation found missing Windows timezone data (`No time zone found with key Asia/Kolkata`). '
              'Fixed by using the standard-library fixed IST offset; no tzdata install required.', '',
              '## Failures, skipped and blocked tests', '']
    for row in result.records:
        if row['status'] != 'PASS':
            lines += [f"### {row['id']} — {row['status']}", '', '```text', row.get('detail', ''), '```', '']
    lines += ['## Remaining risks, manual attention and deployment blockers', '']
    lines += ['- ' + item for item in summary['limitations']]
    lines += ['- NPU generation is unsupported; NPU eligibility/classification is simulated.',
              '- No authentication, durable checkpoints, hard inference cancellation, production isolation or persistent task history.',
              '- Network offline is a routing flag; hybrid retrieval can download weights on first use.',
              '- Benchmarks test cold adapter latency, not factual quality or GPU kernel timings. '
              'Existing Qwen benchmark JSON and weights were not changed; Llama measurements use a separate file.',
              '- Hardware child processes have a 180-second test timeout; a timeout is BLOCKED, never PASS.',
              '- Real inference errors are failures; unavailable hardware/models/RAM are BLOCKED.', '']
    lines += ['## Observed dedicated hardware measurements', '',
              '| Model | Test | Load ms | Generate ms | Adapter total ms | Workflow ms |',
              '|---|---|---|---|---|---|']
    for row in result.records:
        metrics = row.get('observed_metrics')
        if metrics:
            lines.append(f"| {metrics.get('model_id', 'legacy Qwen')} | {row['id'].rsplit('.', 1)[-1]} | {metrics['model_load_ms']} | "
                         f"{metrics['generation_ms']} | {metrics['adapter_total_ms']} | {metrics['workflow_total_ms']} |")
    lines += ['', 'These are short, 12–32-token cold requests, not statistically representative benchmarks.', '']
    if summary.get('stage10_baseline_regression'):
        baseline = summary['stage10_baseline_regression']
        lines += ['## Stage 10 baseline preservation', '',
                  f"{baseline['passed']}/{baseline['total']} original test IDs passed in this run.", '']
    lines += ['## Browser and cached-model evidence', '',
              'Each test outcome in TEST_SUMMARY.json contains numeric/model evidence when available. '
              'Browser tests connect over real loopback HTTP; model output is not mocked. '
              'Real inference screenshots are in reports/browser/.', '']
    for row in result.records:
        if row.get('evidence'):
            lines += [f"### {row['id']}", '', '```json', json.dumps(row['evidence'], indent=2), '```', '']
    (reports / 'EDGEMIND_TEST_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('quick', 'integration', 'browser', 'hardware', 'retrieval', 'all'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    selected = set()
    if args.quick:
        selected.update(('unit', 'regression', 'benchmark'))
    if args.integration:
        selected.update(('integration', 'api', 'smoke'))
    if args.browser:
        selected.add('browser')
    if args.hardware:
        selected.add('hardware')
    if args.retrieval:
        selected.add('retrieval')
    if args.all:
        selected.update(('unit', 'regression', 'integration', 'api', 'browser', 'hardware', 'retrieval', 'benchmark', 'smoke'))
    if not selected:
        selected.update(('unit', 'regression', 'integration', 'api', 'benchmark', 'smoke'))
    if 'hardware' in selected:
        os.environ['EDGEMIND_TEST_HARDWARE'] = '1'
    if 'browser' in selected:
        os.environ['EDGEMIND_TEST_BROWSER'] = '1'
    if 'retrieval' in selected or 'browser' in selected:
        os.environ['EDGEMIND_TEST_RETRIEVAL'] = '1'
    tests = list(flatten(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))))
    order = {'unit': 0, 'regression': 1, 'integration': 2, 'api': 3,
             'browser': 4, 'hardware': 5, 'retrieval': 6, 'benchmark': 7, 'smoke': 8}
    suite = unittest.TestSuite(sorted((t for t in tests if group(t) in selected), key=lambda t: order[group(t)]))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    write_reports(result, selected)
    print('Reports: reports/EDGEMIND_TEST_REPORT.md and reports/TEST_SUMMARY.json')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
