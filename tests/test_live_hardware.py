"""Dedicated opt-in, bounded real inference tests. Never download model files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from edgemind.local_models import for_mode, QWEN

ROOT = Path(__file__).resolve().parents[1]


class LocalInferenceTests(unittest.TestCase):
    def exercise(self, device, task_type, *, mode=QWEN.mode, documents=None, max_new_tokens=12):
        if os.environ.get('EDGEMIND_TEST_HARDWARE') != '1':
            self.skipTest('BLOCKED: enable dedicated hardware tests with --hardware')
        from edgemind.adapters.openvino_qwen import runtime_hardware_profile, LocalModelError
        try:
            hardware = runtime_hardware_profile()
        except LocalModelError as exc:
            self.skipTest('BLOCKED: ' + str(exc))
        specification = for_mode(mode)
        if device != 'auto' and not hardware['devices'].get(device):
            self.skipTest(f'BLOCKED: {device} unavailable')
        if hardware['available_memory_mb'] < specification.required_memory_mb:
            self.skipTest('BLOCKED: insufficient available RAM for selected model')
        variable = 'EDGEMIND_MODEL_PATH' if mode == QWEN.mode else 'EDGEMIND_LLAMA_MODEL_PATH'
        model = Path(os.environ.get(variable, ROOT / specification.directory))
        if not all((model / name).is_file() for name in
                   ('openvino_model.xml', 'openvino_model.bin', 'openvino_tokenizer.xml')):
            self.skipTest('BLOCKED: existing OpenVINO model files unavailable')
        script = '''
import json, os, sys
from fastapi.testclient import TestClient
from edgemind.api import app
os.environ['EDGEMIND_ENABLE_REAL_LOCAL'] = '1'
variable = 'EDGEMIND_MODEL_PATH' if sys.argv[4] == 'local_qwen' else 'EDGEMIND_LLAMA_MODEL_PATH'
os.environ[variable] = sys.argv[1]
with TestClient(app) as client:
    response = client.post('/api/tasks', json={
        'query': 'Summarize the supplied GPU document.' if json.loads(sys.argv[5]) else
                 'Return Python code: x = 1' if sys.argv[3] == 'code' else 'Say hello briefly.',
        'task_type': sys.argv[3], 'execution_mode': sys.argv[4], 'documents': json.loads(sys.argv[5]),
        'local_device': sys.argv[2], 'max_new_tokens': int(sys.argv[6])})
    data = response.json()
    print('EDGEMIND_RESULT=' + json.dumps({
        'http_status': response.status_code, 'status': data.get('status'),
        'real': data.get('is_real_inference'), 'has_content': bool(data.get('result', '').strip()),
        'device': data.get('model', {}).get('device'), 'performance': data.get('performance'),
        'model_id': data.get('model', {}).get('model_id'),
        'evidence_sources': [item['source_id'] for item in data.get('evidence', [])],
        'error': data.get('error')}))
'''
        try:
            process = subprocess.run([sys.executable, '-c', script, str(model), device, task_type,
                                      mode, json.dumps(documents or []), str(max_new_tokens)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            self.skipTest('BLOCKED: local model load/inference exceeded 180 seconds; child terminated')
        self.assertEqual(process.returncode, 0, 'Inference child failed; exit=' + str(process.returncode))
        lines = [line for line in process.stdout.splitlines() if line.startswith('EDGEMIND_RESULT=')]
        self.assertEqual(len(lines), 1, 'Inference child did not return a result')
        result = json.loads(lines[0].split('=', 1)[1])
        self.assertEqual(result['http_status'], 200)
        self.assertEqual(result['status'], 'COMPLETED', result.get('error'))
        self.assertTrue(result['real'])
        self.assertTrue(result['has_content'])
        self.assertEqual(result['model_id'], specification.model_id)
        if device == 'auto':
            self.assertIn(result['device'], ('CPU', 'GPU'))
        else:
            self.assertEqual(result['device'], device)
        if documents:
            self.assertTrue(result['evidence_sources'])
        self.assertGreater(result['performance']['adapter_total_ms'], 0)
        self.assertGreaterEqual(result['performance']['workflow_total_ms'],
                                result['performance']['adapter_total_ms'])
        # Safe numeric evidence only; no prompt, response or machine path.
        self.observed_metrics = result['performance']
        self.evidence = {'model_id': specification.model_id, 'device': result['device'],
                         'task_type': task_type, 'max_new_tokens': max_new_tokens,
                         'evidence_sources': result['evidence_sources']}

    def test_cpu_text(self):
        self.exercise('CPU', 'text')

    def test_cpu_code(self):
        self.exercise('CPU', 'code')

    def test_gpu_text(self):
        self.exercise('GPU', 'text')

    def test_gpu_code(self):
        self.exercise('GPU', 'code')
