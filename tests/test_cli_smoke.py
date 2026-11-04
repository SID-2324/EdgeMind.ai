"""Final end-to-end command-line smoke tests, using only mock generation."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, *args):
        process = subprocess.run([sys.executable, 'main.py', *args],
                                 cwd=Path(__file__).resolve().parents[1],
                                 capture_output=True, text=True, encoding='utf-8',
                                 timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_mock_document_code_end_to_end(self):
        result = self.run_cli('--query', 'Write Python explaining GPU', '--task-type', 'code',
                              '--demo-docs', '--json')
        self.assertEqual(result['status'], 'SIMULATED')
        self.assertEqual([s['operation'] for s in result['plan_steps']],
                         ['retrieve_evidence', 'generate_code', 'verify_output'])
        self.assertTrue(result['evidence'])

    def test_mock_classification_end_to_end(self):
        result = self.run_cli('--query', 'Classify this message', '--task-type', 'classification', '--json')
        self.assertEqual(result['status'], 'SIMULATED')
        self.assertEqual(result['decision']['device'], 'NPU')

    def test_private_no_hardware_end_to_end(self):
        result = self.run_cli('--query', 'Private request', '--cpu-off', '--gpu-off', '--npu-off', '--json')
        self.assertEqual(result['status'], 'DEFERRED')
        self.assertIsNone(result['decision']['provider'])
