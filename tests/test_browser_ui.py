"""Playwright against a real loopback Uvicorn process; no intercepted API calls."""
import json
import gc
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen, ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('EDGEMIND_TEST_BROWSER') != '1':
            raise unittest.SkipTest('NOT TESTED: browser group not selected')
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest('BLOCKED: Playwright not installed')
        cls.driver = sync_playwright().start()
        cls.addClassCleanup(cls.driver.stop)
        # An installed browser failing to launch is an integration error.
        cls.browser = cls.driver.chromium.launch()
        cls.addClassCleanup(cls.browser.close)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        cls.origin = f'http://127.0.0.1:{port}'
        (ROOT / 'reports/browser').mkdir(parents=True, exist_ok=True)
        cls.port = port
        cls.server_env = {**os.environ, 'EDGEMIND_ENABLE_REAL_LOCAL': '1',
               'EDGEMIND_RETRIEVAL_LOCAL_ONLY': '1', 'HF_HUB_OFFLINE': '1'}
        cls.server_log = tempfile.TemporaryFile(mode='w+b')
        cls.addClassCleanup(cls.server_log.close)
        cls.start_server()
        cls.addClassCleanup(cls.stop_server)

    @classmethod
    def start_server(cls):
        cls.server = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'edgemind.api:app', '--host', '127.0.0.1',
             '--port', str(cls.port), '--log-level', 'warning'], cwd=ROOT, env=cls.server_env,
            stdout=cls.server_log, stderr=subprocess.STDOUT)
        opener = build_opener(ProxyHandler({}))
        for _ in range(150):
            if cls.server.poll() is not None:
                raise RuntimeError('Loopback Uvicorn process exited before startup')
            try:
                with opener.open(cls.origin + '/api/health', timeout=1) as response:
                    cls.health = json.load(response)
                break
            except OSError:
                time.sleep(.2)
        else:
            raise RuntimeError('Loopback Uvicorn startup exceeded 30 seconds')

    @classmethod
    def stop_server(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=5)

    def setUp(self):
        if self._testMethodName in {
                'test_real_cpu_text', 'test_real_gpu_code', 'test_llama_gpu_generation',
                'test_llama_qwen_switching_and_rag_real_inference', 'test_llama_auto_uses_own_observations'}:
            # Isolate tests from ONNX/runtime caches retained by earlier scenarios.
            # Admission RAM thresholds remain unchanged in tests and application.
            type(self).stop_server()
            type(self).start_server()
        self.context = self.browser.new_context(permissions=['clipboard-read', 'clipboard-write'])
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.external_requests = []
        self.javascript_errors = []
        def guard(route):
            if route.request.url.startswith(self.origin + '/'):
                route.continue_()
            else:
                self.external_requests.append(route.request.url.split('?')[0])
                route.abort()
        self.page.route('**/*', guard)
        self.page.on('pageerror', lambda error: self.javascript_errors.append(str(error)))
        self.page.goto(self.origin)
        self.page.wait_for_function("document.getElementById('serverStatus').textContent === 'Local server connected'")

    def tearDown(self):
        self.assertEqual(self.external_requests, [], 'Browser attempted a nonlocal request')
        self.assertEqual(self.javascript_errors, [], 'Uncaught browser JavaScript error')

    def advanced(self):
        self.page.locator('details.extras').evaluate('(node) => node.open = true')

    def submit(self, expected='SIMULATED', timeout=180000):
        with self.page.expect_response(lambda r: r.url == self.origin + '/api/tasks', timeout=timeout) as pending:
            self.page.click('#runButton')
        response = pending.value
        data = response.json()
        self.assertEqual(response.status, 200, str(data))
        self.page.wait_for_function('!document.getElementById("runButton").disabled')
        self.assertEqual(self.page.inner_text('#resultStatus'), expected)
        self.assertEqual(data['status'], expected)
        self.assertTrue(data['result'].strip())
        self.evidence = {'transport': 'real loopback HTTP', 'browser': self.browser.version,
                         'status': data['status'], 'is_real_inference': data['is_real_inference'],
                         'device': data['model']['device'], 'performance': data['performance'],
                         'rag_diagnostics': data['rag_diagnostics']}
        return data

    def test_mock_text_plan_trace_copy_and_safe_rendering(self):
        self.page.fill('#query', '<img src=x onerror=alert(1)> Describe GPU inference')
        result = self.submit()
        self.assertIn('[MOCK OUTPUT', self.page.inner_text('#resultText'))
        self.assertEqual(self.page.locator('#resultText img').count(), 0)
        self.assertIn('generate text', self.page.inner_text('#tabPanel'))
        self.page.click('[data-tab=trace]')
        self.assertIn('VERIFIER', self.page.inner_text('#tabPanel'))
        self.page.click('[data-tab=details]')
        self.assertIn('performance', self.page.inner_text('#tabPanel'))
        self.page.click('#copyOutput')
        self.page.wait_for_function("document.getElementById('copyOutput').textContent === 'Copied'")
        self.assertEqual(self.page.evaluate('navigator.clipboard.readText()'), result['result'])

    def test_mock_code_and_classification_selection(self):
        for task, operation in [('code', 'generate_code'), ('classification', 'classify_input')]:
            with self.subTest(task=task):
                self.page.select_option('#taskType', task)
                result = self.submit()
                self.assertIn(operation, [step['operation'] for step in result['plan']])
                self.assertFalse(result['is_real_inference'])

    def test_privacy_policy_and_real_mode_controls(self):
        self.advanced()
        self.page.select_option('#complexity', 'complex')
        # Deferred requests have no content, so inspect the HTTP result separately.
        with self.page.expect_response('**/api/tasks') as pending:
            self.page.click('#runButton')
        self.assertEqual(pending.value.json()['status'], 'DEFERRED')
        self.assertIsNone(pending.value.json()['model']['provider'])
        self.page.wait_for_function('!document.getElementById("runButton").disabled')
        self.page.select_option('#privacyMode', 'cloud_allowed')
        result = self.submit()
        self.assertEqual(result['model']['provider'], 'cloud')
        self.assertFalse(result['is_real_inference'])
        self.page.select_option('#taskType', 'classification')
        self.page.select_option('#executionMode', 'local_qwen')
        self.assertEqual(self.page.input_value('#privacyMode'), 'local_only')
        self.assertEqual(self.page.input_value('#taskType'), 'text')
        self.assertTrue(self.page.locator('#privacyMode option[value=cloud_allowed]').is_disabled())
        self.assertTrue(self.page.locator('#taskType option[value=classification]').is_disabled())

    def test_invalid_input_and_api_error_display(self):
        self.page.fill('#query', '   ')
        self.page.click('#runButton')
        self.assertEqual(self.page.inner_text('#errorMessage'), 'Please enter a task.')
        self.page.fill('#query', 'Hello')
        self.advanced()
        self.page.check('#rerank')
        with self.page.expect_response('**/api/tasks') as pending:
            self.page.click('#runButton')
        self.assertEqual(pending.value.status, 422)
        self.page.wait_for_selector('#errorMessage:not(.hidden)')
        self.assertIn('Reranking requires hybrid', self.page.inner_text('#errorMessage'))
        self.page.uncheck('#rerank')
        self.submit()
        self.assertTrue(self.page.locator('#errorMessage').is_hidden())

    def test_bm25_document_retrieval(self):
        self.advanced()
        self.page.click('#demoDocs')
        self.page.fill('#query', 'Describe GPU inference')
        result = self.submit()
        self.assertTrue(result['evidence'])
        self.assertEqual(result['plan'][0]['operation'], 'retrieve_evidence')

    def test_actual_hybrid_and_reranking_in_browser(self):
        from tests.test_live_retrieval import CachedRetrievalTests
        from edgemind.retrieval import local_models
        probe = CachedRetrievalTests()
        probe.load(local_models.get_embedder)
        probe.load(local_models.get_reranker)
        local_models.get_embedder.cache_clear()
        local_models.get_reranker.cache_clear()
        gc.collect()
        self.advanced()
        self.page.fill('#query', 'Which city is the French capital?')
        self.page.fill('#documentText', 'Paris is the capital city of France. Bananas are yellow fruit.')
        self.page.select_option('#ragMode', 'hybrid')
        self.page.check('#rerank')
        result = self.submit()
        self.assertTrue(result['rag_diagnostics']['semantic_model_used'])
        self.assertTrue(result['rag_diagnostics']['reranked'])
        self.assertTrue(result['evidence'])

    def real_inference(self, device, task):
        from edgemind.adapters.openvino_qwen import runtime_hardware_profile
        hardware = runtime_hardware_profile()
        if not hardware['devices'].get(device) or hardware['available_memory_mb'] < 2300:
            self.skipTest(f'BLOCKED: insufficient RAM or {device} unavailable')
        self.advanced()
        self.page.select_option('#executionMode', 'local_qwen')
        self.page.select_option('#localModel', 'local_qwen')
        self.page.select_option('#taskType', task)
        self.page.select_option('#localDevice', device)
        self.page.fill('#maxTokens', '12')
        self.page.fill('#query', 'Return Python code: x = 1' if task == 'code' else 'Say hello briefly.')
        result = self.submit('COMPLETED')
        self.assertTrue(result['is_real_inference'])
        self.assertEqual(result['model']['device'], device)
        self.assertIn('observed', self.page.inner_text('#metricTime'))
        self.assertGreater(result['performance']['adapter_total_ms'], 0)
        self.page.click('[data-tab=trace]')
        self.assertIn('real LOCAL model executed', self.page.inner_text('#tabPanel'))
        folder = ROOT / 'reports' / 'browser'
        folder.mkdir(exist_ok=True)
        self.page.screenshot(path=str(folder / f'qwen_{device}_{task}.png'), full_page=True)

    def test_real_cpu_text(self):
        self.real_inference('CPU', 'text')

    def test_real_gpu_code(self):
        self.real_inference('GPU', 'code')

    def test_llama_default_selection_and_license(self):
        self.assertEqual(self.page.input_value('#localModel'), 'local_llama')
        self.assertIn('Built with Llama', self.page.inner_text('body'))
        self.assertEqual(self.context.request.get(self.origin + '/licenses/llama').status, 200)
        self.assertEqual(self.context.request.get(self.origin + '/licenses/llama-use-policy').status, 200)

    def test_llama_qwen_switching_and_rag_real_inference(self):
        from edgemind.local_models import LLAMA, QWEN
        from edgemind.adapters.openvino_qwen import runtime_hardware_profile
        hardware = runtime_hardware_profile()
        if hardware['available_memory_mb'] < QWEN.required_memory_mb or not hardware['devices'].get('CPU'):
            self.skipTest('BLOCKED: CPU or RAM unavailable for two-model switching')
        self.advanced()
        self.page.select_option('#executionMode', 'local_qwen')
        self.page.select_option('#localDevice', 'CPU')
        self.page.fill('#maxTokens', '32')
        self.page.fill('#query', 'Summarize this document about GPU inference briefly.')
        self.page.fill('#documentText', 'GPU inference accelerates local parallel computation. Private documents remain local.')
        observations = []
        for model in (LLAMA, QWEN, LLAMA):
            self.page.select_option('#localModel', model.mode)
            result = self.submit('COMPLETED')
            self.assertEqual(result['model']['model_id'], model.model_id)
            self.assertTrue(result['is_real_inference'])
            self.assertTrue(result['evidence'])
            observations.append({'model_id': model.model_id, 'performance': result['performance']})
        self.evidence = {'transport': 'real loopback HTTP', 'switch_sequence': observations}
        self.page.screenshot(path=str(ROOT / 'reports/browser/llama_rag_cpu.png'), full_page=True)

    def test_llama_gpu_generation(self):
        from edgemind.local_models import LLAMA
        from edgemind.adapters.openvino_qwen import runtime_hardware_profile
        hardware = runtime_hardware_profile()
        if hardware['available_memory_mb'] < LLAMA.required_memory_mb or not hardware['devices'].get('GPU'):
            self.skipTest('BLOCKED: GPU or RAM unavailable for Llama')
        self.advanced()
        self.page.select_option('#executionMode', 'local_qwen')
        self.page.select_option('#localModel', 'local_llama')
        self.page.select_option('#localDevice', 'GPU')
        self.page.select_option('#taskType', 'code')
        self.page.fill('#maxTokens', '24')
        self.page.fill('#query', 'Return Python code assigning x = 1.')
        result = self.submit('COMPLETED')
        self.assertEqual(result['model']['model_id'], LLAMA.model_id)
        self.assertEqual(result['model']['device'], 'GPU')
        self.assertTrue(result['is_real_inference'])
        self.page.screenshot(path=str(ROOT / 'reports/browser/llama_code_gpu.png'), full_page=True)

    def test_llama_auto_uses_own_observations(self):
        from edgemind.local_models import LLAMA
        from edgemind.benchmark.benchmarks import eligible_observations
        observed, diagnostic = eligible_observations(
            benchmark_file=os.getenv('EDGEMIND_LLAMA_BENCHMARK_FILE', LLAMA.benchmark_file),
            model_path=os.getenv('EDGEMIND_LLAMA_MODEL_PATH', LLAMA.directory),
            model_id=LLAMA.model_id, max_new_tokens=32)
        if not observed:
            self.skipTest('BLOCKED: ' + diagnostic)
        self.advanced()
        self.page.select_option('#executionMode', 'local_qwen')
        self.page.select_option('#localModel', 'local_llama')
        self.page.select_option('#localDevice', 'auto')
        self.page.fill('#maxTokens', '32')
        result = self.submit('COMPLETED')
        self.assertEqual(result['model']['model_id'], LLAMA.model_id)
        self.assertEqual(result['model']['latency_source'], 'observed_cold_adapter_median')
