"""Real Llama CPU/GPU, code, RAG and benchmark-aware inference (opt-in)."""
from tests import test_live_hardware as hardware
from edgemind.local_models import LLAMA
import os


class LlamaInferenceTests(hardware.LocalInferenceTests):
    def exercise(self, device, task_type, **kwargs):
        return super().exercise(device, task_type, mode=LLAMA.mode, **kwargs)

    def test_llama_rag_summary_cpu(self):
        self.exercise('CPU', 'text', max_new_tokens=32, documents=[
            {'source_id': 'gpu-guide', 'text': 'The GPU accelerates local parallel computation. Private documents remain local.'}])

    def test_llama_auto_from_real_benchmarks(self):
        from edgemind.benchmark.benchmarks import eligible_observations
        observed, diagnostic = eligible_observations(
            benchmark_file=os.getenv('EDGEMIND_LLAMA_BENCHMARK_FILE', LLAMA.benchmark_file),
            model_path=os.getenv('EDGEMIND_LLAMA_MODEL_PATH', LLAMA.directory),
            model_id=LLAMA.model_id, max_new_tokens=32)
        if not observed:
            self.skipTest('BLOCKED: ' + diagnostic)
        self.exercise('auto', 'text', max_new_tokens=32)
