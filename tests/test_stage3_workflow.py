import unittest
from edgemind.contracts import mock_hardware_profile
from edgemind.service import run_task


class TestStage3Workflow(unittest.TestCase):
    def test_entire_workflow_with_no_gpu_chooses_cpu(self):
        hw = mock_hardware_profile()
        hw['devices']['GPU'] = False
        state = run_task('Summarize this', hardware_profile=hw)
        self.assertEqual(state['decision']['device'], 'CPU')
        self.assertEqual(state['status'], 'SIMULATED')

    def test_entire_workflow_low_ram_private_defers(self):
        hw = mock_hardware_profile()
        hw['available_memory_mb'] = 100
        state = run_task('Summarize this', hardware_profile=hw)
        self.assertEqual(state['decision']['outcome'], 'deferred')
        self.assertEqual(state['status'], 'DEFERRED')


if __name__ == '__main__':
    unittest.main()
