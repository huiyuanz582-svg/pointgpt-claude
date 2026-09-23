"""Budget and failure-path tests using a fake CUDA API, never a real GPU."""

from contextlib import redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from utils.gpu_memory import (
    GIB, GPU_OOM_EXIT_CODE, apply_gpu_memory_limit, exit_on_cuda_oom, memory_budget,
)


class MemoryBudgetTests(unittest.TestCase):
    def test_large_gpu_keeps_fixed_cap_even_without_percentage_limit(self):
        for config in ({}, {'gpu_mem_fraction': 0}, {'gpu_mem_fraction': 0.9}):
            with self.subTest(config=config):
                budget = memory_budget(80 * GIB, config)
                self.assertEqual(budget['allocator_limit_bytes'], 47 * GIB)
                self.assertAlmostEqual(budget['fraction'], 47 / 80)

    def test_small_gpu_and_stricter_config_preserve_headroom(self):
        self.assertEqual(memory_budget(24 * GIB, {})['allocator_limit_bytes'], 23 * GIB)
        self.assertEqual(memory_budget(80 * GIB, {'gpu_mem_fraction': 0.25})
                         ['allocator_limit_bytes'], 20 * GIB)
        self.assertEqual(memory_budget(80 * GIB, {'gpu_mem_limit_gib': 32,
                                                'gpu_mem_headroom_gib': 2})
                         ['allocator_limit_bytes'], 30 * GIB)

    def test_invalid_settings_cannot_disable_protection(self):
        cases = ({'gpu_mem_limit_gib': 49}, {'gpu_mem_limit_gib': 0},
                 {'gpu_mem_limit_gib': float('nan')},
                 {'gpu_mem_headroom_gib': 0}, {'gpu_mem_headroom_gib': 48},
                 {'gpu_mem_headroom_gib': float('inf')},
                 {'gpu_mem_fraction': -1}, {'gpu_mem_fraction': 1.1})
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                memory_budget(80 * GIB, config)
        with self.assertRaises(ValueError):
            memory_budget(GIB, {})


class InstallLimitTests(unittest.TestCase):
    def fake_cuda(self):
        return SimpleNamespace(
            is_available=Mock(return_value=True),
            memory=SimpleNamespace(get_allocator_backend=Mock(return_value='native')),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=80 * GIB)),
            memory_reserved=Mock(return_value=0),
            set_per_process_memory_fraction=Mock(),
        )

    def test_installs_on_requested_logical_device(self):
        cuda = self.fake_cuda()
        with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}), \
                redirect_stdout(io.StringIO()):
            budget = apply_gpu_memory_limit({}, 1)
        cuda.set_per_process_memory_fraction.assert_called_once_with(47 / 80, 1)
        self.assertEqual(budget['device'], '1')

    def test_cannot_continue_without_supported_allocator_or_with_excess_allocations(self):
        for reason in ('unavailable', 'backend', 'already_over_budget'):
            cuda = self.fake_cuda()
            if reason == 'unavailable':
                cuda.is_available.return_value = False
            elif reason == 'backend':
                cuda.memory.get_allocator_backend.return_value = 'cudaMallocAsync'
            else:
                cuda.memory_reserved.return_value = 48 * GIB
            with self.subTest(reason=reason), \
                    patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}), \
                    self.assertRaises(RuntimeError):
                apply_gpu_memory_limit({}, 0)
            cuda.set_per_process_memory_fraction.assert_not_called()

    def test_install_error_is_not_swallowed(self):
        cuda = self.fake_cuda()
        error = RuntimeError('driver refused the limit')
        cuda.set_per_process_memory_fraction.side_effect = error
        with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}), \
                self.assertRaises(RuntimeError) as caught:
            apply_gpu_memory_limit({}, 0)
        self.assertIs(caught.exception, error)


class OomExitTests(unittest.TestCase):
    def test_cuda_oom_exits_without_reaching_next_operation(self):
        typed_oom = type('OutOfMemoryError', (RuntimeError,), {})
        for error in (RuntimeError('CUDA out of memory.'),
                      RuntimeError('Caught RuntimeError in replica 0: CUDA out of memory.'),
                      typed_oom('allocation exceeded the allowed memory')):
            reached = []
            with self.subTest(error=error), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as caught:
                with exit_on_cuda_oom():
                    raise error
                reached.append('continued')
            self.assertEqual(caught.exception.code, GPU_OOM_EXIT_CODE)
            self.assertEqual(reached, [])

    def test_unrelated_errors_keep_their_identity(self):
        error = RuntimeError('shape mismatch')
        with self.assertRaises(RuntimeError) as caught:
            with exit_on_cuda_oom():
                raise error
        self.assertIs(caught.exception, error)


if __name__ == '__main__':
    unittest.main()
