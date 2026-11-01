"""Focused tests of the sequential-notebook / RAM-aware in-notebook worker policy."""
import os
import unittest
from unittest.mock import patch

from run_support import resolve_max_workers, resolve_pool_workers


class TestWorkerPolicy(unittest.TestCase):
    """Tests of the CPU- and RAM-aware worker policy."""
    def test_64_cpu_allocation_keeps_two_free(self):
        """With 64 CPUs assigned, two stay free and the count never exceeds the number of tasks."""
        with patch('os.sched_getaffinity', return_value=set(range(64))):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_max_workers(), 62)
                self.assertEqual(resolve_max_workers(n_tasks=8), 8)
                self.assertEqual(resolve_max_workers(n_tasks=100), 62)

    def test_explicit_override_is_only_a_cap(self):
        """THESIS_MAX_WORKERS only lowers the count, it never raises it above CPUs minus two."""
        with patch('os.sched_getaffinity', return_value=set(range(64))):
            with patch.dict(os.environ, {'THESIS_MAX_WORKERS': '100'}, clear=True):
                self.assertEqual(resolve_max_workers(), 62)
            with patch.dict(os.environ, {'THESIS_MAX_WORKERS': '12'}, clear=True):
                self.assertEqual(resolve_max_workers(), 12)

    def test_pool_combines_cpu_tasks_explicit_limit_and_memory(self):
        """The pool size is the minimum of the CPU limit, the tasks, the explicit cap and the memory limit."""
        with patch('os.sched_getaffinity', return_value=set(range(64))):
            with patch('run_support.resolve_memory_workers', side_effect=lambda cpu, **kw: min(cpu, 7)) as mock:
                self.assertEqual(resolve_pool_workers(n_tasks=100), 7)
                self.assertEqual(resolve_pool_workers(n_tasks=5), 5)
                self.assertEqual(resolve_pool_workers(n_tasks=100, max_workers=3), 3)
                self.assertEqual(resolve_pool_workers(n_tasks=100, max_workers=100), 7)
                self.assertEqual(mock.call_count, 4)


if __name__ == '__main__':
    unittest.main()
