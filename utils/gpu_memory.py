"""Per-process PyTorch CUDA allocation budget; no CUDA work at import time.

This is not a Docker-wide VRAM quota. CUDA contexts and allocations made
outside PyTorch's allocator are not covered; the headroom is only a reserve.
"""

from contextlib import contextmanager
import math
import sys


GIB = 1024 ** 3
MAX_GPU_MEMORY_GIB = 48.0
GPU_OOM_EXIT_CODE = 86


def memory_budget(total_bytes, config):
    """Return a serializable budget; callers may lower, but not raise, 48 GiB."""
    limit = float(config.get('gpu_mem_limit_gib', MAX_GPU_MEMORY_GIB))
    headroom = float(config.get('gpu_mem_headroom_gib', 1.0))
    # A legacy zero fraction means no additional percentage restriction. The
    # absolute budget still applies even when that old knob is absent/zero.
    fraction = float(config.get('gpu_mem_fraction', 1.0) or 1.0)
    if not math.isfinite(limit) or not 1.0 < limit <= MAX_GPU_MEMORY_GIB:
        raise ValueError('gpu_mem_limit_gib must be > 1 and <= 48 GiB')
    if not math.isfinite(headroom) or not 1.0 <= headroom < limit:
        raise ValueError('gpu_mem_headroom_gib must be >= 1 and less than gpu_mem_limit_gib')
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError('gpu_mem_fraction must be in (0, 1], or 0 for no extra percentage cap')
    total_bytes = int(total_bytes)
    ceiling = min(int(limit * GIB), total_bytes)
    allocator_bytes = min(ceiling - int(headroom * GIB), int(total_bytes * fraction))
    if total_bytes <= 0 or allocator_bytes <= 0:
        raise ValueError('GPU memory is too small for the requested headroom')
    return dict(limit_gib=limit, headroom_gib=headroom, total_bytes=total_bytes,
                allocator_limit_bytes=allocator_bytes, fraction=allocator_bytes / total_bytes,
                scope='per_process_per_cuda_device_pytorch_allocator',
                oom_action='exit_nonzero_without_retry', oom_exit_code=GPU_OOM_EXIT_CODE)


def apply_gpu_memory_limit(config, device, logger=None):
    """Install before models/checkpoints are moved to CUDA. Fail closed."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Cannot install GPU memory limit: CUDA is unavailable')
    # The deployed torch 2.0.1 exposes this API. Require a known backend rather
    # than silently continuing if a custom/async allocator ignores the cap.
    backend = torch.cuda.memory.get_allocator_backend()
    if backend != 'native':
        raise RuntimeError('GPU memory limit requires the native CUDA allocator; '
                           'remove backend:cudaMallocAsync/custom allocator settings and restart')
    total = torch.cuda.get_device_properties(device).total_memory
    budget = memory_budget(total, config)
    if torch.cuda.memory_reserved(device) > budget['allocator_limit_bytes']:
        raise RuntimeError('GPU allocations already exceed the budget; start a new process '
                           'and install the limit before loading a model')
    # Do not swallow errors here. Starting without the requested limit is unsafe.
    torch.cuda.set_per_process_memory_fraction(float(budget['fraction']), device)
    budget.update(device=str(device), allocator_backend=backend)
    message = (f'[GPU memory limit] device={device}: PyTorch allocator <= '
               f'{budget["allocator_limit_bytes"] / GIB:.2f} GiB; '
               f'target <= {budget["limit_gib"]:g} GiB including '
               f'{budget["headroom_gib"]:g} GiB external-allocation headroom. '
               f'CUDA OOM terminates this process (exit {GPU_OOM_EXIT_CODE}); no batch retry. '
               'External CUDA allocations are not hard-capped by this allocator.')
    print(message, flush=True)
    if logger is not None:
        logger.info(message)
    return budget


def is_cuda_oom(error):
    """Also recognize CUDA OOM wrapped by DataParallel or an extension."""
    text = str(error).lower()
    return (type(error).__name__ == 'OutOfMemoryError' or
            (isinstance(error, RuntimeError) and 'out of memory' in text and
             ('cuda' in text or 'cudnn' in text)))


@contextmanager
def exit_on_cuda_oom():
    """Unwind only this command. Never retry, save a checkpoint, or kill a peer."""
    try:
        yield
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        print(f'[GPU memory limit] CUDA OOM: stopping the current process '
              f'(exit {GPU_OOM_EXIT_CODE}). No retry or emergency checkpoint.\n{error}',
              file=sys.stderr, flush=True)
        raise SystemExit(GPU_OOM_EXIT_CODE) from error
