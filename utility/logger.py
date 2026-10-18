import logging
import time
from functools import wraps
from contextlib import contextmanager
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import json
import os, sys 
from datetime import datetime
from contextlib import nullcontext


# logging.getLogger().setLevel(logging.CRITICAL + 1) # Disable Root Logger

# Save original method
_orig_debug = logging.Logger.debug

def _debug_override(self, msg, *args, **kwargs):
    # If msg is a callable, only call it if DEBUG is enabled
    if callable(msg):
        if self.isEnabledFor(logging.DEBUG):
            text = msg()
            return _orig_debug(self, text, *args, **kwargs)
        else:
            return  # do nothing
    else:
        # msg is a normal string, let original handle it
        return _orig_debug(self, msg, *args, **kwargs)

logging.Logger.debug = _debug_override
_perf_logger = None
_message_logger = None
_torch_profiler = None


def perf_logger():
    global _perf_logger
    if not _perf_logger:
        raise RuntimeError("Perf Logger Not set, run set_perf_logger first")   
    return _perf_logger

def message_logger():
    global _message_logger
    if not _message_logger:
        raise RuntimeError("Message Logger Not set, run set_message_logger first") 
    return _message_logger


def set_perf_logger(filename=None, enable=True):
    global _perf_logger
    _perf_logger = PerformanceTracker()
    if not enable:
        _perf_logger.disable_handler()
    else: 
        _perf_logger.set_handlers(filename)

def set_message_logger(filename=None, enable=True):
    global _message_logger
    _message_logger = setup_message_logger() 
    if not enable:
        disable_handler(_message_logger) 
    else:
        set_handlers(_message_logger, filename, 'debug')

def torch_profiler():
    global _torch_profiler
    if not _torch_profiler:
        raise RuntimeError("Torch Profiler Not set, run set_torch_profiler first") 
    return _torch_profiler

def set_torch_profiler(filename=None, enable=True, torch_profile_dir=None):
    global _torch_profiler 
    _torch_profiler = TorchProfiler(trace_dir=torch_profile_dir)
    if not enable:
        _torch_profiler.disable_handler()
    else: 
        _torch_profiler.set_handlers(filename) 

def default_formatter():
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return formatter

def set_handlers(
    logger: logging.Logger,
    dirname: str = None,
    prefix: str = ''
):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(default_formatter())
    logger.addHandler(stream_handler)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(dirname, f"{prefix}_{timestamp}.log")
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setFormatter(default_formatter())
        logger.addHandler(file_handler)
    
def disable_handler(
    logger: logging.Logger,
):
    logger.setLevel(logging.CRITICAL + 1)


def setup_message_logger(
    name: str = "messaging",
    level: int = logging.DEBUG,
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def setup_profile_logger(
    name: str = "profiling",
    level: int = logging.INFO,
    fmt: str = "%(message)s",
) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class PerformanceTracker:
    """
    Per-tag timers and the perf report.

    :meth:`start` / :meth:`stop` time a tag (a query, an operator, a pipe) on
    the host clock; the sizing search reads a pipe's elapsed time from
    :meth:`stop`.  The only per-tag field the report carries is a pipe's
    ``peak_gpu_bytes``, recorded under ``TQP_EXPERIMENT_TELEMETRY`` (the
    call sites test it).  Per-query sections (:meth:`add_section`) sit
    beside the tags.
    """
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("PerformanceTracker")
        self._peaks = {}
        self._start_times = {}
        self._sections = {}
        self._enable = True

    def set_handlers(self, filename: str):
        set_handlers(self.logger, filename, 'perf')
    
    def disable_handler(self):
        disable_handler(self.logger)
        self._enable = False

    def record(self, tag: str, peak_gpu_bytes: int):
        """Record a pipe's peak GPU allocator-arena occupancy over its
        execution window (bytes, block-rounded): the operator space's peak,
        the cache region not included (``cmp.get_window_peak_bytes``).  A
        high-water mark: a tag keeps the largest value recorded."""
        if not self._enable:
            return
        if peak_gpu_bytes > self._peaks.get(tag, 0):
            self._peaks[tag] = peak_gpu_bytes

    def start(self, tag: str):
        """Begin timing for given tag (wall-clock time)."""
        if not self._enable:
            return
        if tag in self._start_times:
            raise RuntimeError(f"Timing for '{tag}' already started.")
        self._start_times[tag] = time.perf_counter()

    def stop(self, tag: str) -> float:
        """End timing for tag and return the elapsed seconds."""
        if not self._enable:
            return 0.0

        if tag not in self._start_times:
            raise RuntimeError(f"Timing for '{tag}' was not started.")

        return time.perf_counter() - self._start_times.pop(tag)

    @contextmanager
    def time(self, tag: str):
        """Context manager to measure wall-clock time."""
        self.start(tag)
        try:
            yield
        finally:
            self.stop(tag)

    def report(self, filepath: str = "performance_stats.json") -> dict:
        """Serialize the report to JSON: each recorded pipe's
        ``{'totals': {'peak_gpu_bytes': ...}}`` and the sections."""
        if not self._enable:
            return {}

        data = {tag: {'totals': {'peak_gpu_bytes': peak}}
                for tag, peak in self._peaks.items()}
        # Extra non-tag sections (e.g. per-query cache stats) sit alongside
        # the per-operator tag entries at the JSON top level.
        for section_name, section in self._sections.items():
            data[section_name] = section

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        self.logger.info(f"Statistics written to {filepath}")
        return data

    def add_section(self, name: str, section: dict):
        """Attach an extra top-level section (e.g. per-query cache stats) to
        the report.  Emitted by report() alongside the per-operator tags."""
        self._sections[name] = dict(section)


class TorchProfiler:
    def __init__(
        self,
        logger: logging.Logger=None,
        activities=None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        top_k: int = 5,
        trace_dir: str = None
    ):
        # Env overrides let callers disable the expensive collectors (stack
        # unwinding / memory tracking dominate at large scale). Default keeps
        # historical behavior (all True) when the vars are unset.
        def _envflag(name, default):
            v = os.environ.get(name)
            return default if v is None else v not in ('0', 'false', 'False', '')
        self.logger = logger or setup_profile_logger()
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.record_shapes = _envflag('TQP_PROF_SHAPES', record_shapes)
        self.profile_memory = _envflag('TQP_PROF_MEMORY', profile_memory)
        self.with_stack = _envflag('TQP_PROF_STACK', with_stack)
        self.top_k = top_k
        self.profiler = None
        self.trace_dir = trace_dir

    @contextmanager
    def profile(self, tag):
        if not self.trace_dir:
            yield nullcontext(None)
            return
        self.tag = tag
        if self.profile_memory:
            self.record_shapes = True
            self.with_stack    = True
        with profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        ) as prof:
            yield prof
            torch.cuda.synchronize()

        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(self.trace_dir, f'{self.tag}_trace.json'))
        
        if self.profile_memory:
            self.logger.info("Top %d CUDA operations by self memory consumption:", self.top_k)
            cuda_table = prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=self.top_k
            )
            for line in cuda_table.splitlines():
                self.logger.info(line)

            self.logger.info("Top %d CUDA operations by memory consumption:", self.top_k)
            cuda_table = prof.key_averages().table(
                sort_by="cuda_memory_usage", row_limit=self.top_k
            )
            for line in cuda_table.splitlines():
                self.logger.info(line)
    def set_handlers(self, filename: str):
        set_handlers(self.logger, filename, 'profile')
    
    def disable_handler(self):
        disable_handler(self.logger)

