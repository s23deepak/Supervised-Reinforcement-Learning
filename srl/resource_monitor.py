#!/usr/bin/env python3
"""
Resource Monitor for SRL Training

Logs CPU, GPU, RAM, and VRAM usage to TensorBoard.
"""

import os
import threading
import time
from typing import Optional
from dataclasses import dataclass

import psutil
import torch
from transformers import TrainerCallback


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_util_percent: float
    vram_used_gb: float
    vram_total_gb: float
    phase: str = "unknown"


class ResourceMonitor:
    """Background thread that samples resource usage."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.samples: list[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_phase = "idle"
        
    def _get_gpu_stats(self):
        """Get GPU utilization and VRAM usage."""
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util = util.gpu
            vram_used = mem.used / (1024**3)
            vram_total = mem.total / (1024**3)
            pynvml.nvmlShutdown()
            return gpu_util, vram_used, vram_total
        except ImportError:
            # Fallback to torch
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return 0.0, vram_used, vram_total
            
    def _sample(self) -> ResourceSnapshot:
        """Take a single resource snapshot."""
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        gpu_util, vram_used, vram_total = self._get_gpu_stats()
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu,
            ram_used_gb=ram.used / (1024**3),
            ram_total_gb=ram.total / (1024**3),
            gpu_util_percent=gpu_util,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            phase=self._current_phase,
        )
        
    def _run(self):
        """Background sampling loop."""
        while self._running:
            self.samples.append(self._sample())
            time.sleep(self.sample_interval)
            
    def start(self):
        """Start background monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def set_phase(self, phase: str):
        """Set current training phase (generation, training, etc.)."""
        self._current_phase = phase
        
    def get_latest(self) -> Optional[ResourceSnapshot]:
        """Get most recent sample."""
        return self.samples[-1] if self.samples else None


class ResourceMonitorCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that logs resource usage to TensorBoard.
    """
    
    def __init__(self, sample_interval: float = 2.0):
        self.monitor = ResourceMonitor(sample_interval)
        self._step = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.monitor.start()
        self.monitor.set_phase("training")
        
    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.stop()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.monitor.set_phase("step_begin")
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._step += 1
        self.monitor.set_phase("step_end")
        
        # Log to TensorBoard every step
        snapshot = self.monitor.get_latest()
        if snapshot and hasattr(state, 'log_history'):
            # These will be logged by the trainer
            pass
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Inject resource metrics into training logs."""
        snapshot = self.monitor.get_latest()
        if snapshot and logs is not None:
            logs["system/cpu_percent"] = snapshot.cpu_percent
            logs["system/ram_used_gb"] = snapshot.ram_used_gb
            logs["system/gpu_util_percent"] = snapshot.gpu_util_percent
            logs["system/vram_used_gb"] = snapshot.vram_used_gb
