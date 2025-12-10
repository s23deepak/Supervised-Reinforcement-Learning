#!/usr/bin/env python3
"""
Resource Monitor for SRL Training

Provides utilities to log CPU, GPU, VRAM, and RAM usage at each training phase.
Supports logging to console, TensorBoard, and Weights & Biases (wandb).

This helps understand resource efficiency during:
- Generation phase (vLLM active)
- Reward calculation (CPU-bound)
- Training phase (gradients + optimizer)
"""

import time
import psutil
from typing import Optional, Dict, Any, Literal
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import GPU monitoring libraries
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except (ImportError, Exception):
    HAS_PYNVML = False

# Try to import logging backends
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class LoggingBackend(Enum):
    """Available logging backends."""
    CONSOLE = "console"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_utilization: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_percent: Optional[float] = None
    gpu_name: Optional[str] = None
    
    def __str__(self) -> str:
        lines = [
            f"  CPU: {self.cpu_percent:5.1f}%",
            f"  RAM: {self.ram_used_gb:.2f}/{self.ram_total_gb:.2f} GB ({self.ram_percent:.1f}%)",
        ]
        if self.gpu_utilization is not None:
            lines.append(f"  GPU: {self.gpu_utilization:5.1f}%")
        if self.vram_used_gb is not None:
            lines.append(f"  VRAM: {self.vram_used_gb:.2f}/{self.vram_total_gb:.2f} GB ({self.vram_percent:.1f}%)")
        return "\n".join(lines)
    
    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Convert to flat dictionary for logging backends."""
        result = {
            f"{prefix}cpu_percent": self.cpu_percent,
            f"{prefix}ram_used_gb": self.ram_used_gb,
            f"{prefix}ram_percent": self.ram_percent,
        }
        if self.gpu_utilization is not None:
            result[f"{prefix}gpu_utilization"] = self.gpu_utilization
        if self.vram_used_gb is not None:
            result[f"{prefix}vram_used_gb"] = self.vram_used_gb
        if self.vram_percent is not None:
            result[f"{prefix}vram_percent"] = self.vram_percent
        return result


class ResourceMonitor:
    """
    Monitor system resources during training phases.
    
    Supports logging to:
    - Console (verbose print statements)
    - TensorBoard (time series graphs)
    - Weights & Biases (interactive dashboards)
    
    Usage:
        monitor = ResourceMonitor(backend="wandb", project="srl-training")
        
        with monitor.log_phase("Generation", step=100):
            # generation code here
            pass
        
        # Or manual logging:
        snapshot = monitor.get_snapshot()
        monitor.log_metrics({"loss": 0.5}, step=100)
    """
    
    def __init__(
        self, 
        gpu_index: int = 0, 
        verbose: bool = True,
        backend: Literal["console", "tensorboard", "wandb"] = "console",
        log_dir: str = "./runs",
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize resource monitor.
        
        Args:
            gpu_index: GPU index to monitor (default 0)
            verbose: If True, print logs to console
            backend: Logging backend ("console", "tensorboard", "wandb")
            log_dir: Directory for TensorBoard logs
            project: WandB project name
            run_name: WandB run name
            config: Config dict to log with WandB
        """
        self.gpu_index = gpu_index
        self.verbose = verbose
        self.backend = LoggingBackend(backend)
        self._handle = None
        self._step = 0
        
        # Initialize NVML handle if available
        if HAS_PYNVML:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._handle = None
        
        # Initialize logging backend
        self._writer = None
        self._wandb_run = None
        
        if self.backend == LoggingBackend.TENSORBOARD:
            if not HAS_TENSORBOARD:
                print("⚠ TensorBoard not installed. Falling back to console logging.")
                print("  Install with: pip install tensorboard")
                self.backend = LoggingBackend.CONSOLE
            else:
                self._writer = SummaryWriter(log_dir=log_dir)
                print(f"📊 TensorBoard logging initialized: {log_dir}")
                print(f"   View with: tensorboard --logdir {log_dir}")
        
        elif self.backend == LoggingBackend.WANDB:
            if not HAS_WANDB:
                print("⚠ wandb not installed. Falling back to console logging.")
                print("  Install with: pip install wandb")
                self.backend = LoggingBackend.CONSOLE
            else:
                # Initialize wandb run
                self._wandb_run = wandb.init(
                    project=project or "srl-training",
                    name=run_name,
                    config=config,
                    reinit=True,
                )
                print(f"📊 Weights & Biases logging initialized")
                print(f"   View at: {self._wandb_run.url}")
    
    def get_cpu_percent(self) -> float:
        """Get current CPU utilization percentage."""
        return psutil.cpu_percent(interval=None)  # Non-blocking
    
    def get_ram_info(self) -> Dict[str, float]:
        """Get RAM usage information in GB."""
        mem = psutil.virtual_memory()
        return {
            "used_gb": mem.used / (1024**3),
            "total_gb": mem.total / (1024**3),
            "percent": mem.percent,
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU utilization and VRAM usage."""
        result = {
            "utilization": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "vram_percent": None,
            "name": None,
        }
        
        # Try NVML first (more accurate)
        if self._handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                name = pynvml.nvmlDeviceGetName(self._handle)
                
                result["utilization"] = util.gpu
                result["vram_used_gb"] = mem.used / (1024**3)
                result["vram_total_gb"] = mem.total / (1024**3)
                result["vram_percent"] = (mem.used / mem.total) * 100
                result["name"] = name if isinstance(name, str) else name.decode()
                return result
            except Exception:
                pass
        
        # Fallback to PyTorch CUDA
        if HAS_TORCH and torch.cuda.is_available():
            try:
                result["vram_used_gb"] = torch.cuda.memory_allocated(self.gpu_index) / (1024**3)
                result["vram_total_gb"] = torch.cuda.get_device_properties(self.gpu_index).total_memory / (1024**3)
                if result["vram_total_gb"] > 0:
                    result["vram_percent"] = (result["vram_used_gb"] / result["vram_total_gb"]) * 100
                result["name"] = torch.cuda.get_device_name(self.gpu_index)
            except Exception:
                pass
        
        return result
    
    def get_snapshot(self) -> ResourceSnapshot:
        """Get a complete snapshot of current system resources."""
        # Trigger CPU measurement
        psutil.cpu_percent(interval=None)
        time.sleep(0.05)  # Brief pause for measurement
        
        ram = self.get_ram_info()
        gpu = self.get_gpu_info()
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=self.get_cpu_percent(),
            ram_used_gb=ram["used_gb"],
            ram_total_gb=ram["total_gb"],
            ram_percent=ram["percent"],
            gpu_utilization=gpu["utilization"],
            vram_used_gb=gpu["vram_used_gb"],
            vram_total_gb=gpu["vram_total_gb"],
            vram_percent=gpu["vram_percent"],
            gpu_name=gpu["name"],
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the configured backend.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (auto-incremented if not provided)
        """
        if step is None:
            step = self._step
        
        if self.backend == LoggingBackend.TENSORBOARD and self._writer is not None:
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, step)
        
        elif self.backend == LoggingBackend.WANDB and self._wandb_run is not None:
            wandb.log(metrics, step=step)
    
    def log_snapshot_metrics(self, snapshot: ResourceSnapshot, prefix: str = "", step: Optional[int] = None):
        """Log a resource snapshot to the configured backend."""
        metrics = snapshot.to_dict(prefix=prefix)
        self.log_metrics(metrics, step=step)
    
    @contextmanager
    def log_phase(self, phase_name: str, step: Optional[int] = None):
        """
        Context manager to log resources at start and end of a phase.
        
        Args:
            phase_name: Name of the phase (e.g., "Generation", "Training")
            step: Training step for logging
        
        Usage:
            with monitor.log_phase("Generation", step=100):
                # generation code
                pass
        """
        if step is None:
            step = self._step
        
        start_time = time.time()
        start_snapshot = self.get_snapshot()
        
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"📊 Phase: {phase_name} [START]")
            print(start_snapshot)
        
        # Log start metrics
        phase_prefix = f"resources/{phase_name.lower().replace(' ', '_')}/"
        self.log_snapshot_metrics(start_snapshot, prefix=f"{phase_prefix}start/", step=step)
        
        try:
            yield start_snapshot
        finally:
            end_time = time.time()
            end_snapshot = self.get_snapshot()
            duration = end_time - start_time
            
            if self.verbose:
                print(f"\n📊 Phase: {phase_name} [END] ({duration:.2f}s)")
                print(end_snapshot)
                
                # Show deltas
                ram_delta = end_snapshot.ram_used_gb - start_snapshot.ram_used_gb
                print(f"\n  Δ RAM: {ram_delta:+.2f} GB")
                
                if end_snapshot.vram_used_gb is not None and start_snapshot.vram_used_gb is not None:
                    vram_delta = end_snapshot.vram_used_gb - start_snapshot.vram_used_gb
                    print(f"  Δ VRAM: {vram_delta:+.2f} GB")
                
                print(f"{'─'*50}")
            
            # Log end metrics and duration
            self.log_snapshot_metrics(end_snapshot, prefix=f"{phase_prefix}end/", step=step)
            self.log_metrics({
                f"{phase_prefix}duration_sec": duration,
                f"{phase_prefix}ram_delta_gb": end_snapshot.ram_used_gb - start_snapshot.ram_used_gb,
            }, step=step)
            
            if end_snapshot.vram_used_gb is not None and start_snapshot.vram_used_gb is not None:
                self.log_metrics({
                    f"{phase_prefix}vram_delta_gb": end_snapshot.vram_used_gb - start_snapshot.vram_used_gb,
                }, step=step)
    
    def log_snapshot(self, label: str, step: Optional[int] = None) -> ResourceSnapshot:
        """
        Log a single snapshot with a label.
        
        Args:
            label: Description of when this snapshot was taken
            step: Training step for logging
            
        Returns:
            The captured ResourceSnapshot
        """
        snapshot = self.get_snapshot()
        if self.verbose:
            print(f"\n📌 {label}")
            print(snapshot)
        
        # Log to backend
        prefix = f"resources/{label.lower().replace(' ', '_').replace('/', '_')}/"
        self.log_snapshot_metrics(snapshot, prefix=prefix, step=step)
        
        return snapshot
    
    def set_step(self, step: int):
        """Set the current training step for auto-incrementing."""
        self._step = step
    
    def increment_step(self):
        """Increment the training step counter."""
        self._step += 1
    
    def close(self):
        """Cleanup logging backends."""
        if self._writer is not None:
            self._writer.close()
        if self._wandb_run is not None:
            wandb.finish()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global instance for convenience
_global_monitor: Optional[ResourceMonitor] = None


def get_monitor(
    verbose: bool = True,
    backend: str = "console",
    **kwargs
) -> ResourceMonitor:
    """Get or create the global ResourceMonitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor(verbose=verbose, backend=backend, **kwargs)
    return _global_monitor
