#!/usr/bin/env python3
"""
Unified Logger for SRL/RLVR Training

Comprehensive logging system that tracks:
- Model metrics: loss, KL, samples_kept, rewards
- Resource usage: CPU, GPU, RAM, VRAM (overall + phase-wise)
- Output formats: TensorBoard, CSV, matplotlib plots

Usage:
    from unified_logger import UnifiedLoggerCallback, patch_trainer
    
    callback = UnifiedLoggerCallback(output_dir="./checkpoints")
    patch_trainer()
    
    trainer = GRPOTrainer(..., callbacks=[callback])
"""

import os
import time
import threading
import csv
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

import psutil
import torch
from transformers import TrainerCallback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from trl import GRPOTrainer


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""
    timestamp: float
    step: int
    phase: str
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_util_percent: float
    vram_used_gb: float
    vram_total_gb: float


@dataclass
class PhaseMetrics:
    """Metrics for a single phase execution."""
    phase: str
    step: int
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_gpu_percent: float = 0.0
    vram_used_gb: float = 0.0


@dataclass 
class StepMetrics:
    """Metrics for a training step."""
    step: int
    timestamp: float
    # Model metrics
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    kl: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    samples_kept: int = 0
    samples_total: int = 0
    completion_length: float = 0.0
    # Resource metrics
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    vram_used_gb: float = 0.0
    # Phase durations
    generation_ms: float = 0.0
    reward_ms: float = 0.0
    training_ms: float = 0.0


class ResourceMonitor:
    """Background thread that samples resource usage."""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.samples: List[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._current_step = 0
        self._current_phase = "idle"
        
    def _get_gpu_stats(self):
        """Get GPU utilization and VRAM."""
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
        except:
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
            step=self._current_step,
            phase=self._current_phase,
            cpu_percent=cpu,
            ram_used_gb=ram.used / (1024**3),
            ram_total_gb=ram.total / (1024**3),
            gpu_util_percent=gpu_util,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
        )
        
    def _run(self):
        """Background sampling loop."""
        while self._running:
            sample = self._sample()
            with self._lock:
                self.samples.append(sample)
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
            
    def set_step(self, step: int):
        """Update current step."""
        self._current_step = step
        
    def set_phase(self, phase: str):
        """Update current phase."""
        self._current_phase = phase
        
    def get_latest(self) -> Optional[ResourceSnapshot]:
        """Get most recent sample."""
        with self._lock:
            return self.samples[-1] if self.samples else None
            
    def get_samples_since(self, timestamp: float) -> List[ResourceSnapshot]:
        """Get samples since a timestamp."""
        with self._lock:
            return [s for s in self.samples if s.timestamp >= timestamp]


class PhaseTracker:
    """Tracks timing and resources per training phase."""
    
    PHASES = ["generation", "reward", "training"]
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.monitor = resource_monitor
        self.current_phase: Optional[str] = None
        self.phase_start_time: float = 0.0
        self.phase_history: List[PhaseMetrics] = []
        self._current_step = 0
        
        # Aggregated stats
        self.total_duration = {p: 0.0 for p in self.PHASES}
        self.total_count = {p: 0 for p in self.PHASES}
        
    def set_step(self, step: int):
        """Update current step."""
        self._current_step = step
        
    def begin_phase(self, phase: str):
        """Start tracking a phase."""
        self.end_phase()  # End previous
        self.current_phase = phase
        self.phase_start_time = time.time()
        self.monitor.set_phase(phase)
        
    def end_phase(self):
        """End current phase and record metrics."""
        if self.current_phase is None:
            return
            
        end_time = time.time()
        duration_ms = (end_time - self.phase_start_time) * 1000
        
        # Get samples during this phase
        samples = self.monitor.get_samples_since(self.phase_start_time)
        avg_cpu = sum(s.cpu_percent for s in samples) / len(samples) if samples else 0
        avg_gpu = sum(s.gpu_util_percent for s in samples) / len(samples) if samples else 0
        vram = samples[-1].vram_used_gb if samples else 0
        
        metrics = PhaseMetrics(
            phase=self.current_phase,
            step=self._current_step,
            start_time=self.phase_start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            avg_cpu_percent=avg_cpu,
            avg_gpu_percent=avg_gpu,
            vram_used_gb=vram,
        )
        
        self.phase_history.append(metrics)
        
        # Update aggregates
        if self.current_phase in self.PHASES:
            self.total_duration[self.current_phase] += duration_ms
            self.total_count[self.current_phase] += 1
        
        self.current_phase = None
        self.monitor.set_phase("idle")
        
    def get_latest_durations(self) -> Dict[str, float]:
        """Get most recent duration for each phase."""
        latest = {p: 0.0 for p in self.PHASES}
        for m in reversed(self.phase_history):
            if m.phase in latest and latest[m.phase] == 0:
                latest[m.phase] = m.duration_ms
        return latest


class UnifiedLogger:
    """Main logger that combines all metrics."""
    
    def __init__(self, output_dir: str, sample_interval: float = 0.5):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(sample_interval)
        self.phase_tracker = PhaseTracker(self.resource_monitor)
        
        # Metrics storage
        self.step_metrics: List[StepMetrics] = []
        self._current_step = 0
        self._samples_kept = 0
        self._samples_total = 0
        
        # TensorBoard
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir)
            
    def start(self):
        """Start monitoring."""
        self.resource_monitor.start()
        
    def stop(self):
        """Stop monitoring and finalize."""
        self.phase_tracker.end_phase()
        self.resource_monitor.stop()
        if self.tb_writer:
            self.tb_writer.close()
            
    def set_step(self, step: int):
        """Update current step."""
        self._current_step = step
        self.resource_monitor.set_step(step)
        self.phase_tracker.set_step(step)
        
    def begin_phase(self, phase: str):
        """Start a phase."""
        self.phase_tracker.begin_phase(phase)
        
    def end_phase(self):
        """End current phase."""
        self.phase_tracker.end_phase()
        
    def log_samples(self, kept: int, total: int):
        """Log dynamic filter sample counts."""
        self._samples_kept = kept
        self._samples_total = total
        
    def log_step(self, logs: Dict[str, Any]):
        """Log a training step with all metrics."""
        resource = self.resource_monitor.get_latest()
        phase_durations = self.phase_tracker.get_latest_durations()
        
        metrics = StepMetrics(
            step=self._current_step,
            timestamp=time.time(),
            # Model metrics
            loss=logs.get("loss", 0.0),
            grad_norm=logs.get("grad_norm", 0.0),
            learning_rate=logs.get("learning_rate", 0.0),
            kl=logs.get("kl", 0.0),
            reward_mean=logs.get("reward", logs.get("rewards/reward_fn/mean", 0.0)),
            reward_std=logs.get("reward_std", logs.get("rewards/reward_fn/std", 0.0)),
            samples_kept=self._samples_kept,
            samples_total=self._samples_total,
            completion_length=logs.get("completion_length", 0.0),
            # Resource metrics
            cpu_percent=resource.cpu_percent if resource else 0,
            gpu_percent=resource.gpu_util_percent if resource else 0,
            ram_used_gb=resource.ram_used_gb if resource else 0,
            vram_used_gb=resource.vram_used_gb if resource else 0,
            # Phase durations
            generation_ms=phase_durations.get("generation", 0),
            reward_ms=phase_durations.get("reward", 0),
            training_ms=phase_durations.get("training", 0),
        )
        
        self.step_metrics.append(metrics)
        
        # Log to TensorBoard
        if self.tb_writer:
            step = self._current_step
            # Model metrics
            self.tb_writer.add_scalar("train/loss", metrics.loss, step)
            self.tb_writer.add_scalar("train/grad_norm", metrics.grad_norm, step)
            self.tb_writer.add_scalar("train/learning_rate", metrics.learning_rate, step)
            self.tb_writer.add_scalar("train/kl", metrics.kl, step)
            self.tb_writer.add_scalar("train/reward_mean", metrics.reward_mean, step)
            self.tb_writer.add_scalar("train/reward_std", metrics.reward_std, step)
            self.tb_writer.add_scalar("train/samples_kept", metrics.samples_kept, step)
            self.tb_writer.add_scalar("train/completion_length", metrics.completion_length, step)
            # Resources
            self.tb_writer.add_scalar("resources/cpu_percent", metrics.cpu_percent, step)
            self.tb_writer.add_scalar("resources/gpu_percent", metrics.gpu_percent, step)
            self.tb_writer.add_scalar("resources/ram_gb", metrics.ram_used_gb, step)
            self.tb_writer.add_scalar("resources/vram_gb", metrics.vram_used_gb, step)
            # Phase durations
            self.tb_writer.add_scalar("phase/generation_ms", metrics.generation_ms, step)
            self.tb_writer.add_scalar("phase/reward_ms", metrics.reward_ms, step)
            self.tb_writer.add_scalar("phase/training_ms", metrics.training_ms, step)
            
    def save_csv(self):
        """Save all metrics to CSV files."""
        # Step metrics
        if self.step_metrics:
            path = os.path.join(self.output_dir, "metrics.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.step_metrics[0]).keys())
                writer.writeheader()
                for m in self.step_metrics:
                    writer.writerow(asdict(m))
            print(f"Saved: {path}")
            
        # Resource samples
        if self.resource_monitor.samples:
            path = os.path.join(self.output_dir, "resource_samples.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.resource_monitor.samples[0]).keys())
                writer.writeheader()
                for s in self.resource_monitor.samples:
                    writer.writerow(asdict(s))
            print(f"Saved: {path}")
            
        # Phase metrics
        if self.phase_tracker.phase_history:
            path = os.path.join(self.output_dir, "phase_metrics.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.phase_tracker.phase_history[0]).keys())
                writer.writeheader()
                for m in self.phase_tracker.phase_history:
                    writer.writerow(asdict(m))
            print(f"Saved: {path}")
            
    def plot_all(self):
        """Generate all matplotlib plots."""            
        self._plot_resources()
        self._plot_training_curves()
        self._plot_phase_breakdown()
        
    def _plot_resources(self):
        """Plot 4-panel resource usage over time."""
        samples = self.resource_monitor.samples
        if not samples:
            return
            
        # Get relative time
        start_time = samples[0].timestamp
        times = [(s.timestamp - start_time) for s in samples]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Resource Usage During Training", fontsize=14)
        
        # CPU
        ax = axes[0, 0]
        ax.fill_between(times, [s.cpu_percent for s in samples], alpha=0.3, color='blue')
        ax.plot(times, [s.cpu_percent for s in samples], color='blue', linewidth=0.5)
        ax.set_ylabel("CPU %")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("CPU Utilization")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # GPU
        ax = axes[0, 1]
        ax.fill_between(times, [s.gpu_util_percent for s in samples], alpha=0.3, color='orange')
        ax.plot(times, [s.gpu_util_percent for s in samples], color='orange', linewidth=0.5)
        ax.set_ylabel("GPU %")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("GPU Utilization")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # RAM
        ax = axes[1, 0]
        ram_total = samples[0].ram_total_gb
        ax.plot(times, [s.ram_used_gb for s in samples], color='green', linewidth=1)
        ax.axhline(y=ram_total, color='red', linestyle='--', label=f'Total: {ram_total:.1f} GB')
        ax.set_ylabel("RAM (GB)")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("RAM Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # VRAM
        ax = axes[1, 1]
        vram_total = samples[0].vram_total_gb
        ax.plot(times, [s.vram_used_gb for s in samples], color='purple', linewidth=1)
        ax.axhline(y=vram_total, color='red', linestyle='--', label=f'Total: {vram_total:.1f} GB')
        ax.set_ylabel("VRAM (GB)")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("VRAM Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.plots_dir, "resources.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")
        
    def _plot_training_curves(self):
        """Plot loss, KL, and reward curves."""
        if not self.step_metrics:
            return
            
        steps = [m.step for m in self.step_metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Curves", fontsize=14)
        
        # Loss
        ax = axes[0, 0]
        ax.plot(steps, [m.loss for m in self.step_metrics], 'b-', linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        
        # KL
        ax = axes[0, 1]
        ax.plot(steps, [m.kl for m in self.step_metrics], 'r-', linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence")
        ax.grid(True, alpha=0.3)
        
        # Reward
        ax = axes[1, 0]
        ax.plot(steps, [m.reward_mean for m in self.step_metrics], 'g-', linewidth=1.5, label='Mean')
        ax.fill_between(steps, 
                        [m.reward_mean - m.reward_std for m in self.step_metrics],
                        [m.reward_mean + m.reward_std for m in self.step_metrics],
                        alpha=0.3, color='green')
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward (with std)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Samples kept
        ax = axes[1, 1]
        ax.bar(steps, [m.samples_kept for m in self.step_metrics], color='purple', alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Samples Kept")
        ax.set_title("Samples Kept (after dynamic filter)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.plots_dir, "training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")
        
    def _plot_phase_breakdown(self):
        """Plot per-phase resource usage - 3 rows (phases) x 2 cols (CPU, GPU)."""
        samples = self.resource_monitor.samples
        if not samples:
            return
            
        phases = ["generation", "reward", "training"]
        colors = {'generation': '#2ecc71', 'reward': '#3498db', 'training': '#e74c3c'}
        
        # Get samples for each phase
        phase_samples = {p: [] for p in phases}
        for s in samples:
            if s.phase in phases:
                phase_samples[s.phase].append(s)
        
        # Skip if no phase data
        active_phases = [p for p in phases if phase_samples[p]]
        if not active_phases:
            return
        
        fig, axes = plt.subplots(len(active_phases), 2, figsize=(14, 4 * len(active_phases)))
        fig.suptitle("Resource Usage by Phase", fontsize=14, fontweight='bold')
        
        # Handle single phase case (axes not 2D)
        if len(active_phases) == 1:
            axes = [axes]
        
        for row, phase in enumerate(active_phases):
            ps = phase_samples[phase]
            if not ps:
                continue
                
            # Get relative times for this phase
            start_time = ps[0].timestamp
            times = [(s.timestamp - start_time) for s in ps]
            
            color = colors[phase]
            
            # CPU subplot
            ax = axes[row][0]
            ax.fill_between(times, [s.cpu_percent for s in ps], alpha=0.3, color=color)
            ax.plot(times, [s.cpu_percent for s in ps], color=color, linewidth=0.8)
            ax.set_ylabel("CPU %")
            ax.set_ylim(0, 100)
            ax.set_title(f"{phase.upper()} - CPU Utilization")
            ax.grid(True, alpha=0.3)
            
            # Add average line
            avg_cpu = sum(s.cpu_percent for s in ps) / len(ps)
            ax.axhline(y=avg_cpu, color='red', linestyle='--', alpha=0.7, 
                      label=f'Avg: {avg_cpu:.1f}%')
            ax.legend(loc='upper right')
            
            # GPU subplot
            ax = axes[row][1]
            ax.fill_between(times, [s.gpu_util_percent for s in ps], alpha=0.3, color=color)
            ax.plot(times, [s.gpu_util_percent for s in ps], color=color, linewidth=0.8)
            ax.set_ylabel("GPU %")
            ax.set_ylim(0, 100)
            ax.set_title(f"{phase.upper()} - GPU Utilization")
            ax.grid(True, alpha=0.3)
            
            # Add average line
            avg_gpu = sum(s.gpu_util_percent for s in ps) / len(ps)
            ax.axhline(y=avg_gpu, color='red', linestyle='--', alpha=0.7,
                      label=f'Avg: {avg_gpu:.1f}%')
            ax.legend(loc='upper right')
            
            # Add x-label only to bottom row
            if row == len(active_phases) - 1:
                axes[row][0].set_xlabel("Time within phase (seconds)")
                axes[row][1].set_xlabel("Time within phase (seconds)")
        
        plt.tight_layout()
        path = os.path.join(self.plots_dir, "phase_breakdown.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")
        
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Phase summary
        tracker = self.phase_tracker
        for phase in tracker.PHASES:
            if tracker.total_count[phase] > 0:
                avg = tracker.total_duration[phase] / tracker.total_count[phase]
                total = tracker.total_duration[phase] / 1000
                print(f"\n{phase.upper()} PHASE ({tracker.total_count[phase]} calls):")
                print(f"  Total: {total:.1f}s, Avg: {avg:.1f}ms")
                
        # Resource summary
        samples = self.resource_monitor.samples
        if samples:
            avg_cpu = sum(s.cpu_percent for s in samples) / len(samples)
            avg_gpu = sum(s.gpu_util_percent for s in samples) / len(samples)
            max_vram = max(s.vram_used_gb for s in samples)
            print(f"\nRESOURCES:")
            print(f"  Avg CPU: {avg_cpu:.1f}%")
            print(f"  Avg GPU: {avg_gpu:.1f}%")
            print(f"  Max VRAM: {max_vram:.2f} GB")
            
        print("\n" + "="*60)


class UnifiedLoggerCallback(TrainerCallback):
    """Trainer callback that integrates UnifiedLogger."""
    
    def __init__(self, output_dir: str = "./checkpoints", sample_interval: float = 0.5):
        self.logger = UnifiedLogger(output_dir, sample_interval)
        self._step = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.start()
        print("[UnifiedLogger] Started comprehensive logging")
        
    def on_train_end(self, args, state, control, **kwargs):
        self.logger.stop()
        self.logger.save_csv()
        self.logger.plot_all()
        self.logger.print_summary()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self._step += 1
        self.logger.set_step(self._step)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.log_step(logs)


# Global logger reference for phase tracking
_global_logger: Optional[UnifiedLogger] = None


def set_global_logger(logger: UnifiedLogger):
    """Set global logger for phase tracking."""
    global _global_logger
    _global_logger = logger


def begin_phase(phase: str):
    """Begin a phase (for external use)."""
    if _global_logger:
        _global_logger.begin_phase(phase)


def end_phase():
    """End current phase (for external use)."""
    if _global_logger:
        _global_logger.end_phase()


def log_samples(kept: int, total: int):
    """Log sample counts from dynamic filter."""
    if _global_logger:
        _global_logger.log_samples(kept, total)


def patch_trainer():
    """Patch GRPOTrainer to emit phase signals.""" 
    
    # Patch generation
    original_generate = GRPOTrainer._generate_and_score_completions
    
    def patched_generate(self, *args, **kwargs):
        begin_phase("generation")
        result = original_generate(self, *args, **kwargs)
        end_phase()
        return result
    
    GRPOTrainer._generate_and_score_completions = patched_generate
    
    # Patch training
    if hasattr(GRPOTrainer, 'compute_loss'):
        original_loss = GRPOTrainer.compute_loss
        
        def patched_loss(self, *args, **kwargs):
            begin_phase("training")
            result = original_loss(self, *args, **kwargs)
            end_phase()
            return result
            
        GRPOTrainer.compute_loss = patched_loss
    
    print("[UnifiedLogger] Patched GRPOTrainer for phase tracking")
    return True


if __name__ == "__main__":
    # Test
    logger = UnifiedLogger("./test_output")
    logger.start()
    
    for step in range(5):
        logger.set_step(step)
        
        logger.begin_phase("generation")
        time.sleep(0.2)
        logger.end_phase()
        
        logger.begin_phase("reward")
        time.sleep(0.05)
        logger.end_phase()
        
        logger.begin_phase("training")
        time.sleep(0.1)
        logger.end_phase()
        
        logger.log_step({
            "loss": 0.1 - step * 0.01,
            "kl": 0.001,
            "reward": 0.5 + step * 0.1,
        })
        
    logger.stop()
    logger.save_csv()
    logger.plot_all()
    logger.print_summary()
