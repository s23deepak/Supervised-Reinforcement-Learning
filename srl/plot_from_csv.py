#!/usr/bin/env python3
"""
Regenerate plots from existing training data (CSV files).

Usage:
    python plot_from_csv.py ./checkpoints_trained_srl
"""

import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_resource_samples(path):
    """Load resource_samples.csv"""
    samples = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'timestamp': float(row['timestamp']),
                'phase': row['phase'],
                'cpu_percent': float(row['cpu_percent']),
                'gpu_util_percent': float(row['gpu_util_percent']),
                'ram_used_gb': float(row['ram_used_gb']),
                'ram_total_gb': float(row['ram_total_gb']),
                'vram_used_gb': float(row['vram_used_gb']),
                'vram_total_gb': float(row['vram_total_gb']),
            })
    return samples


def plot_phase_breakdown(samples, output_path):
    """Plot per-phase resource usage - 3 rows (phases) x 2 cols (CPU, GPU)."""
    phases = ["generation", "reward", "training"]
    colors = {'generation': '#2ecc71', 'reward': '#3498db', 'training': '#e74c3c'}
    
    # Get samples for each phase
    phase_samples = {p: [] for p in phases}
    for s in samples:
        if s['phase'] in phases:
            phase_samples[s['phase']].append(s)
    
    # Skip if no phase data
    active_phases = [p for p in phases if phase_samples[p]]
    if not active_phases:
        print("No phase data found!")
        return
    
    fig, axes = plt.subplots(len(active_phases), 2, figsize=(14, 4 * len(active_phases)))
    fig.suptitle("Resource Usage by Phase", fontsize=14, fontweight='bold')
    
    # Handle single phase case
    if len(active_phases) == 1:
        axes = [axes]
    
    for row, phase in enumerate(active_phases):
        ps = phase_samples[phase]
        if not ps:
            continue
            
        # Get relative times
        start_time = ps[0]['timestamp']
        times = [(s['timestamp'] - start_time) for s in ps]
        
        color = colors[phase]
        
        # CPU subplot
        ax = axes[row][0]
        cpu_values = [s['cpu_percent'] for s in ps]
        ax.fill_between(times, cpu_values, alpha=0.3, color=color)
        ax.plot(times, cpu_values, color=color, linewidth=0.8)
        ax.set_ylabel("CPU %")
        ax.set_ylim(0, 100)
        ax.set_title(f"{phase.upper()} - CPU Utilization")
        ax.grid(True, alpha=0.3)
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        ax.axhline(y=avg_cpu, color='red', linestyle='--', alpha=0.7, 
                  label=f'Avg: {avg_cpu:.1f}%')
        ax.legend(loc='upper right')
        
        # GPU subplot
        ax = axes[row][1]
        gpu_values = [s['gpu_util_percent'] for s in ps]
        ax.fill_between(times, gpu_values, alpha=0.3, color=color)
        ax.plot(times, gpu_values, color=color, linewidth=0.8)
        ax.set_ylabel("GPU %")
        ax.set_ylim(0, 100)
        ax.set_title(f"{phase.upper()} - GPU Utilization")
        ax.grid(True, alpha=0.3)
        
        avg_gpu = sum(gpu_values) / len(gpu_values)
        ax.axhline(y=avg_gpu, color='red', linestyle='--', alpha=0.7,
                  label=f'Avg: {avg_gpu:.1f}%')
        ax.legend(loc='upper right')
        
        if row == len(active_phases) - 1:
            axes[row][0].set_xlabel("Time within phase (seconds)")
            axes[row][1].set_xlabel("Time within phase (seconds)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_from_csv.py <checkpoint_dir>")
        sys.exit(1)
        
    checkpoint_dir = sys.argv[1]
    csv_path = os.path.join(checkpoint_dir, "resource_samples.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        sys.exit(1)
    
    print(f"Loading data from: {csv_path}")
    samples = load_resource_samples(csv_path)
    print(f"Loaded {len(samples)} samples")
    
    # Create plots dir
    plots_dir = os.path.join(checkpoint_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plot
    output_path = os.path.join(plots_dir, "phase_breakdown.png")
    plot_phase_breakdown(samples, output_path)
    

if __name__ == "__main__":
    main()
