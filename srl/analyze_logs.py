#!/usr/bin/env python3
"""
Analyze TensorBoard logs and generate matplotlib plots with insights.
"""

import os
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

def load_tensorboard_logs(log_dir: str) -> dict:
    """Load all scalar metrics from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_time': [e.wall_time for e in events],
        }
    return metrics


def group_metrics(metrics: dict) -> dict:
    """Group metrics by category."""
    groups = defaultdict(dict)
    for tag, data in metrics.items():
        parts = tag.split('/')
        if len(parts) >= 2:
            category = parts[0]
            groups[category][tag] = data
        else:
            groups['other'][tag] = data
    return dict(groups)


def analyze_and_plot(log_dir: str, output_dir: str = "./analysis"):
    """Analyze logs and generate plots with insights."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading TensorBoard logs...")
    metrics = load_tensorboard_logs(log_dir)
    
    if not metrics:
        print("No metrics found in logs!")
        return
    
    print(f"Found {len(metrics)} metrics")
    groups = group_metrics(metrics)
    
    insights = []
    
    # ==================== TRAINING METRICS ====================
    if 'train' in groups:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')
        
        train_plots = [
            ('train/loss', 'Loss', 'tab:red'),
            ('train/avg_reward', 'Avg Reward', 'tab:green'),
            ('train/kl_divergence', 'KL Divergence', 'tab:blue'),
            ('train/kept_samples', 'Kept Samples', 'tab:purple'),
        ]
        
        for ax, (tag, title, color) in zip(axes.flat, train_plots):
            if tag in metrics:
                data = metrics[tag]
                ax.plot(data['steps'], data['values'], color=color, linewidth=2)
                ax.set_title(title)
                ax.set_xlabel('Step')
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3)
                
                # Add stats
                vals = data['values']
                if vals:
                    ax.axhline(np.mean(vals), color=color, linestyle='--', alpha=0.5, label=f'Mean: {np.mean(vals):.4f}')
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_metrics.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/training_metrics.png")
        
        # Training insights
        if 'train/loss' in metrics:
            loss_vals = metrics['train/loss']['values']
            if loss_vals:
                insights.append(f"📉 Loss: {loss_vals[0]:.4f} → {loss_vals[-1]:.4f} (Δ = {loss_vals[-1] - loss_vals[0]:.4f})")
                if loss_vals[-1] < loss_vals[0]:
                    insights.append("   ✅ Loss is decreasing - model is learning!")
                elif all(v == 0 for v in loss_vals):
                    insights.append("   ⚠️  Loss is always 0 - no gradient updates happening")
        
        if 'train/avg_reward' in metrics:
            reward_vals = metrics['train/avg_reward']['values']
            if reward_vals:
                insights.append(f"🏆 Avg Reward: {np.mean(reward_vals):.4f} (max: {max(reward_vals):.4f})")
                if max(reward_vals) > 0.8:
                    insights.append("   ✅ High rewards achieved!")
                elif max(reward_vals) < 0.3:
                    insights.append("   ⚠️  Low rewards - model struggling with task")
        
        if 'train/kept_samples' in metrics:
            kept_vals = metrics['train/kept_samples']['values']
            if kept_vals:
                total_kept = sum(kept_vals)
                insights.append(f"📊 Kept Samples: {total_kept} total across {len(kept_vals)} steps")
                if total_kept == 0:
                    insights.append("   🚨 No samples kept! Dynamic sampling filter is too aggressive")
    
    # ==================== RESOURCE USAGE ====================
    resource_tags = [t for t in metrics.keys() if 'resources/' in t]
    
    if resource_tags:
        # Phase durations
        duration_tags = [t for t in resource_tags if 'duration_sec' in t]
        if duration_tags:
            fig, ax = plt.subplots(figsize=(12, 6))
            for tag in duration_tags:
                phase = tag.split('/')[1].replace('_', ' ').title()
                data = metrics[tag]
                ax.bar(range(len(data['values'])), data['values'], label=phase, alpha=0.7)
            ax.set_title('Phase Durations per Step')
            ax.set_xlabel('Step')
            ax.set_ylabel('Duration (seconds)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/phase_durations.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/phase_durations.png")
            
            # Duration insights
            for tag in duration_tags:
                phase = tag.split('/')[1].replace('_', ' ').title()
                vals = metrics[tag]['values']
                if vals:
                    insights.append(f"⏱️  {phase}: avg {np.mean(vals):.2f}s, max {max(vals):.2f}s")
        
        # GPU/VRAM usage
        vram_tags = [t for t in resource_tags if 'vram_used_gb' in t and 'end' in t]
        gpu_tags = [t for t in resource_tags if 'gpu_utilization' in t and 'end' in t]
        
        if vram_tags or gpu_tags:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # VRAM plot
            if vram_tags:
                for tag in vram_tags[:5]:  # Limit to 5 for readability
                    phase = tag.split('/')[1].replace('_', ' ').title()
                    data = metrics[tag]
                    axes[0].plot(data['steps'], data['values'], label=phase, marker='o', markersize=3)
                axes[0].set_title('VRAM Usage by Phase')
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('VRAM (GB)')
                axes[0].legend(fontsize=8)
                axes[0].grid(True, alpha=0.3)
                axes[0].axhline(8.0, color='red', linestyle='--', alpha=0.7, label='8GB Limit')
            
            # GPU utilization plot
            if gpu_tags:
                for tag in gpu_tags[:5]:
                    phase = tag.split('/')[1].replace('_', ' ').title()
                    data = metrics[tag]
                    axes[1].plot(data['steps'], data['values'], label=phase, marker='o', markersize=3)
                axes[1].set_title('GPU Utilization by Phase')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('GPU %')
                axes[1].legend(fontsize=8)
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/gpu_vram_usage.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/gpu_vram_usage.png")
        
        # ==================== PER-PHASE RESOURCE PLOTS (3x2 Grid) ====================
        # Each row = phase, Left = Usage (VRAM/RAM), Right = Utilization (GPU%/CPU%)
        phases = ['generation', 'reward_calculation', 'training_(loss_+_backward)']
        phase_labels = ['Generation', 'Reward Calculation', 'Training (Loss + Backward)']
        
        # Check if we have data for these phases
        has_phase_data = any(
            any(phase in t for t in resource_tags) 
            for phase in phases
        )
        
        if has_phase_data:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle('Per-Phase Resource Analysis', fontsize=16, fontweight='bold', y=1.02)
            
            colors = {'vram': 'tab:blue', 'ram': 'tab:orange', 'gpu': 'tab:green', 'cpu': 'tab:red'}
            
            for row, (phase_key, phase_label) in enumerate(zip(phases, phase_labels)):
                # --- LEFT COLUMN: Usage (VRAM + RAM) ---
                ax_usage = axes[row, 0]
                
                # Find VRAM usage for this phase
                vram_tag = next((t for t in resource_tags if phase_key in t and 'vram_used_gb' in t and 'end' in t), None)
                ram_tag = next((t for t in resource_tags if phase_key in t and 'ram_used_gb' in t and 'end' in t), None)
                
                if vram_tag and vram_tag in metrics:
                    data = metrics[vram_tag]
                    ax_usage.plot(data['steps'], data['values'], color=colors['vram'], 
                                  marker='o', markersize=4, linewidth=2, label='VRAM (GB)')
                
                if ram_tag and ram_tag in metrics:
                    data = metrics[ram_tag]
                    # Create secondary y-axis for RAM
                    ax_ram = ax_usage.twinx()
                    ax_ram.plot(data['steps'], data['values'], color=colors['ram'], 
                                marker='s', markersize=4, linewidth=2, label='RAM (GB)', linestyle='--')
                    ax_ram.set_ylabel('RAM (GB)', color=colors['ram'])
                    ax_ram.tick_params(axis='y', labelcolor=colors['ram'])
                
                ax_usage.set_title(f'{phase_label} - Usage')
                ax_usage.set_xlabel('Step')
                ax_usage.set_ylabel('VRAM (GB)', color=colors['vram'])
                ax_usage.tick_params(axis='y', labelcolor=colors['vram'])
                ax_usage.grid(True, alpha=0.3)
                ax_usage.axhline(8.0, color='red', linestyle=':', alpha=0.5, linewidth=1)
                ax_usage.legend(loc='upper left', fontsize=8)
                
                # --- RIGHT COLUMN: Utilization (GPU% + CPU%) ---
                ax_util = axes[row, 1]
                
                # Find GPU and CPU utilization for this phase
                gpu_tag = next((t for t in resource_tags if phase_key in t and 'gpu_utilization' in t and 'end' in t), None)
                cpu_tag = next((t for t in resource_tags if phase_key in t and 'cpu_percent' in t and 'end' in t), None)
                
                if gpu_tag and gpu_tag in metrics:
                    data = metrics[gpu_tag]
                    ax_util.plot(data['steps'], data['values'], color=colors['gpu'], 
                                 marker='o', markersize=4, linewidth=2, label='GPU %')
                
                if cpu_tag and cpu_tag in metrics:
                    data = metrics[cpu_tag]
                    ax_util.plot(data['steps'], data['values'], color=colors['cpu'], 
                                 marker='s', markersize=4, linewidth=2, label='CPU %', linestyle='--')
                
                ax_util.set_title(f'{phase_label} - Utilization')
                ax_util.set_xlabel('Step')
                ax_util.set_ylabel('Utilization %')
                ax_util.set_ylim(0, 105)
                ax_util.grid(True, alpha=0.3)
                ax_util.axhline(50, color='orange', linestyle=':', alpha=0.4, linewidth=1)
                ax_util.axhline(80, color='green', linestyle=':', alpha=0.4, linewidth=1)
                ax_util.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/per_phase_resources.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/per_phase_resources.png")
            
            # VRAM insights
            for tag in vram_tags:
                phase = tag.split('/')[1].replace('_', ' ').title()
                vals = metrics[tag]['values']
                if vals:
                    max_vram = max(vals)
                    insights.append(f"💾 VRAM ({phase}): avg {np.mean(vals):.2f}GB, max {max_vram:.2f}GB")
                    if max_vram > 7.5:
                        insights.append(f"   ⚠️  Close to 8GB limit during {phase}!")
        
        # CPU usage during reward calculation
        cpu_tags = [t for t in resource_tags if 'cpu_percent' in t and 'reward' in t.lower()]
        if cpu_tags:
            for tag in cpu_tags:
                vals = metrics[tag]['values']
                if vals:
                    insights.append(f"🖥️  CPU (Reward Calc): avg {np.mean(vals):.1f}%, max {max(vals):.1f}%")
                    if np.mean(vals) < 30:
                        insights.append("   ⚠️  Low CPU usage - parallelism may not be working")
        
        # ==================== CPU UTILIZATION BY PHASE ====================
        cpu_phase_tags = [t for t in resource_tags if 'cpu_percent' in t and 'end' in t]
        if cpu_phase_tags:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for tag in cpu_phase_tags[:5]:  # Limit to 5 for readability
                phase = tag.split('/')[1].replace('_', ' ').title()
                data = metrics[tag]
                ax.plot(data['steps'], data['values'], label=phase, marker='o', markersize=3, linewidth=2)
            
            ax.set_title('CPU Utilization by Phase', fontsize=14, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('CPU %')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)  # 0-100% range
            
            # Add reference lines
            ax.axhline(50, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(80, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.5, 52, '50% (moderate)', fontsize=8, color='orange', alpha=0.7)
            ax.text(0.5, 82, '80% (good utilization)', fontsize=8, color='green', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cpu_utilization.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/cpu_utilization.png")
            
            # CPU insights per phase
            for tag in cpu_phase_tags:
                phase = tag.split('/')[1].replace('_', ' ').title()
                vals = metrics[tag]['values']
                if vals:
                    avg_cpu = np.mean(vals)
                    max_cpu = max(vals)
                    status = "✅" if avg_cpu > 50 else "⚠️" if avg_cpu > 20 else "🚨"
                    insights.append(f"🖥️  CPU ({phase}): avg {avg_cpu:.1f}%, max {max_cpu:.1f}% {status}")
        
        # RAM deltas
        ram_delta_tags = [t for t in resource_tags if 'ram_delta' in t]
        if ram_delta_tags:
            for tag in ram_delta_tags:
                phase = tag.split('/')[1].replace('_', ' ').title()
                vals = metrics[tag]['values']
                if vals:
                    avg_delta = np.mean(vals)
                    if abs(avg_delta) > 0.5:
                        insights.append(f"📈 RAM Δ ({phase}): avg {avg_delta:+.2f}GB per step")
    
    # ==================== PRINT INSIGHTS ====================
    print("\n" + "="*60)
    print("📊 TRAINING ANALYSIS INSIGHTS")
    print("="*60)
    
    for insight in insights:
        print(insight)
    
    print("\n" + "="*60)
    print(f"📁 Plots saved to: {output_dir}/")
    print("="*60)
    
    # Save insights to file
    with open(f'{output_dir}/insights.txt', 'w') as f:
        f.write("TRAINING ANALYSIS INSIGHTS\n")
        f.write("="*60 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")
    print(f"Saved: {output_dir}/insights.txt")
    
    return insights


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs"
    analyze_and_plot(log_dir)
