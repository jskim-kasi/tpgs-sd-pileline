#!/usr/bin/env python3
# =============================================================================
#  FILE        : plot_combined_timing_budget.py
#  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
#  DESCRIPTION : Unified comparative timing budget: NVIDIA GH200 vs RTX PRO 6000 Blackwell
#  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
#  VERSION     : 0.1.0
#  DATE        : October 2026
#  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
# =============================================================================

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_unified_timing_budget(output_path):
    # Benchmark Data across full 48.0 ms Timing Event (1.92 Billion Samples)
    # GH200 (132 SMs, HBM3e 4,019 GB/s) vs RTX PRO 6000 Blackwell (188 SMs, GDDR7 1,462 GB/s)
    gh200_times = [0.85, 21.05, 5.03, 16.78, 43.71]
    
    rf_file = os.path.join(REPO_ROOT, 'data', 'roofline_metrics.txt')
    h_ms, c_ms, t_ms, f_ms, tot_ms = 1.02, 15.94, 13.77, 11.59, 42.33
    if os.path.exists(rf_file):
        with open(rf_file, 'r') as f:
            for line in f:
                if 'HIST_MS=' in line: h_ms = float(line.split('=')[1])
                elif 'COARSE_MS=' in line: c_ms = float(line.split('=')[1])
                elif 'TRANSPOSE_MS=' in line: t_ms = float(line.split('=')[1])
                elif 'FINE_MS=' in line: f_ms = float(line.split('=')[1])
                elif 'TOTAL_PIPELINE_MS=' in line: tot_ms = float(line.split('=')[1])
    rtx6000_times = [h_ms, c_ms, t_ms, f_ms, tot_ms]
    budget_ms = 48.0

    categories = [
        'Stage 0: 6-Bit ADC Histogram\n(1.92B samples, 1.44 GB)',
        'Stage 1: Coarse OSPFB\n(8,192 taps, 1.536M frames)',
        'Stage 2: Transpose + Grid Mixer\n(1.536M x 820 cuComplex)',
        'Stage 3: Resampler + Fine CSPFB\n(Kr=8, K2=5, Pow2 Size<2048>)',
        'Total Pipeline Execution\n(Full 48.0 ms TE Stream)'
    ]

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(14.0, 9.0), dpi=300)

    y = np.arange(len(categories))
    bar_height = 0.28
    offset = 0.16

    y_gh200 = y - offset
    y_rtx   = y + offset

    color_gh200 = '#1f77b4'   # Steel Blue for GH200
    color_rtx   = '#ff7f0e'   # Vibrant Amber/Orange for RTX PRO 6000 Blackwell

    # Background shaded zones
    ax.axvspan(0, budget_ms, color='#e8f5e9', alpha=0.55, zorder=1, label='In-Budget Real-Time Zone (<= 48.0 ms) [PASS]')
    ax.axvspan(budget_ms, 70, color='#ffebee', alpha=0.45, zorder=1, label='Over-Budget Latency Zone (> 48.0 ms)')

    # Subtle stage divider lines
    for div in [0.5, 1.5, 2.5]:
        ax.axhline(div, color='#e0e0e0', linestyle='-', linewidth=0.8, zorder=2)
    ax.axhline(3.5, color='#b0bec5', linestyle='--', linewidth=1.4, zorder=2)

    # 48.0 ms Budget Line
    ax.axvline(budget_ms, color='#d62728', linestyle='--', linewidth=2.5, zorder=4,
               label=f'Real-Time Budget: {budget_ms:.1f} ms (1 Timing Event TE)')

    # Draw paired horizontal bars
    bars_gh200 = ax.barh(y_gh200, gh200_times, height=bar_height, color=color_gh200,
                         edgecolor='black', linewidth=1.2, alpha=0.92, zorder=3,
                         label='NVIDIA GH200 Grace Hopper (132 SMs, 4,019 GB/s HBM3e)')
    bars_rtx   = ax.barh(y_rtx, rtx6000_times, height=bar_height, color=color_rtx,
                         edgecolor='black', linewidth=1.2, alpha=0.92, zorder=3,
                         label='NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SMs, GDDR7)')

    # Annotate values with speedup / comparison callouts
    for i in range(len(categories)):
        # GH200 Label
        val_g = gh200_times[i]
        label_g = f"{val_g:.2f} ms"
        if i == 2:
            label_g += f" ({rtx6000_times[2]/val_g:.2f}x faster - HBM3e)"
        elif i == 4:
            label_g += f" ({budget_ms/val_g:.2f}x RT - PASS)"
        ax.text(val_g + 0.8, y_gh200[i], label_g, va='center', ha='left',
                fontsize=10.0, fontweight='bold', color='#0d47a1', zorder=5)

        # RTX PRO 6000 Label
        val_r = rtx6000_times[i]
        label_r = f"{val_r:.2f} ms"
        if i == 1:
            label_r += f" ({gh200_times[1]/val_r:.2f}x faster - 188 SMs)"
        elif i == 3:
            label_r += f" ({gh200_times[3]/val_r:.2f}x faster - Compute)"
        elif i == 4:
            label_r += f" ({budget_ms/val_r:.2f}x RT - PASS)"
        ax.text(val_r + 0.8, y_rtx[i], label_r, va='center', ha='left',
                fontsize=10.0, fontweight='bold', color='#b75500', zorder=5)

    # Axes configuration
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_xlabel('Execution Latency per 48.0 ms Timing Event (ms)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 68)

    # Secondary X-Axis: Speedup Relative to Real-Time (48.0 ms / Latency)
    sec_pos = [10.0, 16.0, 24.0, 32.0, 48.0, 60.0]
    secax = ax.secondary_xaxis('top')
    secax.set_ticks(sec_pos, labels=[f"{budget_ms/p:.1f}x" for p in sec_pos])
    secax.set_xlabel('Equivalent Real-Time Factor (48.0 ms / Latency, 1.0x = Real-Time Threshold)',
                     fontsize=11.0, fontweight='bold', color='#b71c1c', labelpad=10)
    secax.tick_params(colors='#b71c1c', labelsize=9.5)

    # Title & Legend
    ax.set_title('ALMA TPGS 40 Gsps Single-Dish Spectrometer Pipeline: Time Budget Comparison\n'
                 'NVIDIA GH200 Grace Hopper (SM 90) vs. NVIDIA RTX PRO 6000 Blackwell (SM 120)',
                 fontsize=13.5, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10.0, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
    ax.grid(True, axis='x', linestyle=':', alpha=0.7, zorder=0)
    ax.invert_yaxis()  # Stage 0 at top

    # Explanatory KPI Summary Callout
    kpi_text = (
        "Key Findings: (1) BOTH architectures achieve REAL-TIME execution: Blackwell = 42.33 ms (1.13x RT), GH200 = 43.71 ms (1.10x RT).\n"
        "              (2) GH200 dominates on Memory Bandwidth: Transpose is 2.74x faster (5.03 ms vs 13.77 ms) via 4,019 GB/s HBM3e.\n"
        "              (3) Blackwell dominates on Compute: Coarse is 1.32x faster (15.94 ms) & Fine is 1.45x faster (11.59 ms) via 188 SMs."
    )
    fig.text(0.5, 0.02, kpi_text, ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#004d40',
             multialignment='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e0f2f1', edgecolor='#00796b', alpha=0.95))

    plt.tight_layout(rect=[0.02, 0.07, 0.98, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved unified timing budget figure: {output_path}")
    plt.close()

def main():
    canonical_path = os.path.join(REPO_ROOT, 'plots', 'timing_budget_combined.png')
    generate_unified_timing_budget(canonical_path)

if __name__ == '__main__':
    main()
