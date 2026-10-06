#!/usr/bin/env python3
# =============================================================================
#  FILE        : plot_combined_timing_budget.py
#  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
#  DESCRIPTION : Unified comparative timing budget: NVIDIA GH200 vs RTX PRO 6000
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
    # Benchmark Data (Full 48.0 ms Timing Event TE)
    gh200_times = [25.120, 9.112, 36.128, 70.369]
    rtx6000_times = [26.130, 43.753, 20.793, 90.680]
    budget_ms = 48.0

    categories = [
        'Stage 1: Coarse OSPFB\n(8,192 taps, 1.5M frames)',
        'Stage 2: Transpose + Mixer\n(1.5M x 820 cuComplex)',
        'Stage 3: Resamp + Fine CSPFB\n(Kr=8, Bluestein 2000)',
        'Total Pipeline Execution\n(Full 48.0 ms TE Stream)'
    ]

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)

    y = np.arange(len(categories))
    bar_height = 0.30
    offset = 0.17

    y_gh200 = y - offset
    y_rtx   = y + offset

    color_gh200 = '#1f77b4'   # Steel Blue for GH200
    color_rtx   = '#ff7f0e'   # Vibrant Amber/Orange for RTX PRO 6000 Blackwell

    # Background shaded zones
    ax.axvspan(0, budget_ms, color='#e8f5e9', alpha=0.55, zorder=1, label='In-Budget Real-Time Zone (<= 48.0 ms)')
    ax.axvspan(budget_ms, 130, color='#ffebee', alpha=0.45, zorder=1, label='Over-Budget Latency Zone (> 48.0 ms)')

    # Subtle stage divider lines
    ax.axhline(0.5, color='#e0e0e0', linestyle='-', linewidth=0.8, zorder=2)
    ax.axhline(1.5, color='#e0e0e0', linestyle='-', linewidth=0.8, zorder=2)
    ax.axhline(2.5, color='#b0bec5', linestyle='--', linewidth=1.2, zorder=2)

    # 48.0 ms Budget Line
    ax.axvline(budget_ms, color='#d62728', linestyle='--', linewidth=2.5, zorder=4,
               label=f'Real-Time Budget: {budget_ms:.1f} ms (1 Timing Event TE)')

    # Draw paired horizontal bars
    bars_gh200 = ax.barh(y_gh200, gh200_times, height=bar_height, color=color_gh200,
                         edgecolor='black', linewidth=1.2, alpha=0.92, zorder=3,
                         label='NVIDIA Grace Hopper GH200 (SM 90, 480 GB LPDDR5X + 96 GB HBM3)')
    bars_rtx   = ax.barh(y_rtx, rtx6000_times, height=bar_height, color=color_rtx,
                         edgecolor='black', linewidth=1.2, alpha=0.92, zorder=3,
                         label='NVIDIA RTX PRO 6000 Blackwell (SM 120, PCIe Gen5)')

    # Annotate values with speedup / comparison callouts
    for i in range(len(categories)):
        # GH200 Label
        val_g = gh200_times[i]
        label_g = f"{val_g:.2f} ms"
        if i == 1:
            label_g += " (4.80x faster)"
        elif i == 3:
            label_g += f" (0.68x RT)"
        ax.text(val_g + 1.2, y_gh200[i], label_g, va='center', ha='left',
                fontsize=10.5, fontweight='bold', color='#0d47a1', zorder=5)

        # RTX PRO 6000 Label
        val_r = rtx6000_times[i]
        label_r = f"{val_r:.2f} ms"
        if i == 2:
            label_r += " (1.74x faster)"
        elif i == 3:
            label_r += f" (0.53x RT)"
        ax.text(val_r + 1.2, y_rtx[i], label_r, va='center', ha='left',
                fontsize=10.5, fontweight='bold', color='#b75500', zorder=5)

    # Axes configuration
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_xlabel('Execution Latency per 48.0 ms Timing Event (ms)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 125)

    # Secondary X-Axis: Speedup Relative to Real-Time (48.0 ms / Latency)
    secax = ax.secondary_xaxis('top', functions=(
        lambda x: np.where(x >= 0.5, budget_ms / np.clip(x, 0.5, 125.0), 96.0),
        lambda r: np.where(r >= 0.38, budget_ms / np.clip(r, 0.38, 96.0), 125.0)
    ))
    secax.set_xlabel('Equivalent Real-Time Speedup Factor (1.0x = Real-Time Threshold)',
                     fontsize=11.5, fontweight='bold', color='#b71c1c', labelpad=10)
    secax.tick_params(colors='#b71c1c', labelsize=10)

    # Title & Legend
    ax.set_title('ALMA TPGS 40 Gsps Single-Dish Spectrometer Pipeline: Time Budget Comparison\n'
                 'NVIDIA GH200 Grace Hopper vs. NVIDIA RTX PRO 6000 Blackwell',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10.2, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
    ax.grid(True, axis='x', linestyle=':', alpha=0.7, zorder=0)
    ax.invert_yaxis()  # Stage 1 at top

    # Explanatory KPI Summary Callout
    kpi_text = (
        "Key Takeaways:  (1) Transpose is 4.80x faster on GH200 due to massive coherent memory bandwidth.\n"
        "                (2) Bluestein Fine FFT is 1.74x faster on Blackwell due to enhanced FP32 ALUs & cache.\n"
        "                (3) Pipelining 16 ms chunks across streams satisfies real-time budget on both architectures."
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
