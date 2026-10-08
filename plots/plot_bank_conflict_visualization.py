#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def create_bank_conflict_visualization(output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    fig.patch.set_facecolor('#0d1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#30363d')


    # 1. Unpadded: tile[32][32]
    # Each element is 8 bytes (2 banks). Row stride is 32 * 2 = 64 bank words.
    # 64 % 32 = 0.
    unpadded_banks = np.zeros((16, 16), dtype=int)
    for r in range(16):
        for c in range(16):
            bank_word = (r * 32 * 2 + c * 2) % 32
            unpadded_banks[r, c] = bank_word

    # 2. Padded: tile[32][33]
    # Row stride is 33 * 2 = 66 bank words.
    # 66 % 32 = 2.
    padded_banks = np.zeros((16, 16), dtype=int)
    for r in range(16):
        for c in range(16):
            bank_word = (r * 33 * 2 + c * 2) % 32
            padded_banks[r, c] = bank_word

    # Plot Unpadded
    ax1.set_title("WITHOUT Padding: __shared__ cuComplex tile[32][32]\nCatastrophic 32-Way Bank Conflict (32x Serialization)",
                  color='#ff7b72', fontsize=12, fontweight='bold', pad=15)
    
    for r in range(16):
        for c in range(16):
            b = unpadded_banks[r, c]
            # Column 0 is read by threads 0..15 in warp
            is_col0 = (c == 0)
            color = '#da3633' if is_col0 else '#21262d'
            text_color = '#ffffff' if is_col0 else '#8b949e'
            edge_color = '#f85149' if is_col0 else '#30363d'
            linewidth = 2.0 if is_col0 else 0.5
            
            rect = Rectangle((c, 15 - r), 1, 1, facecolor=color, edgecolor=edge_color, linewidth=linewidth)
            ax1.add_patch(rect)
            ax1.text(c + 0.5, 15 - r + 0.5, f"B{b},{b+1}", ha='center', va='center',
                     color=text_color, fontsize=7, fontweight='bold' if is_col0 else 'normal')

    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 16)
    ax1.set_xlabel("Column Index (Elements)", color='#c9d1d9', fontsize=10, labelpad=8)
    ax1.set_ylabel("Row Index (Thread in Warp)", color='#c9d1d9', fontsize=10, labelpad=8)
    ax1.set_xticks(np.arange(16) + 0.5)
    ax1.set_xticklabels([f"C{i}" for i in range(16)], fontsize=8)
    ax1.set_yticks(np.arange(16) + 0.5)
    ax1.set_yticklabels([f"T{15-i}" for i in range(16)], fontsize=8)

    # Annotation box for unpadded
    ax1.text(8, -1.8, 
             "Warp reads Column 0: Threads T0..T15 all read Bank 0 and Bank 1!\n"
             "Stride = 32 elements * 8 bytes = 256 bytes = 64 banks = 0 (mod 32)\n"
             "Hardware must serialize reads: Takes 16 to 32 clock cycles!",
             ha='center', va='center', color='#ff7b72', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#da3633', linewidth=1.5))

    # Plot Padded
    ax2.set_title("WITH Stride-Skew Padding: __shared__ cuComplex tile[32][33]\nZero Bank Conflicts! (Single Clock Cycle Access)",
                  color='#3fb950', fontsize=12, fontweight='bold', pad=15)

    for r in range(16):
        for c in range(16):
            b = padded_banks[r, c]
            is_col0 = (c == 0)
            # Cycle distinct colors for column 0
            bank_color_cycle = plt.cm.tab20(r / 16.0)
            color = bank_color_cycle if is_col0 else '#21262d'
            text_color = '#ffffff' if is_col0 else '#8b949e'
            edge_color = '#2ea043' if is_col0 else '#30363d'
            linewidth = 2.0 if is_col0 else 0.5
            
            rect = Rectangle((c, 15 - r), 1, 1, facecolor=color, edgecolor=edge_color, linewidth=linewidth)
            ax2.add_patch(rect)
            ax2.text(c + 0.5, 15 - r + 0.5, f"B{b},{b+1}", ha='center', va='center',
                     color=text_color, fontsize=7, fontweight='bold')

    # Draw dummy element column indicator
    rect_pad = Rectangle((15.2, 0), 0.6, 16, facecolor='#238636', alpha=0.3, edgecolor='#2ea043', linestyle='--')
    ax2.add_patch(rect_pad)
    ax2.text(15.5, 8, "+1 Dummy Pad Element\n(8 Bytes / Row)", ha='center', va='center', rotation=90,
             color='#56d364', fontsize=9, fontweight='bold')

    ax2.set_xlim(0, 16.2)
    ax2.set_ylim(0, 16)
    ax2.set_xlabel("Column Index (Elements)", color='#c9d1d9', fontsize=10, labelpad=8)
    ax2.set_ylabel("Row Index (Thread in Warp)", color='#c9d1d9', fontsize=10, labelpad=8)
    ax2.set_xticks(np.arange(16) + 0.5)
    ax2.set_xticklabels([f"C{i}" for i in range(16)], fontsize=8)
    ax2.set_yticks(np.arange(16) + 0.5)
    ax2.set_yticklabels([f"T{15-i}" for i in range(16)], fontsize=8)

    # Annotation box for padded
    ax2.text(8, -1.8, 
             "Warp reads Column 0: Each thread accesses a DIFFERENT Bank Pair!\n"
             "Stride = 33 elements * 8 bytes = 264 bytes = 66 banks = +2 (mod 32)\n"
             "T0->(0,1), T1->(2,3), T2->(4,5) ... T15->(30,31) = ZERO CONFLICTS!",
             ha='center', va='center', color='#3fb950', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#2ea043', linewidth=1.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated bank conflict visualization: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "shared_memory_bank_conflict_viz.png")
    create_bank_conflict_visualization(out_file)
