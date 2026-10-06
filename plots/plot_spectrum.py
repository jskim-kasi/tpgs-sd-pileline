#!/usr/bin/env python3
# =============================================================================
#  FILE        : plot_spectrum.py
#  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
#  DESCRIPTION : 16 GHz Contiguous Power Spectrum Plotter (Full & CW Tones Zoom)
#  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
#  VERSION     : 0.1.0
#  DATE        : October 2026
#  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
# =============================================================================

import struct
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_spectrum(bin_path):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")
    
    with open(bin_path, 'rb') as f:
        hdr = struct.unpack('4i', f.read(16))
        total_channels, n_subbands, keep_base, num_chunks = hdr
        print(f"Header: total_channels={total_channels}, n_subbands={n_subbands}, keep_base={keep_base}, chunks={num_chunks}")
        data = np.fromfile(f, dtype=np.float32, count=total_channels)
    
    return hdr, data

def get_tone_info(sb, local_ch):
    k = 102 + sb
    delta_f_coarse = 19531250.0
    delta_f_fine = 13500.0
    f0_hz = k * delta_f_coarse + (local_ch - 723) * delta_f_fine
    return f0_hz

def plot_full_spectrum(data, f_start_ghz, delta_f_khz, output_path):
    n_bins = len(data)
    delta_f_hz = delta_f_khz * 1e3
    
    # Calculate exact IF frequency for each channel across the 16 GHz band
    n_subbands = 820
    keep_base = 1446
    
    sb_indices = np.repeat(np.arange(n_subbands), keep_base)[:n_bins]
    local_indices = (np.arange(n_bins) % keep_base)
    k_coarse = 102 + sb_indices
    freqs_ghz = (k_coarse * 19531250.0 + (local_indices - 723) * delta_f_hz) / 1e9
    
    # Log power (dB)
    eps = 1e-12
    power_db = 10.0 * np.log10(np.maximum(data, eps))
    median_db = np.median(power_db)
    max_db = np.max(power_db)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), dpi=300, sharex=True)
    
    # Reference subbands: 10, 200, 410, 600, 810
    ref_subbands = [10, 200, 410, 600, 810]
    ref_tones = []
    for sb in ref_subbands:
        sb_slice = data[sb * keep_base : (sb + 1) * keep_base]
        peak_ch = int(np.argmax(sb_slice))
        f0_hz = get_tone_info(sb, peak_ch)
        ref_tones.append((sb, f0_hz / 1e9, power_db[sb * keep_base + peak_ch], data[sb * keep_base + peak_ch]))
    
    # ── Panel 1: Log Power (dB) ──
    ax1.plot(freqs_ghz, power_db, color='#1f77b4', linewidth=0.5, alpha=0.85, label='Stitched Auto-Correlation $|X(f)|^2$')
    ax1.axhline(median_db, color='#888888', linestyle='--', linewidth=1.2, label=f'Median Noise Floor ({median_db:.1f} dB)')
    
    for idx, (sb, f0_g, p_db, p_lin) in enumerate(ref_tones):
        ax1.axvline(f0_g, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.85)
        ax1.plot(f0_g, p_db, marker='*', markersize=9, color='#d62728')
        ax1.annotate(f'Tone {idx+1}\n{f0_g:.3f} GHz', xy=(f0_g, p_db), 
                     xytext=(f0_g - 0.25 if idx > 2 else f0_g + 0.15, p_db + 3.0),
                     arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                     fontsize=8.5, fontweight='bold', color='#d62728',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffdd', alpha=0.9, edgecolor='#d62728'))
    
    ax1.set_ylabel('Power Spectral Density (dB arb.)', fontsize=12, fontweight='bold')
    ax1.set_title('NVIDIA GH200 (SM 90a, 132 SMs): 16.0 GHz Seamless Stitched Spectrum (13.500 kHz / bin)', 
                  fontsize=13.5, fontweight='bold', pad=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim([median_db - 8.0, max_db + 8.0])
    
    # ── Panel 2: Linear Power ──
    ax2.plot(freqs_ghz, data, color='#2ca02c', linewidth=0.5, alpha=0.9, label='Linear Power ($|X|^2$)')
    for idx, (sb, f0_g, p_db, p_lin) in enumerate(ref_tones):
        ax2.axvline(f0_g, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.85)
        ax2.plot(f0_g, p_lin, marker='*', markersize=9, color='#d62728')
        ax2.annotate(f'Tone {idx+1}: {f0_g:.3f} GHz', xy=(f0_g, p_lin), 
                     xytext=(f0_g - 0.3 if idx > 2 else f0_g + 0.15, p_lin * 0.9),
                     arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                     fontsize=8.5, fontweight='bold', color='#b22222',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffdd', alpha=0.9, edgecolor='#b22222'))
    
    ax2.set_xlabel('IF Frequency (GHz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Linear Power ($|X|^2$)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim([freqs_ghz[0], freqs_ghz[-1]])
    ax2.set_ylim([0, np.max(data) * 1.15])
    ax2.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved full spectrum plot: {output_path}")

def plot_tone_zooms(data, f_start_ghz, delta_f_khz, output_path):
    keep_base = 1446
    delta_f_hz = delta_f_khz * 1e3
    
    ref_subbands = [10, 200, 410, 600, 810]
    
    fig, axes = plt.subplots(1, 5, figsize=(23, 5.5), dpi=300)
    
    for idx, sb in enumerate(ref_subbands):
        ax = axes[idx]
        sb_start = sb * keep_base
        sb_end = (sb + 1) * keep_base
        sb_data = data[sb_start : sb_end]
        
        # 1. Locate the true detected peak in this subband
        local_peak_ch = int(np.argmax(sb_data))
        peak_val = float(sb_data[local_peak_ch])
        noise_median = float(np.median(sb_data))
        snr_db = 10.0 * np.log10(peak_val / noise_median)
        
        # Exact tone IF frequency f0
        f0_hz = get_tone_info(sb, local_peak_ch)
        f0_ghz = f0_hz / 1e9
        
        # 2. Window of +/- 22 channels around the peak
        win = 22
        ch_min = max(0, local_peak_ch - win)
        ch_max = min(keep_base, local_peak_ch + win + 1)
        ch_slice = np.arange(ch_min, ch_max)
        
        slice_vals = sb_data[ch_slice]
        # Exact IF Frequency in GHz for each channel
        slice_freqs_ghz = ((102 + sb) * 19531250.0 + (ch_slice - 723) * delta_f_hz) / 1e9
        slice_db = 10.0 * np.log10(np.maximum(slice_vals, 1e-12))
        noise_floor_db = 10.0 * np.log10(noise_median)
        peak_db = np.max(slice_db)
        
        # 3. Plot spectrum line and discrete channel points
        ax.plot(slice_freqs_ghz, slice_db, 'o-', color='#1f77b4', markersize=3.5, linewidth=1.4, 
                label='PSD ($|X(f)|^2$)')
        
        # 4. Prominently indicate f0 location with vertical dashed line and star marker
        ax.axvline(f0_ghz, color='#d62728', linestyle='--', linewidth=1.8, 
                   label=f'$f_0 = {f0_ghz:.5f}$ GHz')
        ax.plot(f0_ghz, peak_db, marker='*', markersize=14, color='#d62728', zorder=6)
        
        # 5. Noise floor reference line
        ax.axhline(noise_floor_db, color='#7f7f7f', linestyle=':', linewidth=1.3, 
                   label=f'Noise Floor ({noise_floor_db:.1f} dB)')
        
        # 6. Dynamic vertical range tailored to this panel to clearly showcase the tone
        y_bottom = noise_floor_db - 4.0
        y_top = peak_db + 4.5
        ax.set_ylim([y_bottom, y_top])
        
        # 7. Callout annotation pointing directly to f0 peak
        if sb == 200:
            ax.annotate(f'$f_0$: {f0_ghz:.5f} GHz\nPeak: {peak_db:.1f} dB\nSNR: {snr_db:.1f} dB',
                        xy=(f0_ghz, peak_db),
                        xytext=(f0_ghz - 0.00014, peak_db - 3.0),
                        arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                        fontsize=8.0, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffdd', alpha=0.95, edgecolor='#d62728'))
        else:
            ax.annotate(f'$f_0$: {f0_ghz:.5f} GHz\nPeak: {peak_db:.1f} dB\nSNR: {snr_db:.1f} dB',
                        xy=(f0_ghz, peak_db),
                        xytext=(f0_ghz + 0.00004, peak_db - 0.25 * (peak_db - noise_floor_db)),
                        arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                        fontsize=8.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffdd', alpha=0.95, edgecolor='#d62728'))
        
        # Dynamic Spur Check for Tone 2 (Subband 200)
        if sb == 200:
            peak_slice_idx = int(np.argmax(slice_vals))
            mask = np.ones(len(slice_vals), dtype=bool)
            mask[max(0, peak_slice_idx - 1) : min(len(slice_vals), peak_slice_idx + 2)] = False
            if np.any(mask):
                non_carrier_vals = slice_vals[mask]
                sec_peak_idx_in_mask = int(np.argmax(non_carrier_vals))
                sec_val = float(non_carrier_vals[sec_peak_idx_in_mask])
                sec_db = 10.0 * np.log10(max(sec_val, 1e-12))
                sec_snr = sec_db - noise_floor_db
                true_sec_idx = np.where(mask)[0][sec_peak_idx_in_mask]
                sec_ghz = slice_freqs_ghz[true_sec_idx]

                if sec_snr >= 3.0:
                    ax.plot(sec_ghz, sec_db, marker='^', markersize=8, color='#ff7f0e', zorder=6)
                    ax.annotate(f'Residual Spur: {sec_db:.1f} dB\n(SNR: {sec_snr:.1f} dB)',
                                xy=(sec_ghz, sec_db),
                                xytext=(sec_ghz + 0.000045, sec_db + 2.5),
                                arrowprops=dict(facecolor='#ff7f0e', edgecolor='#ff7f0e', shrink=0.08, width=1.4, headwidth=5),
                                fontsize=8.0, fontweight='bold', color='#b25900',
                                bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff5eb', alpha=0.95, edgecolor='#ff7f0e'))
                else:
                    ax.text(0.04, 0.90, f'Spur Suppressed (<{noise_floor_db + 3.0:.1f} dB)\nBelow 3 dB SNR [PASS]',
                            transform=ax.transAxes, fontsize=7.8, fontweight='bold', color='#1b5e20',
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f5e9', alpha=0.95, edgecolor='#4caf50'))
        
        # 8. Titles and labels with exact units of GHz
        ax.set_title(f'Tone {idx+1}: Subband {sb}\n$f_0 = {f0_ghz:.6f}$ GHz | SNR = {snr_db:.1f} dB', 
                     fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xlabel('IF Frequency (GHz)', fontsize=10.5, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Power Spectral Density (dB arb.)', fontsize=10.5, fontweight='bold')
        
        # 9. Format bottom tick labels with full explicit numbers in GHz (no scientific offset)
        ax.set_xlim([slice_freqs_ghz[0], slice_freqs_ghz[-1]])
        
        # Generate 4 evenly spaced ticks in GHz
        t_min = slice_freqs_ghz[0]
        t_max = slice_freqs_ghz[-1]
        ticks = np.linspace(t_min + 0.00005, t_max - 0.00005, 4)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9.0)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=7.8, framealpha=0.9)
    
    plt.suptitle('NVIDIA GH200: CW Tones Point Spread Function & Spectral Purity (13.500 kHz / bin)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved tone zoom plot: {output_path}")

def main():
    bin_file = os.path.join(REPO_ROOT, 'data', 'power_spectrum_16ghz.bin')
    hdr, data = load_spectrum(bin_file)
    
    f_start_ghz = 2.000000000 # 2.0 GHz
    delta_f_khz = 13.50000000 # 13.5 kHz
    
    out_full = os.path.join(REPO_ROOT, 'plots', 'gh200_spectrum_16ghz_full.png')
    out_zoom = os.path.join(REPO_ROOT, 'plots', 'gh200_spectrum_cw_tones_zoom.png')
    
    plot_full_spectrum(data, f_start_ghz, delta_f_khz, out_full)
    plot_tone_zooms(data, f_start_ghz, delta_f_khz, out_zoom)

if __name__ == '__main__':
    main()
