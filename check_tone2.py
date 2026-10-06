#!/usr/bin/env python3
# =============================================================================
#  FILE        : check_tone2.py
#  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
#  DESCRIPTION : Standalone Tone Inspection Utility for Subband 200
#  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
#  VERSION     : 0.1.0
#  DATE        : October 2026
#  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
# =============================================================================

import os
import struct
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    bin_path = os.path.join(REPO_ROOT, "data", "power_spectrum_16ghz.bin")
    if not os.path.exists(bin_path):
        print(f"File not found: {bin_path}")
        return

    with open(bin_path, "rb") as f:
        hdr = struct.unpack("4i", f.read(16))
        total_channels, n_subbands, keep_base, num_chunks = hdr
        data = np.fromfile(f, dtype=np.float32)

    sb = 200
    sb_data = data[sb * keep_base : (sb + 1) * keep_base]
    k = 102 + sb
    f_sb_center = k * 19531250.0

    print(f"Subband {sb} center frequency: {f_sb_center / 1e9:.6f} GHz")

    # Find top peaks in this subband
    top_chs = np.argsort(sb_data)[::-1][:10]
    for rank, ch in enumerate(top_chs):
        val = sb_data[ch]
        p_db = 10.0 * np.log10(val)
        f_ch = (f_sb_center + (ch - 723) * 13500.0) / 1e9
        print(f"Rank {rank+1}: local_ch={ch:4d}, val={val:.4e}, p_db={p_db:6.2f} dB, freq={f_ch:.6f} GHz")

    # Inspect slice [894:906] around the peak
    print("\nChannels 894 to 906:")
    for ch in range(894, 906):
        val = sb_data[ch]
        p_db = 10.0 * np.log10(val)
        f_ch = (f_sb_center + (ch - 723) * 13500.0) / 1e9
        print(f"  ch={ch:4d}: val={val:.4e} ({p_db:6.2f} dB) | freq={f_ch:.6f} GHz")

if __name__ == "__main__":
    main()
