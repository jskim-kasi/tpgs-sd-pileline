# Stage 3: Fused Rational Resampler & Fine CSPFB Kernel

**Project**: ALMA TPGS 40.0 Gsps Digital Spectrometer Pipeline  
**Kernel Symbol**: `resample_cspfb_stitch_kernel`  
**Version**: 0.1.0  
**Source Location**: `src/tpgs_pipeline.cu`  
**Author**: Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)

---

## 1. Algorithmic Purpose & Pipeline Necessity

Stage 3 executes the high-resolution spectral channelization and final power spectrum synthesis for all 820 astronomical subbands.

### Multi-Rate Requirements:
1. **Rate Conversion**: Coarse subbands are output at 32.000 MSPS. To produce an exact 13.5000 kHz channel spacing with a native power-of-two 2,048-point FFT, the sampling rate must be rationally scaled by:
   ```
   Fs_fine = 32.000 MSPS * (108 / 125) = 27.648 MSPS
   Delta_f = 27.648 MHz / 2048 = 13.5000 kHz (EXACT)
   ```
2. **Fine Filtering**: A 10,240-tap prototype filter (5 taps/branch x 2048 channels) provides sharp bandpass response (> 70 dB out-of-band rejection) with minimal scallop loss.
3. **Power Spectrum Synthesis**: Computes auto-power detection (P = real^2 + imag^2) and stitches the flat center passband (1,445 channels per subband) into a contiguous 16 GHz spectrum (1,185,185 channels).

---

## 2. Kernel Fusion Architecture

Executing rational resampling, fine FIR filtering, cuFFTDx FFT, and spectrum stitching in separate kernels would require multiple intermediate device memory passes, exceeding the 48.0 ms real-time deadline.

`resample_cspfb_stitch_kernel` fuses all four operations into a single continuous compute pipeline per subband:

```
+-------------------------------------------------------------------------------+
|                       FUSED STAGE 3 FINE KERNEL PIPELINE                      |
+-------------------------------------------------------------------------------+
  Transposed Subband Time Series (820 Subbands x 1,536,000 Complex Samples)
       |
       v (Coalesced streaming read from d_trans_buf)
  [ Rational Polyphase Resampler (I=108, D=125, Kr=8) ]
       |  --> 864 Remez FIR coefficients in Constant/Shared Memory
       |  --> Outputs complex samples @ 27.648 MSPS into registers
       v
  [ Fine CSPFB FIR Polyphase Filter (10,240 Taps, K2=5, M=2048) ]
       |  --> In-register multiply-accumulate across 5 branches
       v
  [ NVIDIA cuFFTDx Size<2048> Complex-to-Complex FFT ]
       |  --> Native register-resident FFT execution
       v
  [ Auto-Power Detection & Spectrum Stitcher ]
       |  --> P = Real^2 + Imag^2
       v
  Contiguous 16 GHz Power Spectrum (1,185,185 Channels @ 13.5 kHz)
```

---

## 3. High-Performance Mathematical Optimizations

### 3.1 864-Tap Remez Equiripple Resampler Cache:
The rational resampler uses 108 polyphase phases with 8 taps per phase (864 floats = 3.456 KB). The entire filter is placed in GPU Constant Memory (`__constant__ float c_h_resamp[864]`), providing broadcast-speed access with zero DRAM memory traffic.

### 3.2 Fused Multiply-Add (FMAF) Math Hardware:
Polyphase resampling and FIR filtering utilize native hardware `fmaf()` instructions, executing two floating-point operations per clock cycle with full IEEE 754 precision.

### 3.3 Seamless Pre-Data State Management:
To guarantee continuous phase and boundary continuity across 48.0 ms Timing Events, the kernel maintains persistent history buffers (`d_fine_state` and `d_resamp_state`). The final samples of the previous epoch serve as the initial delay line of the subsequent epoch, preventing spectral leakage or Gibbs artifacts at event boundaries.

---

## 4. Single-Window Contiguous 16 GHz Spectrum Stitching

Each fine FFT yields 2,048 channels. Because coarse subbands are oversampled (OS = 1.6384), the edges contain transition band rolloff and aliasing.

The kernel extracts only the pristine flat center channels:
- **Base Channels per Subband**: N_FINE_KEEP_BASE = 1445 channels
- **Frequency Coverage per Subband**: 1445 * 13.5 kHz = 19.5075 MHz
- **Center Channel Offset**: Centered at bin 1024
- **Single 16 GHz SPW**:
  ```
  Total Stitched Channels = (820 subbands * 1445 channels) + 150 boundary channels
                          = 1,185,185 contiguous channels (0.0 to 16.0 GHz)
  ```

Power spectra are accumulated into single-precision floating-point arrays and written directly to `d_stitched_spw` (4.52 MB).

---

## 5. Measured Performance on NVIDIA GH200

```
========================================================================================
                      STAGE 3 FINE CSPFB BENCHMARK RESULTS
========================================================================================
  Input Payload                      : 820 Subbands x 1,536,000 samples @ 32 MSPS
  Output Resampled Payload           : 820 Subbands x 1,327,104 samples @ 27.648 MSPS
  Fine FFT Transform Size            : 2,048 Channels (Complex-to-Complex)
  Total Fine Frames per Subband      : 648 frames
  Total Fine Taps                    : 10,240 taps (K2=5, M=2048)
  Resampler Taps                     : 864 taps (108 phases x 8 taps)
  Block Dimension                    : 256 threads (cuFFTDx block_dim)
  Grid Dimension                     : (820, 4) = 3,280 blocks
  Execution Time (48 ms TE)          : 16.778 ms (38.4% of total pipeline)
  Final Spectrum Channels            : 1,185,185 contiguous channels
========================================================================================
```
