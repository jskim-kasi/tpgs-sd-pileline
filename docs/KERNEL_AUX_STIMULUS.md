# Auxiliary Kernel: 6-Bit Multi-Tone Gaussian Stimulus Generator

**Project**: ALMA TPGS 40.0 Gsps Digital Spectrometer Pipeline  
**Kernel Symbol**: `gen_stimulus_6bit_kernel`  
**Version**: 0.1.0  
**Source Location**: `src/tpgs_pipeline.cu`  
**Author**: Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)

---

## 1. Purpose & Design

To validate the spectrometer pipeline under realistic astronomical observation conditions without requiring a physical 40 Gsps FPGA digitizer testbed, `gen_stimulus_6bit_kernel` synthesizes a bit-accurate ALMA WSU input stream directly in GPU global memory.

The generated stream includes:
1. **Gaussian White Noise**: Simulates astronomical thermal receiver noise with configurable standard deviation (sigma = 8.0, standard 6-bit quantization setting).
2. **Coherent CW Carrier Tones**: Injects precise continuous-wave (CW) carrier signals across selected subbands with known frequencies and amplitudes to verify channel mapping, passband flatness, and dynamic range.
3. **6-Bit WSU Format Packing**: Formats 4 signed 6-bit samples into 3 bytes with little-endian packing.

---

## 2. Kernel Architecture & Implementation Details

- **Random Number Generation**: Utilizes the GPU-native cuRAND Philox-4x32-10 PRNG algorithm (`curandStatePhilox4_32_10_t`). Each thread generates a `float4` vector containing 4 independent Gaussian random variables (`curand_normal4`).
- **CW Tone Superposition**:
  ```cpp
  double phase = 2.0 * M_PI * d_tones[t_idx].freq_hz * t;
  v += d_tones[t_idx].amp * (float)cos(phase);
  ```
- **Quantization & Clamping**: Clamps floating-point values to signed 6-bit integer limits `[-32, +31]`.
- **3-Byte Packing**: Packs 4 samples into 3 consecutive bytes via bit shifts and masks:
  ```cpp
  uint32_t chunk = (u0 & 0x3F) | ((u1 & 0x3F) << 6) | ((u2 & 0x3F) << 12) | ((u3 & 0x3F) << 18);
  dst[0] = (uint8_t)(chunk & 0xFF);
  dst[1] = (uint8_t)((chunk >> 8) & 0xFF);
  dst[2] = (uint8_t)((chunk >> 16) & 0xFF);
  ```

---

## 3. Operational Characteristics

```
========================================================================================
                    STIMULUS GENERATOR BENCHMARK METRICS
========================================================================================
  Generated Epoch Duration           : 48.0 ms
  Total Generated Samples            : 1,920,000,000 samples
  Output Memory Footprint            : 1.44 GB (1,440,000,000 bytes)
  Sampling Rate                      : 40.0 Gsps
  Synthesized Test Tones             : 820 CW carriers (1 tone per subband)
  Generation Throughput              : > 800 Gsps (> 20x real-time generation speed)
========================================================================================
```
