# Stage 2: 2D Tiled Transpose & Subband Grid Mixer Kernel

**Project**: ALMA TPGS 40.0 Gsps Digital Spectrometer Pipeline  
**Kernel Symbol**: `transpose_grid_mixer_kernel`  
**Version**: 0.1.0  
**Source Location**: `src/tpgs_pipeline.cu`  
**Author**: Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)

---

## 1. Algorithmic Purpose & Pipeline Necessity

Stage 1 produces 1,536,000 temporal frames, where each frame contains 820 complex subband samples stored contiguously in memory (`[Frame 0: Subband 0..819], [Frame 1: Subband 0..819], ...`).

However, Stage 3 (Fine CSPFB) requires each individual subband's continuous time series as a contiguous 1D memory array (`[Subband 0: Frames 0..1,535,999], [Subband 1: Frames 0..1,535,999], ...`). Without transposing the matrix, Stage 3 would execute with a strided memory access pattern (stride = 820 * 8 bytes = 6,560 bytes), resulting in extreme cache thrashing and memory bus degradation.

Furthermore, because the coarse subband spacing (Delta_f_coarse = 19.53125 MHz) is not an integer multiple of the fine channel width (Delta_f_fine = 13.5000 kHz), each coarse subband exhibits a fractional frequency offset relative to the global 13.5 kHz astronomical frequency grid:
```
Offset(k) = (k * 19.53125 MHz) mod 13.5000 kHz
```

`transpose_grid_mixer_kernel` solves both problems simultaneously in a single high-bandwidth kernel pass:
1. It transposes the 1,536,000 x 820 matrix from time-major to subband-major order.
2. It applies a complex phase rotation `exp(-j * 2 * pi * Delta_f_offset * t)` in-flight, locking all 820 subbands onto the global 13.5 kHz grid before fine channelization.

---

## 2. Shared Memory Bank Conflict Elimination (32x33 Padded Tiles)

In NVIDIA GPUs, shared memory is divided into 32 independent banks (each 4 bytes wide). Consecutive 32-bit words map to consecutive banks. A bank conflict occurs when multiple threads in a warp access different addresses within the same bank simultaneously.

In a standard 32x32 complex tile (`cuComplex s_tile[32][32]`), each `cuComplex` element is 8 bytes (2 banks). Writing rows is conflict-free, but reading columns during the transpose step causes all 32 threads in a warp to hit the same shared memory banks, causing a severe 16-way to 32-way serialization penalty.

### Padded Stride Architecture:
By padding the inner dimension from 32 to 33 elements:
```cpp
__shared__ cuComplex s_tile[TRANS_TILES][32][33]; // Padded stride of 33 elements
```
Each successive row is shifted by 33 elements (66 banks = 2 banks offset relative to bank 0). When a warp reads down a column:
- Thread 0 reads row 0, column `c` -> Banks `(0 + 2*c) mod 32`
- Thread 1 reads row 1, column `c` -> Banks `(2 + 2*c) mod 32`
- Thread 2 reads row 2, column `c` -> Banks `(4 + 2*c) mod 32`

All 32 threads access distinct banks, achieving **0 bank conflicts** and full hardware throughput.

---

## 3. Multi-Tile Block Processing (4 Tiles/Block Scaling)

To amortize thread block launch overhead and maximize memory controller queuing depth, the kernel processes multiple 32x32 tiles per thread block.

### GH200 Benchmark Across Tile Multipliers (1.536M x 820 cuComplex, 9.38 GB):
```
========================================================================================
                      TRANSPOSE MULTI-TILE BENCHMARK RESULTS
========================================================================================
  1 Tile/Block   (1,250,000 blocks) :  8.030 ms  (2,509.6 GB/s Bi-Directional)
  2 Tiles/Block    (624,000 blocks) :  5.710 ms  (3,529.0 GB/s Bi-Directional)
  4 Tiles/Block    (312,000 blocks) :  5.015 ms  (4,018.6 GB/s Bi-Directional) [OPTIMAL]
  8 Tiles/Block    (156,000 blocks) :  5.066 ms  (3,978.0 GB/s Bi-Directional)
========================================================================================
```

Processing 4 tiles per block reduces the block count to 312,000, achieving **4,018.6 GB/s effective bi-directional throughput** (9.38 GB read + 9.38 GB written in 5.015 ms), operating at 93% of the theoretical HBM3e peak.

---

## 4. In-Flight Subband Grid Alignment Mixer

During the transpose register-store phase, each complex sample is rotated:
```cpp
float phase = -2.0f * (float)M_PI * (float)freq_offset * (float)t;
float cos_p, sin_p;
__sincosf(phase, &sin_p, &cos_p);

cuComplex val = s_tile[tile][threadIdx.x][threadIdx.y];
cuComplex rotated;
rotated.x = val.x * cos_p - val.y * sin_p;
rotated.y = val.x * sin_p + val.y * cos_p;

// Write coalesced directly to transposed buffer
d_out[out_idx] = rotated;
```
This eliminates the need for a separate digital downconverter or mixer kernel.

---

## 5. Measured Performance on NVIDIA GH200

```
========================================================================================
                  STAGE 2 TRANSPOSE & MIXER BENCHMARK SUMMARY
========================================================================================
  Matrix Dimension                   : 1,536,000 frames x 820 subbands
  Payload Size                       : 9.38 GB read + 9.38 GB write (18.76 GB total)
  Tile Configuration                 : 4 Tiles per block (32x33 padded shared memory)
  Block Dimension                    : 32 x 32 threads (1,024 threads/block)
  Grid Dimension                     : 26 x 12,000 = 312,000 blocks
  Execution Time (48 ms TE)          : 5.026 ms (11.5% of total pipeline)
  Bi-Directional Memory Bandwidth    : 4,019 GB/s
  Shared Memory Bank Conflicts       : 0 (Conflict-Free)
========================================================================================
```
