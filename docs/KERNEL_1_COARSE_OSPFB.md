# Stage 1: Fused 6-Bit Unpack & Coarse OSPFB Kernel

**Project**: ALMA TPGS 40.0 Gsps Digital Spectrometer Pipeline  
**Kernel Symbol**: `unpack_coarse_ospfb_kernel`  
**Version**: 0.1.0  
**Source Location**: `src/tpgs_pipeline.cu`  
**Author**: Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)

---

## 1. Scientific & Mathematical Purpose

Stage 1 ingests the 40.0 Gsps real-valued time-domain data stream and performs coarse frequency channelization using an Overlap-Save Polyphase Filterbank (OSPFB). It extracts 820 contiguous subbands, each spanning 19.53125 MHz, covering the 0.0 to 16.0 GHz science bandwidth.

### Mathematical Invariants:
- **Sampling Frequency**: Fs = 40.0 GHz
- **FFT Transform Size**: M_C = 2048 channels
- **Decimation Stride**: D_C = 1250 samples
- **Oversampling Ratio**: OS = M_C / D_C = 2048 / 1250 = 1.6384 (63.84% overlap)
- **Coarse FIR Filter Order**: K1_C = 4 taps per branch (8,192 total taps)
- **Subband Output Sampling Rate**: Fs_coarse = Fs / D_C = 32.000 MSPS
- **Subband Frequency Spacing**: Delta_f_coarse = Fs / M_C = 19.53125 MHz
- **Processed Frames per 48 ms TE**: 1,536,000 frames

---

## 2. Kernel Fusion & Zero-Intermediate-Buffer Architecture

In conventional pipelines, 6-bit sample unpacking is executed as an isolated kernel writing 32-bit floats to global memory, followed by separate FIR convolution and FFT kernels. At 40.0 Gsps, writing unpacked 32-bit floats would generate 7.68 GB of traffic per 48 ms TE, saturating the GPU memory bus.

`unpack_coarse_ospfb_kernel` fuses all four operations into a single execution pass:
1. **Direct Global Memory Unpacking**: Threads unpack 6-bit signed integers directly into local registers using fast 32-bit arithmetic, requiring 0 bytes of shared memory for raw data.
2. **FIR Polyphase Convolution**: Multiplies the unpacked input samples by the 8,192-tap prototype low-pass filter across 4 overlapping branches.
3. **cuFFTDx Real-to-Complex Transform**: Executes an in-register 2,048-point R2C FFT using NVIDIA MathDX.
4. **Subband Decimation & Storage**: Extracts the 820 active positive-frequency subbands (discarding guard channels and negative frequencies) and writes complex output directly to `d_coarse_buf`.

```
+-------------------------------------------------------------------------------+
|                       FUSED STAGE 1 COARSE KERNEL PIPELINE                    |
+-------------------------------------------------------------------------------+
  Raw 6-Bit Stream (1.34 GB)
       |
       v (Coalesced 32-bit read)
  [ In-Register 6-Bit Unpack ] (Zero SMEM staging)
       |
       v (Multiply-accumulate with 8,192-tap FIR filter)
  [ Polyphase Summation Buffer ] (M = 2048 floats)
       |
       v (NVIDIA cuFFTDx Size<2048> Real-to-Complex)
  [ In-Register FFT Execution ]
       |
       v (Extract Subbands 0..819)
  Direct Global Memory Write to d_coarse_buf (1,536,000 frames x 820 subbands)
```

---

### 2.1 Dual-Frame Delay Line Sizing & Unpack Geometry

To process 2 coarse FFTs per block concurrently (`FFTsPerBlock<2>()`), the thread block maintains a shared delay line `s_unpacked` across consecutive frames `m0` (lane 0) and `m0 + 1` (lane 1):

1. **Required Delay Span**:
   - Lane 0 evaluates frame `m0` spanning samples `[0 .. 8,191]` (`N_TAPS_C = 8,192`).
   - Lane 1 evaluates frame `m0 + 1` starting at stride `D_C = 1,250`, spanning samples `[1,250 .. 9,441]`.
   - Total sample span needed across both lanes:
     `COARSE_DUAL_FRAME_SPAN = (FFTS_PER_BLOCK_C - 1) * D_C + N_TAPS_C = 1 * 1250 + 8192 = 9,442 samples`.

2. **Vectorized Thread Unpack**:
   - Each thread collaboratively unpacks four 24-bit chunks (12 bytes) into four `float4` vectors (16 floats per thread).
   - `COARSE_UNPACK_SAMPLES_PER_TH = 16`.
   - `COARSE_UNPACK_BYTES_PER_TH = 12`.

3. **Active Unpack Threads**:
   - `COARSE_UNPACK_THREADS = ceil(9,442 / 16) = (9442 + 16 - 1) / 16 = 592 threads`.
   - Threads `flat_tid < 592` execute collaborative unpacking directly into `s_unpacked`.

4. **Shared Memory Buffer Allocation**:
   - `COARSE_UNPACK_SMEM_FLOATS = COARSE_UNPACK_THREADS * COARSE_UNPACK_SAMPLES_PER_TH = 592 * 16 = 9,472 floats`.
   - `COARSE_UNPACK_SMEM_BYTES = 9,472 * 4 = 37,888 bytes`.
   - Because 37,888 bytes is a multiple of 16 bytes, it guarantees strict 128-bit alignment for `s_filter` which is placed immediately after `s_unpacked` in dynamic shared memory.

---

## 3. Coarse Filter Coefficient Caching Modes

The kernel provides three compile-time template modes for storing the 8,192 FIR prototype filter coefficients:

- **Mode 0 (Registers)**: Coefficients are loaded into per-thread registers. High register pressure limits block occupancy.
- **Mode 1 (Dynamic Shared Memory - OPTIMAL)**: All 8,192 coefficients (32 KB) are loaded collaboratively into shared memory at block startup. Threads execute FIR accumulation with zero global memory latency and zero L1 cache contention.
- **Mode 2 (L1 / Constant Cache)**: Coefficients are stored in global memory and accessed via the GPU L1 texture/constant cache.

### Performance Comparison on GH200 (1,536,000 Frames / 48 ms TE):
- **Mode 0 (Registers)**: 22.469 ms
- **Mode 1 (Shared Memory)**: **20.998 ms** (1.47 ms faster, Optimal)
- **Mode 2 (L1 Cache)**: 22.692 ms

---

## 4. Latency Hiding & Tail-Wave Elimination

To maximize GPU hardware utilization across different architectures (GH200 with 132 SMs, Blackwell RTX 6000 with 188 SMs), the grid dimension is dynamically scaled:
```cpp
int n_blocks_coarse = num_sms * 6; // 792 blocks on GH200, 1128 blocks on RTX 6000
```

Launching exactly 6 blocks per SM ensures that:
1. Each SM maintains 2 active blocks concurrently.
2. Three complete waves of thread blocks execute sequentially across the 1.536M frame grid.
3. Tail-wave imbalances (where SMs idle waiting for the last wave to finish) are eliminated.

---

## 5. Measured Performance on NVIDIA GH200

```
========================================================================================
                      STAGE 1 COARSE OSPFB BENCHMARK RESULTS
========================================================================================
  Input Payload                      : 1,920,000,000 samples (1.44 GB 6-bit packed)
  Total Output Frames                : 1,536,000 frames x 820 subbands (9.38 GB)
  Filter Length                      : 8,192 taps (M=2048, K1=4)
  Decimation Stride                  : 1,250 samples (OS = 1.6384)
  Selected Architecture Mode         : Mode 1 (Shared Memory Coefficients)
  Block Dimension                    : 256 threads (cuFFTDx block_dim)
  Grid Dimension                     : 792 blocks (132 SMs x 6 blocks/SM)
  Execution Time (48 ms TE)          : 21.049 ms (48.2% of total pipeline)
  Active Occupancy                   : 2 blocks per SM
  cuFFTDx Shared Memory Size         : 9,472 bytes (FFT) + 32,768 bytes (FIR)
========================================================================================
```
