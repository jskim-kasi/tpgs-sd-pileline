# Stage 0: 6-Bit ADC Histogram Calculation Kernel

**Project**: ALMA TPGS 40.0 Gsps Digital Spectrometer Pipeline  
**Kernel Symbol**: `calc_histogram_6bit_kernel`  
**Version**: 0.1.0  
**Source Location**: `src/tpgs_pipeline.cu`  
**Author**: Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)

---

## 1. Scientific & Operational Purpose

In astronomical high-speed digital receivers, real-time statistical monitoring of the analog-to-digital converter (ADC) is required to ensure signal integrity:
1. **Dynamic Range Leveling**: Verifies that input Gaussian noise powers match the optimal quantization threshold (target standard deviation sigma = 8.0 for 6-bit quantization).
2. **Clipping & Saturation Detection**: Quantifies the fraction of samples clipped at the extreme quantization bins (-32 and +31), identifying non-linear saturation or radio-frequency interference (RFI).
3. **DC Offset Tracking**: Ensures zero mean across the sampled voltage distribution.

Stage 0 processes the complete 48.0 ms Timing Event epoch (1,920,000,000 samples / 1.44 GB) at the start of each observation cycle, extracting a 64-bin power distribution and verifying statistical health before spectroscopic channelization.

---

## 2. 6-Bit WSU Packing & Vectorized Memory Access

The ALMA WSU transmission format packs four signed 6-bit samples [-32, +31] into three consecutive bytes (24 bits) in little-endian byte order:

```
Byte 0: [S0(5:0)] | [S1(1:0) << 6]
Byte 1: [S1(5:2)] | [S2(3:0) << 4]
Byte 2: [S2(5:4)] | [S3(5:0) << 2]
```

Reading individual 3-byte chunks would result in misaligned 24-bit global memory transactions, dropping memory bus efficiency below 30%. 

To maximize HBM3e bus utilization, `calc_histogram_6bit_kernel` vectors memory access into 12-byte units (3 consecutive 32-bit words `w0`, `w1`, `w2`), containing exactly four 3-byte chunks (16 samples):

```
+-------------------------------------------------------------------------------+
|                       12-BYTE VECTORIZED UNPACKING SCHEME                     |
+-------------------------------------------------------------------------------+
  Word 0 (32-bit): Bytes [0, 1, 2, 3]
  Word 1 (32-bit): Bytes [4, 5, 6, 7]
  Word 2 (32-bit): Bytes [8, 9, 10, 11]

  Chunk 0 (Samples 0..3)   = w0 & 0x00FFFFFF
  Chunk 1 (Samples 4..7)   = (w0 >> 24) | ((w1 & 0x0000FFFF) << 8)
  Chunk 2 (Samples 8..11)  = (w1 >> 16) | ((w2 & 0x000000FF) << 16)
  Chunk 3 (Samples 12..15) = w2 >> 8
```

This mathematical transformation allows the GPU warp to load 3 clean cache-line-aligned transactions across 32 threads, achieving 100% memory bus coalescing.

---

## 3. Shared Memory Privatization & Bank-Conflict Elimination

A major bottleneck in GPU histogram kernels is atomic contention on shared memory bins. When all threads in a warp atomically update 64 bins (`s_hist[64]`), threads hitting the same bin are serialized by the shared memory controller.

To eliminate intra-warp serialization, the kernel allocates four interleaved privatized histogram banks in fast shared memory:

```cpp
__shared__ unsigned int s_hist[4][64]; // 256 uint32_t elements = 1,024 bytes smem
```

Each thread in a warp is assigned to one of the four banks based on its thread ID:
```cpp
int bank = threadIdx.x & 3; // Modulo-4 bank assignment
```

Each 6-bit sample is extracted and binned into the thread's assigned bank:
```cpp
#define HIST_ADD_CHUNK(c) do { \
    atomicAdd(&s_hist[bank][((c) >> 0)  & 0x3F], 1u); \
    atomicAdd(&s_hist[bank][((c) >> 6)  & 0x3F], 1u); \
    atomicAdd(&s_hist[bank][((c) >> 12) & 0x3F], 1u); \
    atomicAdd(&s_hist[bank][((c) >> 18) & 0x3F], 1u); \
} while(0)
```

Because adjacent threads in the warp write to different banks, shared memory bank conflicts are reduced by 4x, achieving near-theoretical execution throughput.

### Warp-Level Reduction:
Once the entire payload has been binned into shared memory, the first 64 threads sum the four privatized banks and issue a single atomic add to device memory:
```cpp
if (tid < 64) {
    unsigned int total_bin = s_hist[0][tid] + s_hist[1][tid] + s_hist[2][tid] + s_hist[3][tid];
    if (total_bin > 0) {
        atomicAdd(&d_hist[tid], (unsigned long long)total_bin);
    }
}
```

---

## 4. Measured Performance on NVIDIA GH200

```
========================================================================================
                      STAGE 0 ADC HISTOGRAM BENCHMARK RESULTS
========================================================================================
  Payload Size                       : 1,920,000,000 samples (1.44 GB / 1.34 GiB)
  Thread Block Dimensions            : 256 threads per block
  Grid Dimension                     : 528 blocks (132 SMs x 4 blocks/SM)
  Standalone Execution Time          : 0.828 ms
  In-Pipeline Execution Time         : 0.848 ms
  Sustained Memory Bandwidth         : 1,740.0 GB/s (80.5% of HBM3e 2,160 GB/s Peak)
  Shared Memory Allocation           : 1,024 bytes per thread block
  Active Occupancy                   : 100% active warps
========================================================================================
```

### Statistical Verification:
- **Measured Mean**: -0.0001 (Zero-mean verified)
- **Measured Sigma**: 8.8720 (Within linear operating range of 6-bit ADC)
- **Measured Clipping Fraction**: 4.86e-04 (< 0.05% of total samples)
- **Validation Status**: **PASS**
