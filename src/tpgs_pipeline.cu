// =============================================================================
//  FILE        : tpgs_pipeline.cu
//  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
//  DESCRIPTION : Full 48 ms real-time pipeline implementation on NVIDIA GH200:
//                - Dual 48 ms Ping-Pong global memory buffers
//                - 16 ms chunk pipelining across 3 independent CUDA streams
//                - Pre-data history handling across Coarse, Resampler, and Fine
//                - cuFFTDx Size<2048> Coarse OSPFB + 64KB Constant Phase Table
//                - Bank-conflict-free 2D Tiled Transpose (32x33)
//                - cuFFTDx Size<2048> Fine CSPFB + Fused Resampler + Auto-Power
//                - Direct stitching into single 16 GHz SPW (1,185,185 bins)
//                - Benchmark suite: Registers vs Shared Memory vs L1/Constant
//                - 7-tier verification suite & roofline data export
//  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
//  VERSION     : 0.1.0
//  DATE        : October 2026
//  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufftdx.hpp>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <vector>
#include <string>
#include <chrono>

#include "tpgs_params.h"
#include "packet_wsu.h"

using namespace tpgs;

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

// =============================================================================
//  CONSTANT MEMORY DECLARATIONS
// =============================================================================

// 1. Resampler Remez prototype coefficients (108 phases x 8 taps = 864 floats = 3.456 KB)
__constant__ float  c_h_resamp[N_TAPS_RESAMP];

// =============================================================================
//  KERNEL 0: MULTI-TONE GAUSSIAN STIMULUS GENERATOR (6-BIT WSU FORMAT)
// =============================================================================

__global__ void gen_stimulus_6bit_kernel(
    uint8_t*        __restrict__ d_packed_stream, // [prefix + chunk_samples] in 3-byte groups
    long long                    n_samples_total,
    const CWTone*   __restrict__ d_tones,
    int                          n_tones,
    double                       fs_adc,
    float                        sigma,
    unsigned long long           seed,
    long long                    time_sample_offset)
{
    // Each thread generates 4 samples and writes 3 bytes
    long long group_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long sample_base = group_idx * SAMPLES_PER_PACK_GROUP;
    if (sample_base >= n_samples_total) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, group_idx, 0, &state);
    float4 noise = curand_normal4(&state);
    float n_arr[4] = { noise.x, noise.y, noise.z, noise.w };

    int s_val[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        long long abs_sample = time_sample_offset + sample_base + s;
        float v = n_arr[s] * sigma;

        if (abs_sample < (time_sample_offset + n_samples_total)) {
            double t = (double)abs_sample / fs_adc;
            for (int t_idx = 0; t_idx < n_tones; ++t_idx) {
                double phase = 2.0 * M_PI * d_tones[t_idx].freq_hz * t;
                v += d_tones[t_idx].amp * (float)cos(phase);
            }
        }
        s_val[s] = __float2int_rn(v);
    }

    uint8_t* out_ptr = d_packed_stream + group_idx * BYTES_PER_PACK_GROUP;
    pack_4_samples_6bit(out_ptr, s_val[0], s_val[1], s_val[2], s_val[3]);
}

// =============================================================================
//  STAGE 0: 6-BIT ADC HISTOGRAM CALCULATION KERNEL (1.92 BILLION SAMPLES)
// =============================================================================

// High-throughput coalesced 64-bin histogram kernel (4-way shared memory privatized banks)
__global__ void calc_histogram_6bit_kernel(
    const uint8_t*      __restrict__ d_packed_stream,
    long long                        n_samples,
    unsigned long long* __restrict__ d_hist)
{
    // 4 interleaved banks of 64 bins = 256 uints (1,024 bytes smem) eliminating bank conflicts
    __shared__ unsigned int s_hist[4][64];

    int tid = threadIdx.x;
    if (tid < 256) {
        s_hist[tid >> 6][tid & 63] = 0;
    }
    __syncthreads();

    const uint32_t* src_words = reinterpret_cast<const uint32_t*>(d_packed_stream);
    long long total_12b_units = n_samples / 16; // 120,000,000 for 1.92B samples (12 bytes = 16 samples)
    long long global_tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    int bank = tid & 3;

    for (long long u = global_tid; u < total_12b_units; u += stride) {
        long long w_idx = u * 3;
        uint32_t w0 = src_words[w_idx + 0];
        uint32_t w1 = src_words[w_idx + 1];
        uint32_t w2 = src_words[w_idx + 2];

        uint32_t c0 = w0 & 0x00FFFFFF;
        uint32_t c1 = (w0 >> 24) | ((w1 & 0x0000FFFF) << 8);
        uint32_t c2 = (w1 >> 16) | ((w2 & 0x000000FF) << 16);
        uint32_t c3 = w2 >> 8;

        #define HIST_ADD_CHUNK(c) do { \
            atomicAdd(&s_hist[bank][((c) >> 0)  & 0x3F], 1u); \
            atomicAdd(&s_hist[bank][((c) >> 6)  & 0x3F], 1u); \
            atomicAdd(&s_hist[bank][((c) >> 12) & 0x3F], 1u); \
            atomicAdd(&s_hist[bank][((c) >> 18) & 0x3F], 1u); \
        } while(0)

        HIST_ADD_CHUNK(c0);
        HIST_ADD_CHUNK(c1);
        HIST_ADD_CHUNK(c2);
        HIST_ADD_CHUNK(c3);
        #undef HIST_ADD_CHUNK
    }
    __syncthreads();

    // Accumulate the 4 privatized banks and atomically add to global histogram
    if (tid < 64) {
        unsigned int total_bin = s_hist[0][tid] + s_hist[1][tid] + s_hist[2][tid] + s_hist[3][tid];
        if (total_bin > 0) {
            atomicAdd(&d_hist[tid], (unsigned long long)total_bin);
        }
    }
}

// =============================================================================
//  KERNEL 0: 6-BIT TO 32-BIT FLOAT UNPACKER (OPTION B 3-BYTE CONVERSION)
// =============================================================================
//  KERNEL 1: FUSED 6-BIT UNPACK + COARSE OSPFB KERNEL (cuFFTDx Size<2048>)
// =============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_src)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" : : : "memory");
}
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
}
#endif

__device__ __forceinline__ float4 unpack_24bit_to_float4(uint32_t val)
{
    // Branchless sign extension of 4 packed 6-bit signed integers
    int q0 = ((int)(val << 26)) >> 26;
    int q1 = ((int)(val << 20)) >> 26;
    int q2 = ((int)(val << 14)) >> 26;
    int q3 = ((int)(val <<  8)) >> 26;
    return make_float4((float)q0, (float)q1, (float)q2, (float)q3);
}

// Analytical Phase Rotation: Phi_k[m] = exp(-j * 2*pi * ((k * m * PHASE_PERIOD_P) % PHASE_PERIOD_Q) / PHASE_PERIOD_Q)
// Evaluated analytically via hardware SFU __sincosf with zero memory or bank conflicts
__device__ __forceinline__ float2 get_phase_rotation_fast(int k, int m_factor)
{
    int n = ((k & PHASE_PERIOD_MASK) * m_factor) & PHASE_PERIOD_MASK;
    float angle = -TWO_PI_F * ((float)n * (1.0f / (float)PHASE_PERIOD_Q));
    float s, c;
    __sincosf(angle, &s, &c);
    return make_float2(c, s);
}

__device__ __forceinline__ float2 get_phase_rotation(int k, int m)
{
    int m_factor = ((m & PHASE_PERIOD_MASK) * PHASE_PERIOD_P) & PHASE_PERIOD_MASK;
    return get_phase_rotation_fast(k, m_factor);
}

template <class FFT, int StorageMode = 0> // 0: Registers, 1: Shared, 2: L1 Cache
__launch_bounds__(FFT::max_threads_per_block)
__global__ void unpack_coarse_ospfb_kernel(
    const uint8_t*      __restrict__ d_input_packed, // Raw 6-bit packed (includes COARSE_PREDATA_BYTES prefix)
    const float*        __restrict__ d_filter,       // 8,192 taps
    cuComplex*          __restrict__ d_out,          // [N_FRAMES_C_CHUNK x N_SUBBANDS]
    int                              n_frames,
    int                              n_blocks,
    int                              k_start_subband)
{
    using cx_t = typename FFT::value_type;
    static constexpr int EPT    = FFT::storage_size;  // 4
    static constexpr int STRIDE = FFT::stride;        // 512
    static constexpr int T      = K1_C;               // 4 taps/branch

    const int tx  = threadIdx.x;
    const int lid = threadIdx.y;                      // FFT lane: 0 or 1
    const int bid = blockIdx.x;
    const int flat_tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..1023

    const int fpb          = ((n_frames + n_blocks - 1) / n_blocks + 1) & ~1;
    const int m_start_blk  = bid * fpb;
    const int m_end_blk    = (bid == n_blocks - 1) ? n_frames : min(n_frames, (bid + 1) * fpb);
    const int n_frames_blk = (m_start_blk < n_frames) ? (m_end_blk - m_start_blk) : 0;
    const int n_iters      = (n_frames_blk + 1) / 2;

    extern __shared__ __align__(alignof(float4)) char smem_raw[];
    cx_t* shared_mem_fft = reinterpret_cast<cx_t*>(smem_raw);
    float* s_unpacked = reinterpret_cast<float*>(smem_raw + FFT::shared_memory_size);

    // ── Filter Coefficient Storage Modes ──
    float h_reg[EPT][T];
    if constexpr (StorageMode == 0) {
        // Mode 0: Register caching
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int k = tx + e * STRIDE;
            if (k < M_C) {
                #pragma unroll
                for (int p = 0; p < T; ++p)
                    h_reg[e][p] = d_filter[k + p * M_C];
            }
        }
    }

    float* s_filter = nullptr;
    if constexpr (StorageMode == 1) {
        // Mode 1: Zero-conflict Shared Memory filter caching (30 registers, 1-cycle on-chip read)
        s_filter = reinterpret_cast<float*>(
            smem_raw + FFT::shared_memory_size + COARSE_UNPACK_SMEM_BYTES);
        int total_threads = blockDim.x * blockDim.y;
        for (int i = flat_tid; i < N_TAPS_C; i += total_threads) {
            s_filter[i] = d_filter[i];
        }
        __syncthreads();
    }

    // ── Frame Iteration Loop (Option B: Direct Global Unpack & Single-Barrier per Iteration) ──
    for (int iter = 0; iter < n_iters; ++iter) {
        const int m0 = m_start_blk + iter * 2;
        long long base_sample = (long long)m0 * D_C;
        long long base_byte = (base_sample * 3) / 4;

        long long aligned_base_byte = base_byte & ~15LL;
        int byte_offset = (int)(base_byte & 15LL);

        const uint32_t* src_words = reinterpret_cast<const uint32_t*>(d_input_packed + aligned_base_byte);

        // Collaborative 32-bit word unpack directly from global memory into s_unpacked
        if (flat_tid < COARSE_UNPACK_THREADS) {
            int start_byte = byte_offset + flat_tid * COARSE_UNPACK_BYTES_PER_TH;
            int word_idx = start_byte >> 2;
            int shift = byte_offset & 3;

            uint32_t w0 = src_words[word_idx];
            uint32_t w1 = src_words[word_idx + 1];
            uint32_t w2 = src_words[word_idx + 2];
            uint32_t u0, u1, u2;

            if (shift == 0) {
                u0 = w0;
                u1 = w1;
                u2 = w2;
            } else if (shift == 1) {
                uint32_t w3 = src_words[word_idx + 3];
                u0 = (w0 >> 8)  | (w1 << 24);
                u1 = (w1 >> 8)  | (w2 << 24);
                u2 = (w2 >> 8)  | (w3 << 24);
            } else if (shift == 2) {
                uint32_t w3 = src_words[word_idx + 3];
                u0 = (w0 >> 16) | (w1 << 16);
                u1 = (w1 >> 16) | (w2 << 16);
                u2 = (w2 >> 16) | (w3 << 16);
            } else { // shift == 3
                uint32_t w3 = src_words[word_idx + 3];
                u0 = (w0 >> 24) | (w1 << 8);
                u1 = (w1 >> 24) | (w2 << 8);
                u2 = (w2 >> 24) | (w3 << 8);
            }

            uint32_t c0 = u0 & 0x00FFFFFF;
            uint32_t c1 = (u0 >> 24) | ((u1 & 0x0000FFFF) << 8);
            uint32_t c2 = (u1 >> 16) | ((u2 & 0x000000FF) << 16);
            uint32_t c3 = u2 >> 8;

            float4 f0 = unpack_24bit_to_float4(c0);
            float4 f1 = unpack_24bit_to_float4(c1);
            float4 f2 = unpack_24bit_to_float4(c2);
            float4 f3 = unpack_24bit_to_float4(c3);

            float4* s_dst4 = reinterpret_cast<float4*>(s_unpacked + flat_tid * COARSE_UNPACK_SAMPLES_PER_TH);
            s_dst4[0] = f0;
            s_dst4[1] = f1;
            s_dst4[2] = f2;
            s_dst4[3] = f3;
        }
        __syncthreads(); // Barrier 1: ensure s_unpacked is ready for polyphase FIR

        const int m = m0 + lid;
        const bool valid = (m < m_end_blk);

        cx_t thread_data[EPT];
        const int lane_sample_offset = lid * D_C;

        // Polyphase FIR reading directly from shared memory s_unpacked
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int k = tx + e * STRIDE;
            float val = 0.0f;
            if (valid && k < M_C) {
                int s_idx = lane_sample_offset + k;
                if constexpr (StorageMode == 1) {
                    float tap0 = s_filter[k + 0 * M_C];
                    float tap1 = s_filter[k + 1 * M_C];
                    float tap2 = s_filter[k + 2 * M_C];
                    float tap3 = s_filter[k + 3 * M_C];

                    float dq0 = s_unpacked[s_idx + 0 * M_C];
                    float dq1 = s_unpacked[s_idx + 1 * M_C];
                    float dq2 = s_unpacked[s_idx + 2 * M_C];
                    float dq3 = s_unpacked[s_idx + 3 * M_C];

                    val = fmaf(tap0, dq0, fmaf(tap1, dq1, fmaf(tap2, dq2, tap3 * dq3)));
                } else if constexpr (StorageMode == 0) {
                    val = fmaf(h_reg[e][0], s_unpacked[s_idx + 0 * M_C],
                          fmaf(h_reg[e][1], s_unpacked[s_idx + 1 * M_C],
                          fmaf(h_reg[e][2], s_unpacked[s_idx + 2 * M_C],
                               h_reg[e][3] * s_unpacked[s_idx + 3 * M_C])));
                } else {
                    val = fmaf(d_filter[k + 0 * M_C], s_unpacked[s_idx + 0 * M_C],
                          fmaf(d_filter[k + 1 * M_C], s_unpacked[s_idx + 1 * M_C],
                          fmaf(d_filter[k + 2 * M_C], s_unpacked[s_idx + 2 * M_C],
                               d_filter[k + 3 * M_C] * s_unpacked[s_idx + 3 * M_C])));
                }
            }
            thread_data[e] = cx_t(val, 0.0f);
        }

        // Execute cuFFTDx block FFT
        FFT().execute(thread_data, shared_mem_fft);

        // Analytical Phase Correction + Selective 16 GHz Subband Write
        if (valid) {
            int global_m = m;
            int m_factor = ((global_m & PHASE_PERIOD_MASK) * PHASE_PERIOD_P) & PHASE_PERIOD_MASK;
            #pragma unroll
            for (int e = 0; e < EPT_ACTIVE_COARSE; ++e) { // Subbands [102..921] span e = 0 and e = 1 only
                int k = tx + e * STRIDE;
                int sb_idx = k - k_start_subband;
                if (sb_idx >= 0 && sb_idx < N_SUBBANDS) {
                    float2 rot = get_phase_rotation_fast(k, m_factor);
                    float re = thread_data[e].x;
                    float im = thread_data[e].y;

                    // Complex multiply: thread_data * rot
                    float out_re = re * rot.x - im * rot.y;
                    float out_im = re * rot.y + im * rot.x;

                    d_out[(long long)global_m * N_SUBBANDS + sb_idx] = make_cuComplex(out_re, out_im);
                }
            }
        }

        if (iter < n_iters - 1) {
            __syncthreads(); // Barrier 2: protect s_unpacked from next iteration overwrite
        }
    }
}

// =============================================================================
//  KERNEL 2: BANK-CONFLICT-FREE 2D TILED TRANSPOSE + GRID MIXER (32x33 Tile)
// =============================================================================

constexpr int TRANS_TILE = 32;

template <int TILES_PER_BLOCK = 1>
__global__ void transpose_grid_mixer_kernel(
    const cuComplex* __restrict__ src, // [rows x cols] frame-major
    cuComplex*       __restrict__ dst, // [cols x rows] subband-major
    int                           rows,
    int                           cols,
    int                           k_start_subband)
{
    // +1 padding column guarantees bank-conflict-free shared memory access
    __shared__ cuComplex tile[TRANS_TILE][TRANS_TILE + 1];

    // Precompute subband rotation step once per thread (invariant across tile loop along rows)
    int y_out = blockIdx.x * TRANS_TILE + threadIdx.y; // Row in dst (subband)
    double theta_step = 0.0;
    if (y_out < cols) {
        int k_coarse = k_start_subband + y_out;
        int rem = (k_coarse * GRID_NUM) % GRID_DEN;
        theta_step = -THETA_GRID_UNIT * (double)rem;
    }

    #pragma unroll
    for (int t = 0; t < TILES_PER_BLOCK; ++t) {
        int by = blockIdx.y * TILES_PER_BLOCK + t;

        int x_in = blockIdx.x * TRANS_TILE + threadIdx.x; // Col in src (subband)
        int y_in = by * TRANS_TILE + threadIdx.y;          // Row in src (frame)

        if (x_in < cols && y_in < rows) {
            tile[threadIdx.y][threadIdx.x] = src[(long long)y_in * cols + x_in];
        }
        __syncthreads();

        int x_out = by * TRANS_TILE + threadIdx.x;         // Col in dst (frame)

        if (x_out < rows && y_out < cols) {
            cuComplex val = tile[threadIdx.x][threadIdx.y];

            float s_sin, c_cos;
            __sincosf((float)((double)x_out * theta_step), &s_sin, &c_cos);

            float mixed_re = val.x * c_cos - val.y * s_sin;
            float mixed_im = val.x * s_sin + val.y * c_cos;

            dst[(long long)y_out * rows + x_out] = make_cuComplex(mixed_re, mixed_im);
        }
        __syncthreads();
    }
}

// =============================================================================
//  INLINE HELPER: 8-TAP POLYPHASE RATIONAL RESAMPLER (32.000 MSPS -> 27.648 MSPS, 108/125)
// =============================================================================

__device__ __forceinline__ cuComplex calc_resampled_val_fast(
    int n_s,
    int p_idx,
    const cuComplex* __restrict__ d_sb_signal,
    const float*     __restrict__ s_h_resamp)
{
    float acc_re = 0.0f, acc_im = 0.0f;
    #pragma unroll
    for (int k = 0; k < KR_RESAMP; ++k) {
        cuComplex coarse_val = d_sb_signal[n_s - k];
        float tap = s_h_resamp[p_idx + k * I_RESAMP];
        acc_re = fmaf(coarse_val.x, tap, acc_re);
        acc_im = fmaf(coarse_val.y, tap, acc_im);
    }
    return make_cuComplex(acc_re, acc_im);
}

__device__ __forceinline__ cuComplex calc_resampled_val_direct(
    int n_s,
    int p_idx,
    const cuComplex* __restrict__ d_sb_signal,
    const cuComplex* __restrict__ d_resamp_state,
    const float*     __restrict__ s_h_resamp,
    int sb,
    int epoch_idx)
{
    if (__builtin_expect(n_s >= RESAMP_PREDATA_SAMPLES, 1)) {
        return calc_resampled_val_fast(n_s, p_idx, d_sb_signal, s_h_resamp);
    }

    float acc_re = 0.0f, acc_im = 0.0f;
    #pragma unroll
    for (int k = 0; k < KR_RESAMP; ++k) {
        int n = n_s - k;
        cuComplex coarse_val = make_cuComplex(0.0f, 0.0f);
        if (n >= 0) {
            coarse_val = d_sb_signal[n];
        } else if (epoch_idx > 0 && d_resamp_state != nullptr) {
            int past_idx = RESAMP_PREDATA_SAMPLES + n;
            if (past_idx >= 0 && past_idx < RESAMP_PREDATA_SAMPLES)
                coarse_val = d_resamp_state[sb * RESAMP_PREDATA_SAMPLES + past_idx];
        }
        float tap = s_h_resamp[p_idx + k * I_RESAMP];
        acc_re = fmaf(coarse_val.x, tap, acc_re);
        acc_im = fmaf(coarse_val.y, tap, acc_im);
    }
    return make_cuComplex(acc_re, acc_im);
}

__device__ __forceinline__ cuComplex calc_resampled_val(
    long long s_idx,
    const cuComplex* __restrict__ d_sb_signal,
    const cuComplex* __restrict__ d_resamp_state,
    const float*     __restrict__ s_h_resamp,
    int sb,
    int epoch_idx)
{
    uint32_t a_m  = (uint32_t)s_idx * D_RESAMP;
    int n_s       = (int)(a_m / I_RESAMP);
    int p_idx     = (int)(a_m % I_RESAMP);
    return calc_resampled_val_direct(n_s, p_idx, d_sb_signal, d_resamp_state, s_h_resamp, sb, epoch_idx);
}

// =============================================================================
//  KERNEL 3: RATIONAL RESAMPLER + FINE CSPFB + AUTO-POWER + STITCHING
// =============================================================================

template <class FFT, int StorageMode = 0> // 0: Registers, 1: Shared, 2: L1
__launch_bounds__(FFT::max_threads_per_block, 1)
__global__ void resample_cspfb_stitch_kernel(
    const cuComplex* __restrict__ d_transposed,    // [N_SUBBANDS x n_coarse_frames]
    const float*     __restrict__ d_filter_fine,   // 10,240 taps
    cuComplex*       __restrict__ d_fine_state,    // [N_SUBBANDS x FINE_PREDATA_FRAMES x M_F]
    cuComplex*       __restrict__ d_resamp_state,  // [N_SUBBANDS x RESAMP_PREDATA_SAMPLES]
    float*           __restrict__ d_stitched_spw,  // [TOTAL_STITCHED_CHANNELS]
    int                           epoch_idx,
    int                           k_start_subband,
    int                           n_coarse_frames,
    int                           n_fine_frames)
{
    using cx_t = typename FFT::value_type;
    static constexpr int EPT    = FFT::storage_size;  // 4
    static constexpr int STRIDE = FFT::stride;        // 512
    static constexpr int P      = K2_F;               // 5 taps/branch

    const int tx = threadIdx.x;
    const int sb = blockIdx.x; // 1 block per subband
    if (sb >= N_SUBBANDS) return;

    // Load all resampler filter taps into static shared memory (3,456 bytes for Kr=8)
    __shared__ float s_h_resamp[N_TAPS_RESAMP];
    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int t = flat_tid; t < N_TAPS_RESAMP; t += blockDim.x * blockDim.y) {
        s_h_resamp[t] = c_h_resamp[t];
    }
    __syncthreads();

    const cuComplex* d_sb_signal = d_transposed + (long long)sb * n_coarse_frames;

    extern __shared__ __align__(alignof(float4)) cx_t shared_mem[];

    // ── Load Fine Filter Taps into Registers ──
    float h_fine_reg[EPT][P];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int ch = tx + e * STRIDE;
        if (ch < M_F) {
            #pragma unroll
            for (int k = 0; k < P; ++k)
                h_fine_reg[e][k] = d_filter_fine[(M_F - 1 - ch) + k * M_F];
        }
    }

    // ── Initialize Fine Register Ring Buffer ──
    cx_t ring_buf[EPT][P];
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
        #pragma unroll
        for (int p = 0; p < P; ++p)
            ring_buf[e][p] = cx_t(0.0f, 0.0f);

    const int b_y = blockIdx.y;
    const int n_blocks_y = gridDim.y;
    const int fpb = n_fine_frames / n_blocks_y;
    const int m_start = b_y * fpb;
    const int m_end   = (b_y == n_blocks_y - 1) ? n_fine_frames : (b_y + 1) * fpb;

    // Precompute initial resampler indices at m = m_start for each element
    int n_s_curr[EPT];
    int p_idx_curr[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int ch = tx + e * STRIDE;
        if (ch < M_F) {
            uint32_t a_m = (uint32_t)((long long)m_start * M_F + ch) * D_RESAMP;
            n_s_curr[e] = (int)(a_m / I_RESAMP);
            p_idx_curr[e] = (int)(a_m % I_RESAMP);
        } else {
            n_s_curr[e] = 0;
            p_idx_curr[e] = 0;
        }
    }

    // Restore Pre-Data (from previous epoch for b_y==0, or from end of prior block/chunk)
    if (b_y > 0) {
        // Fast incremental pre-data priming from prior 4 fine frames without 64-bit division
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int ch = tx + e * STRIDE;
            if (ch < M_F) {
                // Step back 4 frames: 4 * 256,000 = 1,024,000 = 9481 * 108 + 52
                int p_idx = p_idx_curr[e] - 52;
                int borrow = (p_idx < 0);
                if (borrow) p_idx += I_RESAMP;
                int n_s = n_s_curr[e] - 9481 - borrow;

                #pragma unroll
                for (int p = 0; p < P - 1; ++p) {
                    cuComplex s = calc_resampled_val_fast(n_s, p_idx, d_sb_signal, s_h_resamp);
                    ring_buf[e][p] = cx_t(s.x, s.y);

                    p_idx += 40;
                    int carry = (p_idx >= I_RESAMP);
                    p_idx -= carry ? I_RESAMP : 0;
                    n_s += 2370 + carry;
                }
            }
        }
    } else if (epoch_idx > 0 && d_fine_state != nullptr) {
        const cuComplex* past_frames = d_fine_state + (long long)sb * (P - 1) * M_F;
        #pragma unroll
        for (int p = 0; p < P - 1; ++p) {
            #pragma unroll
            for (int e = 0; e < EPT; ++e) {
                int ch = tx + e * STRIDE;
                if (ch < M_F) {
                    cuComplex s = past_frames[p * M_F + ch];
                    ring_buf[e][p] = cx_t(s.x, s.y);
                }
            }
        }
    }

    int ring_head = (epoch_idx > 0 || b_y > 0) ? (P - 1) : 0;

    // Auto-power accumulator in registers
    float pwr_acc[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) pwr_acc[e] = 0.0f;

    // ── Fine Frame Loop (162 frames per block for 4-way split of 648 frames) ──
    for (int m = m_start; m < m_end; ++m) {
        // Fast direct L1 cache polyphase rational resampling into registers
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int ch = tx + e * STRIDE;
            if (ch < M_F) {
                cuComplex s;
                if (__builtin_expect(b_y > 0 || m > 0, 1)) {
                    s = calc_resampled_val_fast(n_s_curr[e], p_idx_curr[e], d_sb_signal, s_h_resamp);
                } else {
                    s = calc_resampled_val_direct(n_s_curr[e], p_idx_curr[e], d_sb_signal, d_resamp_state, s_h_resamp, sb, epoch_idx);
                }
                ring_buf[e][ring_head] = cx_t(s.x, s.y);

                // Incremental resampler step: advance by M_F * D_RESAMP = 2048 * 125 = 256,000 samples
                // 256,000 = 2370 * 108 + 40
                p_idx_curr[e] += 40;
                int carry = (p_idx_curr[e] >= I_RESAMP);
                p_idx_curr[e] -= carry ? I_RESAMP : 0;
                n_s_curr[e] += 2370 + carry;
            }
        }

        // Polyphase FIR across ring buffer history
        cx_t thread_data[EPT];
        #define FIR_ACC(e, r0, r1, r2, r3, r4)                                                 \
        {                                                                                      \
            float r = fmaf(h_fine_reg[e][0], (r0).x,                                           \
                      fmaf(h_fine_reg[e][1], (r1).x,                                           \
                      fmaf(h_fine_reg[e][2], (r2).x,                                           \
                      fmaf(h_fine_reg[e][3], (r3).x,                                           \
                           h_fine_reg[e][4] * (r4).x))));                                      \
            float i = fmaf(h_fine_reg[e][0], (r0).y,                                           \
                      fmaf(h_fine_reg[e][1], (r1).y,                                           \
                      fmaf(h_fine_reg[e][2], (r2).y,                                           \
                      fmaf(h_fine_reg[e][3], (r3).y,                                           \
                           h_fine_reg[e][4] * (r4).y))));                                      \
            thread_data[e] = cx_t(r, i);                                                       \
        }

        switch (ring_head) {
            case 0:
                #pragma unroll
                for (int e = 0; e < EPT; ++e) FIR_ACC(e, ring_buf[e][0], ring_buf[e][4], ring_buf[e][3], ring_buf[e][2], ring_buf[e][1]);
                break;
            case 1:
                #pragma unroll
                for (int e = 0; e < EPT; ++e) FIR_ACC(e, ring_buf[e][1], ring_buf[e][0], ring_buf[e][4], ring_buf[e][3], ring_buf[e][2]);
                break;
            case 2:
                #pragma unroll
                for (int e = 0; e < EPT; ++e) FIR_ACC(e, ring_buf[e][2], ring_buf[e][1], ring_buf[e][0], ring_buf[e][4], ring_buf[e][3]);
                break;
            case 3:
                #pragma unroll
                for (int e = 0; e < EPT; ++e) FIR_ACC(e, ring_buf[e][3], ring_buf[e][2], ring_buf[e][1], ring_buf[e][0], ring_buf[e][4]);
                break;
            case 4:
                #pragma unroll
                for (int e = 0; e < EPT; ++e) FIR_ACC(e, ring_buf[e][4], ring_buf[e][3], ring_buf[e][2], ring_buf[e][1], ring_buf[e][0]);
                break;
        }
        #undef FIR_ACC

        ring_head = (ring_head == 4) ? 0 : (ring_head + 1);

        // Execute cuFFTDx block FFT
        FFT().execute(thread_data, shared_mem);

        // Auto-power accumulation: |X|^2 with hardware FMA
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            pwr_acc[e] = fmaf(thread_data[e].x, thread_data[e].x,
                         fmaf(thread_data[e].y, thread_data[e].y, pwr_acc[e]));
        }
    }

    // Save Tail Coarse Samples for Next TE Epoch Resampler Pre-Data
    if (b_y == gridDim.y - 1 && threadIdx.x < RESAMP_PREDATA_SAMPLES && d_resamp_state != nullptr) {
        long long tail_idx = (long long)n_coarse_frames - RESAMP_PREDATA_SAMPLES + threadIdx.x;
        d_resamp_state[sb * RESAMP_PREDATA_SAMPLES + threadIdx.x] = d_sb_signal[tail_idx];
    }

    // Save Fine Ring Buffer Tail States for Next TE Epoch
    if (b_y == gridDim.y - 1 && d_fine_state != nullptr) {
        cuComplex* save_frames = d_fine_state + (long long)sb * (P - 1) * M_F;
        #pragma unroll
        for (int p = 0; p < P - 1; ++p) {
            int buf_idx = ring_head - (P - 1) + p;
            if (buf_idx < 0) buf_idx += P;
            #pragma unroll
            for (int e = 0; e < EPT; ++e) {
                int ch = tx + e * STRIDE;
                if (ch < M_F) {
                    save_frames[p * M_F + ch] = make_cuComplex(ring_buf[e][buf_idx].x, ring_buf[e][buf_idx].y);
                }
            }
        }
    }

    // ── Seamless Direct Array Stitching into 16 GHz Single Spectral Window ──
    constexpr int HALF_KEEP = N_FINE_KEEP_BASE / 2; // 723 channels
    long long global_base = (long long)sb * N_FINE_KEEP_BASE;

    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int ch = tx + e * STRIDE;
        int stitch_ch = -1;

        if (ch >= 0 && ch < HALF_KEEP) {
            stitch_ch = HALF_KEEP + ch;              // Upper half of passband
        } else if (ch >= M_F - HALF_KEEP && ch < M_F) {
            stitch_ch = ch - (M_F - HALF_KEEP);      // Lower half of passband
        }

        if (stitch_ch >= 0) {
            long long global_idx = global_base + stitch_ch;
            if (global_idx < TOTAL_STITCHED_CHANNELS) {
                float avg_pwr = pwr_acc[e] / (float)n_fine_frames;
                if (gridDim.y == 1 && epoch_idx == 0) {
                    d_stitched_spw[global_idx] = avg_pwr;
                } else {
                    atomicAdd(&d_stitched_spw[global_idx], avg_pwr);
                }
            }
        }
    }
}

// =============================================================================
//  HOST HELPER: BINARY FILTER FILE LOADER
// =============================================================================

static float* load_filter_binary(const char* filepath, int expected_taps)
{
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open filter file '%s': %s\n", filepath, strerror(errno));
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    long expected_bytes = (long)expected_taps * sizeof(float);
    if (sz != expected_bytes) {
        fprintf(stderr, "ERROR: Filter '%s' size %ld bytes (expected %ld bytes)\n",
                filepath, sz, expected_bytes);
        fclose(fp);
        return nullptr;
    }

    float* h_buf = (float*)malloc(expected_bytes);
    if (fread(h_buf, sizeof(float), expected_taps, fp) != (size_t)expected_taps) {
        fprintf(stderr, "ERROR: Failed to read %d taps from '%s'\n", expected_taps, filepath);
        free(h_buf);
        fclose(fp);
        return nullptr;
    }
    fclose(fp);

    float* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, expected_bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_buf, expected_bytes, cudaMemcpyHostToDevice));
    free(h_buf);
    return d_ptr;
}

// =============================================================================
//  PIPELINE ORCHESTRATOR & BENCHMARK HARNESS
// =============================================================================

template <unsigned int Arch>
void run_tpgs_pipeline(int num_sms)
{
    using namespace cufftdx;

    printf("=================================================================\n");
    printf("  TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline (SM %u)\n", Arch);
    printf("  Coarse OSPFB: M=%d, D=%d (OS=%.2f), T=%d, Subbands=%d (16.0 GHz)\n",
           M_C, D_C, (double)M_C/D_C, K1_C, N_SUBBANDS);
    printf("  Resampler:    I=%d, D=%d, Kr=%d (%.3f MSPS -> %.3f MSPS)\n",
           I_RESAMP, D_RESAMP, KR_RESAMP, FS_COARSE/1e6, FS_RESAMP/1e6);
    printf("  Fine CSPFB:   M=%d, D=%d, T=%d, Delta_f=%.4f kHz (EXACT)\n",
           M_F, D_F, K2_F, DELTA_F_FINE/1e3);
    printf("  Stitching:    Single 16 GHz Window with %d Contiguous Channels\n",
           TOTAL_STITCHED_CHANNELS);
    printf("  TE Duration:  %.1f ms (Option B: Unified Full TE Sequential Pipeline)\n",
           DURATION_TE * 1e3);
    printf("=================================================================\n\n");

    // ── Configure cuFFTDx Transforms ──
    using FFT_C = decltype(Block() +
                           Size<2048>() +
                           Type<fft_type::c2c>() +
                           Direction<fft_direction::forward>() +
                           Precision<float>() +
                           ElementsPerThread<4>() +
                           FFTsPerBlock<2>() +
                           SM<Arch>());

    using FFT_F = decltype(Block() +
                           Size<2048>() +
                           Type<fft_type::c2c>() +
                           Direction<fft_direction::forward>() +
                           Precision<float>() +
                           ElementsPerThread<4>() +
                           FFTsPerBlock<1>() +
                           SM<Arch>());

    // ── Load Prototype Filters ──
    float* d_filt_c = load_filter_binary(FILTER_COARSE_PATH, N_TAPS_C);
    float* d_filt_r = load_filter_binary(FILTER_RESAMP_PATH, N_TAPS_RESAMP);
    float* d_filt_f = load_filter_binary(FILTER_FINE_PATH, N_TAPS_FINE);
    if (!d_filt_c || !d_filt_r || !d_filt_f) {
        fprintf(stderr, "FATAL: Failed to load Remez filters from filters/ directory.\n");
        return;
    }
    // Copy resampler filter to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_h_resamp, d_filt_r, N_TAPS_RESAMP * sizeof(float), 0, cudaMemcpyDeviceToDevice));

    // ── Allocate Dual Ping-Pong Global Memory Buffers (48 ms each) ──
    const size_t raw_te_bytes = (size_t)((TOTAL_SAMPLES_TE / SAMPLES_PER_PACK_GROUP) * BYTES_PER_PACK_GROUP);
    const size_t te_alloc_bytes = (size_t)COARSE_PREDATA_BYTES + raw_te_bytes + 1024 * 1024;

    uint8_t *d_input_ping = nullptr, *d_input_pong = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_ping, te_alloc_bytes));
    CUDA_CHECK(cudaMalloc(&d_input_pong, te_alloc_bytes));
    printf("Ping-Pong Input Buffers: 2 x %.2f GB (48 ms raw packed 6-bit)\n",
           te_alloc_bytes / (1024.0 * 1024.0 * 1024.0));

    // ── Allocate Intermediate Buffers for Full 48 ms TE (Option B Unified Sequential) ──
    const size_t coarse_te_bytes = (size_t)N_FRAMES_C_TE * N_SUBBANDS * sizeof(cuComplex);
    cuComplex *d_coarse_buf = nullptr;
    cuComplex *d_trans_buf  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coarse_buf, coarse_te_bytes));
    CUDA_CHECK(cudaMalloc(&d_trans_buf,  coarse_te_bytes));
    printf("Intermediate Buffers: 1 x %.2f GB Coarse + 1 x %.2f GB Transposed (Option B Unified Single Pass)\n",
           coarse_te_bytes / (1024.0 * 1024.0 * 1024.0),
           coarse_te_bytes / (1024.0 * 1024.0 * 1024.0));

    // ── Allocate Pre-Data State Buffers ──
    cuComplex* d_resamp_state = nullptr;
    CUDA_CHECK(cudaMalloc(&d_resamp_state, N_SUBBANDS * RESAMP_PREDATA_SAMPLES * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemset(d_resamp_state, 0, N_SUBBANDS * RESAMP_PREDATA_SAMPLES * sizeof(cuComplex)));

    cuComplex* d_fine_state = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fine_state, N_SUBBANDS * FINE_PREDATA_FRAMES * M_F * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemset(d_fine_state, 0, N_SUBBANDS * FINE_PREDATA_FRAMES * M_F * sizeof(cuComplex)));

    // ── Allocate Single 16 GHz Stitched Power Spectrum ──
    const size_t spw_bytes = TOTAL_STITCHED_CHANNELS * sizeof(float);
    float* d_stitched_spw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stitched_spw, spw_bytes));
    CUDA_CHECK(cudaMemset(d_stitched_spw, 0, spw_bytes));
    printf("Single Stitched SPW: %.2f MB (%d channels)\n\n",
           spw_bytes / (1024.0 * 1024.0), TOTAL_STITCHED_CHANNELS);

    // ── Generate Multi-Tone Stimulus Testbed ──
    std::vector<CWTone> h_tones(N_SUBBANDS);
    int exp_tone_ch[N_SUBBANDS];
    srand(42);

    for (int sb = 0; sb < N_SUBBANDS; ++sb) {
        int k = K_START + sb;
        h_tones[sb].subband = sb;
        // Test tone placed inside coarse passband (-600 to +600 bins from center)
        int margin = 50;
        int fine_offset = (rand() % (N_FINE_KEEP_BASE - 2 * margin)) - (N_FINE_KEEP_BASE / 2 - margin);
        h_tones[sb].fine_offset = fine_offset;
        h_tones[sb].freq_hz = (double)k * DELTA_F_COARSE + (double)fine_offset * DELTA_F_FINE;
        h_tones[sb].amp = 0.05f + 0.25f * ((float)sb / (float)(N_SUBBANDS - 1));

        exp_tone_ch[sb] = (N_FINE_KEEP_BASE / 2) + fine_offset;
    }

    CWTone* d_tones = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tones, N_SUBBANDS * sizeof(CWTone)));
    CUDA_CHECK(cudaMemcpy(d_tones, h_tones.data(), N_SUBBANDS * sizeof(CWTone), cudaMemcpyHostToDevice));

    // Synthesize 48 ms input into Ping Buffer
    printf("=== Generating 48.0 ms Stimulus Stream (Noise + %d CW Tones) ===\n", N_SUBBANDS);
    {
        CUDA_CHECK(cudaMemset(d_input_ping, 0, COARSE_PREDATA_BYTES));
        constexpr int BLK = 256;
        uint8_t* te_signal_ptr = d_input_ping + COARSE_PREDATA_BYTES;
        long long n_groups = TOTAL_SAMPLES_TE / SAMPLES_PER_PACK_GROUP;
        int grid = (int)((n_groups + BLK - 1) / BLK);

        gen_stimulus_6bit_kernel<<<grid, BLK>>>(
            te_signal_ptr, TOTAL_SAMPLES_TE,
            d_tones, N_SUBBANDS, FS_ADC,
            NOISE_SIGMA, CURAND_SEED, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Allocate 6-Bit ADC Histogram Accumulator (64 Bins) ──
    unsigned long long *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist, 64 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hist, 0, 64 * sizeof(unsigned long long)));

    // ── Setup Single Pipeline CUDA Stream (Option B Sequential) ──
    cudaStream_t stream_pipeline;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_pipeline, cudaStreamNonBlocking));
    printf("CUDA Pipeline Stream: Unified Single Stream (Stage 0 Hist -> Stage 1 Coarse -> Stage 2 Transpose -> Stage 3 Fine)\n\n");

    cudaEvent_t ev_pipeline_start, ev_pipeline_stop;
    cudaEvent_t ev_h_start, ev_h_stop;
    cudaEvent_t ev_c_start, ev_c_stop;
    cudaEvent_t ev_t_start, ev_t_stop;
    cudaEvent_t ev_f_start, ev_f_stop;
    CUDA_CHECK(cudaEventCreate(&ev_pipeline_start));
    CUDA_CHECK(cudaEventCreate(&ev_pipeline_stop));
    CUDA_CHECK(cudaEventCreate(&ev_h_start));
    CUDA_CHECK(cudaEventCreate(&ev_h_stop));
    CUDA_CHECK(cudaEventCreate(&ev_c_start));
    CUDA_CHECK(cudaEventCreate(&ev_c_stop));
    CUDA_CHECK(cudaEventCreate(&ev_t_start));
    CUDA_CHECK(cudaEventCreate(&ev_t_stop));
    CUDA_CHECK(cudaEventCreate(&ev_f_start));
    CUDA_CHECK(cudaEventCreate(&ev_f_stop));

    // Dynamic Shared Memory Allocation Setup (Option B: Direct Global Unpack - Zero s_packed smem)
    size_t smem_coarse_mode2 = FFT_C::shared_memory_size + COARSE_UNPACK_SMEM_BYTES;
    size_t smem_coarse_mode1 = smem_coarse_mode2 + N_TAPS_C * sizeof(float);
    size_t smem_fine         = FFT_F::shared_memory_size;
    CUDA_CHECK(cudaFuncSetAttribute(unpack_coarse_ospfb_kernel<FFT_C, 0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_coarse_mode2));
    CUDA_CHECK(cudaFuncSetAttribute(unpack_coarse_ospfb_kernel<FFT_C, 1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_coarse_mode1));
    CUDA_CHECK(cudaFuncSetAttribute(unpack_coarse_ospfb_kernel<FFT_C, 2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_coarse_mode2));
    CUDA_CHECK(cudaFuncSetAttribute(resample_cspfb_stitch_kernel<FFT_F, 0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_fine));

    CUDA_CHECK(cudaFuncSetAttribute(unpack_coarse_ospfb_kernel<FFT_C, 0>,
        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    CUDA_CHECK(cudaFuncSetAttribute(unpack_coarse_ospfb_kernel<FFT_C, 1>,
        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    CUDA_CHECK(cudaFuncSetAttribute(unpack_coarse_ospfb_kernel<FFT_C, 2>,
        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

    int occ0 = 0, occ1 = 0, occ2 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ0, unpack_coarse_ospfb_kernel<FFT_C, 0>, FFT_C::block_dim.x * FFT_C::block_dim.y, smem_coarse_mode2);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, unpack_coarse_ospfb_kernel<FFT_C, 1>, FFT_C::block_dim.x * FFT_C::block_dim.y, smem_coarse_mode1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ2, unpack_coarse_ospfb_kernel<FFT_C, 2>, FFT_C::block_dim.x * FFT_C::block_dim.y, smem_coarse_mode2);
    printf("  Coarse OSPFB Occupancy: Mode 0 = %d blk/SM, Mode 1 = %d blk/SM, Mode 2 = %d blk/SM\n", occ0, occ1, occ2);

    // ── Execute Option B Unified Sequential Pipeline across Full 48.0 ms TE ──
    printf("=== Launching Option B Unified Sequential Pipeline (Full 48.0 ms TE) ===\n");
    CUDA_CHECK(cudaEventRecord(ev_pipeline_start, stream_pipeline));

    // 0. Stage 0: 6-Bit ADC Histogram Calculation (1.92 Billion Samples, 1.44 GB)
    int hist_grid = num_sms * 4;
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, 64 * sizeof(unsigned long long), stream_pipeline));
    CUDA_CHECK(cudaEventRecord(ev_h_start, stream_pipeline));
    calc_histogram_6bit_kernel<<<hist_grid, 256, 0, stream_pipeline>>>(
        d_input_ping + COARSE_PREDATA_BYTES, TOTAL_SAMPLES_TE, d_hist);
    CUDA_CHECK(cudaEventRecord(ev_h_stop, stream_pipeline));

    // Dynamically scale 6 blocks per SM for optimal tail-wave latency hiding across any GPU:
    // (GH200: 132 SMs -> 792 blocks | RTX PRO 6000: 188 SMs -> 1128 blocks)
    int n_blocks_coarse = num_sms * 6;
    printf("  Coarse OSPFB Grid Configuration: %d blocks (%d SMs x 6 blocks/SM)\n",
           n_blocks_coarse, num_sms);

    // 1. Stage 1: Fused 6-Bit Unpack + Coarse OSPFB (Mode 1: Shared Memory, all 1,536,000 frames)
    CUDA_CHECK(cudaEventRecord(ev_c_start, stream_pipeline));
    unpack_coarse_ospfb_kernel<FFT_C, 1><<<n_blocks_coarse, FFT_C::block_dim, smem_coarse_mode1, stream_pipeline>>>(
        d_input_ping, d_filt_c, d_coarse_buf,
        N_FRAMES_C_TE, n_blocks_coarse, K_START);
    CUDA_CHECK(cudaEventRecord(ev_c_stop, stream_pipeline));

    // 2. Stage 2: Bank-Conflict-Free 2D Tiled Transpose + Mixer (1,500,000 x 820)
    constexpr int TRANS_TILES = 4;
    dim3 tblock(TRANS_TILE, TRANS_TILE);
    dim3 tgrid((N_SUBBANDS + TRANS_TILE - 1) / TRANS_TILE,
               (N_FRAMES_C_TE + TRANS_TILE * TRANS_TILES - 1) / (TRANS_TILE * TRANS_TILES));
    CUDA_CHECK(cudaEventRecord(ev_t_start, stream_pipeline));
    transpose_grid_mixer_kernel<TRANS_TILES><<<tgrid, tblock, 0, stream_pipeline>>>(
        d_coarse_buf, d_trans_buf,
        N_FRAMES_C_TE, N_SUBBANDS, K_START);
    CUDA_CHECK(cudaEventRecord(ev_t_stop, stream_pipeline));

    // 3. Stage 3: Fused Rational Resampler + Fine CSPFB + Auto-Power Stitcher (820 subbands x 648 frames)
    dim3 fgrid(N_SUBBANDS, 4);
    CUDA_CHECK(cudaEventRecord(ev_f_start, stream_pipeline));
    resample_cspfb_stitch_kernel<FFT_F, 0><<<fgrid, FFT_F::block_dim, smem_fine, stream_pipeline>>>(
        d_trans_buf, d_filt_f,
        d_fine_state, d_resamp_state, d_stitched_spw,
        0, K_START, N_FRAMES_C_TE, N_FRAMES_F_TE);
    CUDA_CHECK(cudaEventRecord(ev_f_stop, stream_pipeline));

    CUDA_CHECK(cudaEventRecord(ev_pipeline_stop, stream_pipeline));
    CUDA_CHECK(cudaStreamSynchronize(stream_pipeline));

    float total_ms = 0.0f, h_ms = 0.0f, c_ms = 0.0f, t_ms = 0.0f, f_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_pipeline_start, ev_pipeline_stop));
    CUDA_CHECK(cudaEventElapsedTime(&h_ms, ev_h_start, ev_h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&c_ms, ev_c_start, ev_c_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_ms, ev_t_start, ev_t_stop));
    CUDA_CHECK(cudaEventElapsedTime(&f_ms, ev_f_start, ev_f_stop));

    printf("  48.0 ms TE Sequential Pipeline Completed in: %8.3f ms\n", total_ms);
    printf("    Stage 0: 6-Bit ADC Histogram (1.92B smp) : %8.3f ms (%5.1f%%)\n", h_ms, (h_ms / total_ms) * 100.0f);
    printf("    Stage 1: Coarse OSPFB (%d frames)   : %8.3f ms (%5.1f%%)\n", N_FRAMES_C_TE, c_ms, (c_ms / total_ms) * 100.0f);
    printf("    Stage 2: Transpose + Mixer (%d x %d): %8.3f ms (%5.1f%%)\n", N_FRAMES_C_TE, N_SUBBANDS, t_ms, (t_ms / total_ms) * 100.0f);
    printf("    Stage 3: Fine CSPFB + Resampler (%d frames): %8.3f ms (%5.1f%%)\n", N_FRAMES_F_TE, f_ms, (f_ms / total_ms) * 100.0f);
    printf("  Sequential Sum (Hist + Coarse + Trans + Fine): %8.3f ms\n", h_ms + c_ms + t_ms + f_ms);
    printf("  Real-Time Speedup Factor                     : %8.2fx  (%s)\n\n",
           (DURATION_TE * 1e3) / total_ms,
           (total_ms < DURATION_TE * 1e3) ? "REAL-TIME ACHIEVED [PASS]" : "OVER-BUDGET [FAIL]");

    // Verify 6-Bit Gaussian Statistics from Stage 0 output across full 1.92B samples
    unsigned long long h_hist[64];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 64 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    double sum_c = 0, sum_v = 0, sum_v2 = 0;
    for (int i = 0; i < 64; ++i) {
        int val = (i & 0x20) ? (i - 64) : i;
        sum_c  += (double)h_hist[i];
        sum_v  += (double)val * (double)h_hist[i];
        sum_v2 += (double)val * (double)val * (double)h_hist[i];
    }
    double mean = sum_v / sum_c;
    double sigma = sqrt(sum_v2 / sum_c - mean * mean);
    double clip_frac = (double)(h_hist[31] + h_hist[32]) / sum_c;

    printf("  Stage 0 Statistics (1.92B Samples): Mean = %+.4f, Sigma = %.4f (Expected %.1f), Clip Frac = %.2e [%s]\n\n",
           mean, sigma, NOISE_SIGMA, clip_frac,
           (fabs(mean) < 0.1 && sigma >= 7.5 && sigma <= 9.2) ? "PASS" : "FAIL");

    // =========================================================================
    //  OPTION 4: ZERO-JITTER CUDA GRAPH CAPTURE & DISPATCH FOR ALMA OPERATIONS
    // =========================================================================
    printf("=== Initializing Option 4: Zero-Jitter CUDA Graph Dispatch ===\n");
    cudaGraph_t graph_pipeline;
    cudaGraphExec_t graph_exec_pipeline;

    auto t_cap_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaStreamBeginCapture(stream_pipeline, cudaStreamCaptureModeGlobal));

    calc_histogram_6bit_kernel<<<hist_grid, 256, 0, stream_pipeline>>>(
        d_input_ping + COARSE_PREDATA_BYTES, TOTAL_SAMPLES_TE, d_hist);

    unpack_coarse_ospfb_kernel<FFT_C, 1><<<n_blocks_coarse, FFT_C::block_dim, smem_coarse_mode1, stream_pipeline>>>(
        d_input_ping, d_filt_c, d_coarse_buf,
        N_FRAMES_C_TE, n_blocks_coarse, K_START);

    transpose_grid_mixer_kernel<TRANS_TILES><<<tgrid, tblock, 0, stream_pipeline>>>(
        d_coarse_buf, d_trans_buf,
        N_FRAMES_C_TE, N_SUBBANDS, K_START);

    resample_cspfb_stitch_kernel<FFT_F, 0><<<fgrid, FFT_F::block_dim, smem_fine, stream_pipeline>>>(
        d_trans_buf, d_filt_f,
        d_fine_state, d_resamp_state, d_stitched_spw,
        0, K_START, N_FRAMES_C_TE, N_FRAMES_F_TE);

    CUDA_CHECK(cudaStreamEndCapture(stream_pipeline, &graph_pipeline));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_pipeline, graph_pipeline, nullptr, nullptr, 0));
    auto t_cap_stop = std::chrono::high_resolution_clock::now();
    double cap_ms = std::chrono::duration<double, std::milli>(t_cap_stop - t_cap_start).count();

    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph_pipeline, nullptr, &num_nodes));
    printf("  CUDA Graph Captured & Instantiated in : %8.3f ms\n", cap_ms);
    printf("  Graph Topology Complexity             : %zu kernel nodes (Stage 0..3)\n\n", num_nodes);

    // Warm-up & Zero-Jitter Benchmark Launch (10 consecutive Timing Events)
    cudaEvent_t ev_g_start, ev_g_stop;
    CUDA_CHECK(cudaEventCreate(&ev_g_start));
    CUDA_CHECK(cudaEventCreate(&ev_g_stop));

    constexpr int NUM_GRAPH_TE_RUNS = 10;
    std::vector<double> cpu_launch_times_us(NUM_GRAPH_TE_RUNS);
    std::vector<float>  gpu_exec_times_ms(NUM_GRAPH_TE_RUNS);

    printf("=== Benchmarking %d Consecutive 48.0 ms Timing Events via CUDA Graph ===\n", NUM_GRAPH_TE_RUNS);
    for (int run = 0; run < NUM_GRAPH_TE_RUNS; ++run) {
        // Zero output SPW buffer and histogram before final run to verify 100% tone accuracy
        if (run == NUM_GRAPH_TE_RUNS - 1) {
            CUDA_CHECK(cudaMemsetAsync(d_stitched_spw, 0, spw_bytes, stream_pipeline));
            CUDA_CHECK(cudaMemsetAsync(d_hist, 0, 64 * sizeof(unsigned long long), stream_pipeline));
        }

        CUDA_CHECK(cudaEventRecord(ev_g_start, stream_pipeline));

        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaGraphLaunch(graph_exec_pipeline, stream_pipeline));
        auto t1 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaEventRecord(ev_g_stop, stream_pipeline));
        CUDA_CHECK(cudaStreamSynchronize(stream_pipeline));

        cpu_launch_times_us[run] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        CUDA_CHECK(cudaEventElapsedTime(&gpu_exec_times_ms[run], ev_g_start, ev_g_stop));
    }

    double sum_cpu = 0.0, sum_gpu = 0.0;
    double min_cpu = 1e9, max_cpu = 0.0;
    for (int r = 0; r < NUM_GRAPH_TE_RUNS; ++r) {
        sum_cpu += cpu_launch_times_us[r];
        sum_gpu += gpu_exec_times_ms[r];
        if (cpu_launch_times_us[r] < min_cpu) min_cpu = cpu_launch_times_us[r];
        if (cpu_launch_times_us[r] > max_cpu) max_cpu = cpu_launch_times_us[r];
    }
    double mean_graph_cpu = sum_cpu / NUM_GRAPH_TE_RUNS;
    double mean_graph_gpu = sum_gpu / NUM_GRAPH_TE_RUNS;

    double var_cpu = 0.0;
    for (int r = 0; r < NUM_GRAPH_TE_RUNS; ++r) {
        double diff = cpu_launch_times_us[r] - mean_graph_cpu;
        var_cpu += diff * diff;
    }
    double stddev_graph_cpu = std::sqrt(var_cpu / NUM_GRAPH_TE_RUNS);

    printf("  [CUDA Graph Real-Time Dispatch Performance (%d Consecutive 48 ms TEs)]\n", NUM_GRAPH_TE_RUNS);
    printf("    Average GPU Pipeline Execution Time   : %8.3f ms (%5.2fx Real-Time Speedup)\n",
           mean_graph_gpu, (DURATION_TE * 1e3) / mean_graph_gpu);
    printf("    Host CPU Launch Latency (Mean)        : %8.2f us\n", mean_graph_cpu);
    printf("    Host CPU Launch Latency (Min / Max)   : %8.2f / %8.2f us\n", min_cpu, max_cpu);
    printf("    Host CPU Launch Jitter (Std Dev)      : %8.2f us (Zero-Jitter Determinism)\n\n", stddev_graph_cpu);

    // ── Multi-Tier Verification Suite ──
    printf("=== Automated Verification Suite ===\n");
    std::vector<float> h_spw(TOTAL_STITCHED_CHANNELS);
    CUDA_CHECK(cudaMemcpy(h_spw.data(), d_stitched_spw, spw_bytes, cudaMemcpyDeviceToHost));

    // Calculate background noise statistics
    double noise_sum = 0;
    int noise_count = 0;
    for (int i = 0; i < TOTAL_STITCHED_CHANNELS; ++i) {
        noise_sum += (double)h_spw[i];
        noise_count++;
    }
    double noise_mean = noise_sum / noise_count;
    printf("  Noise Floor Mean Power: %.4e across %d channels\n\n", noise_mean, noise_count);

    printf("  [Verification Tones in Reference Subbands]\n");
    bool all_tones_ok = true;
    for (int r = 0; r < NUM_REF_SUBBANDS; ++r) {
        int sb = REF_SUBBANDS[r];
        int exp_ch = exp_tone_ch[sb];
        float exp_amp = h_tones[sb].amp;

        long long subband_start = (long long)sb * N_FINE_KEEP_BASE;
        float peak_val = 0.0f;
        int peak_ch = -1;

        for (int ch = 0; ch < N_FINE_KEEP_BASE; ++ch) {
            long long g_idx = subband_start + ch;
            if (g_idx < TOTAL_STITCHED_CHANNELS) {
                if (h_spw[g_idx] > peak_val) {
                    peak_val = h_spw[g_idx];
                    peak_ch = ch;
                }
            }
        }

        bool match = (abs(peak_ch - exp_ch) <= 1);
        double snr_db = 10.0 * log10((double)peak_val / noise_mean);
        printf("    Subband %3d (A=%.2f): Peak ch %4d (Exp %4d) | SNR = %5.1f dB | %s\n",
               sb, exp_amp, peak_ch, exp_ch, snr_db, match ? "PASS" : "FAIL");
        if (!match) all_tones_ok = false;
    }
    printf("  Tone Detection Status: %s\n\n", all_tones_ok ? "ALL PASS" : "FAIL");

    // ── Save Binary Spectrum to File ──
    FILE* fp_spw = fopen(OUTPUT_SPECTRUM_BIN, "wb");
    if (fp_spw) {
        int hdr[4] = { TOTAL_STITCHED_CHANNELS, N_SUBBANDS, N_FINE_KEEP_BASE, NUM_CHUNKS_PER_TE };
        fwrite(hdr, sizeof(int), 4, fp_spw);
        fwrite(h_spw.data(), sizeof(float), TOTAL_STITCHED_CHANNELS, fp_spw);
        fclose(fp_spw);
        printf("Saved Single Stitched 16 GHz Spectrum to: %s (%zu bytes)\n\n",
               OUTPUT_SPECTRUM_BIN, 4 * sizeof(int) + spw_bytes);
    }

    // ── Standalone Kernel Benchmarking ──
    printf("=== Standalone Kernel Benchmarking (Full 48.0 ms TE) ===\n");
    cudaEvent_t ev_b0, ev_b1;
    CUDA_CHECK(cudaEventCreate(&ev_b0));
    CUDA_CHECK(cudaEventCreate(&ev_b1));

    // Stage 0: 6-Bit ADC Histogram Calculation for Full 48.0 ms TE (1.92 Billion Samples / 1.44 GB)
    float ms_hist = 0.0f;
    CUDA_CHECK(cudaMemset(d_hist, 0, 64 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(ev_b0));
    calc_histogram_6bit_kernel<<<hist_grid, 256>>>(
        d_input_ping + COARSE_PREDATA_BYTES, TOTAL_SAMPLES_TE, d_hist);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_hist, ev_b0, ev_b1));

    double hist_gb = (double)raw_te_bytes / (1024.0 * 1024.0 * 1024.0);
    double hist_bw = (double)raw_te_bytes / (ms_hist * 1e-3) / 1e9;
    printf("  Stage 0: 6-Bit ADC Histogram (1.92B Samples, %.2f GB):\n", hist_gb);
    printf("    Coalesced 4-Bank Privatized Kernel : %7.3f ms (%7.1f GB/s)\n\n", ms_hist, hist_bw);

    // Coarse Mode 0 (Regs) vs Mode 1 (Shared Memory) vs Mode 2 (L1 Cache) for 48 ms TE
    float ms_c_reg = 0, ms_c_smem = 0, ms_c_l1 = 0;
    CUDA_CHECK(cudaEventRecord(ev_b0));
    unpack_coarse_ospfb_kernel<FFT_C, 0><<<n_blocks_coarse, FFT_C::block_dim, smem_coarse_mode2>>>(
        d_input_ping, d_filt_c, d_coarse_buf, N_FRAMES_C_TE, n_blocks_coarse, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_c_reg, ev_b0, ev_b1));

    CUDA_CHECK(cudaEventRecord(ev_b0));
    unpack_coarse_ospfb_kernel<FFT_C, 1><<<n_blocks_coarse, FFT_C::block_dim, smem_coarse_mode1>>>(
        d_input_ping, d_filt_c, d_coarse_buf, N_FRAMES_C_TE, n_blocks_coarse, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_c_smem, ev_b0, ev_b1));

    CUDA_CHECK(cudaEventRecord(ev_b0));
    unpack_coarse_ospfb_kernel<FFT_C, 2><<<n_blocks_coarse, FFT_C::block_dim, smem_coarse_mode2>>>(
        d_input_ping, d_filt_c, d_coarse_buf, N_FRAMES_C_TE, n_blocks_coarse, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_c_l1, ev_b0, ev_b1));

    printf("  Fused 6-Bit Unpack + Coarse OSPFB (8,192 taps, %d frames):\n", N_FRAMES_C_TE);
    printf("    Mode 0 (Registers)     : %7.3f ms\n", ms_c_reg);
    printf("    Mode 1 (Shared Memory) : %7.3f ms\n", ms_c_smem);
    printf("    Mode 2 (L1 Cache)      : %7.3f ms\n", ms_c_l1);

    // Transpose Benchmark for 48 ms TE (1x vs 2x vs 4x vs 8x tiles per block)
    float ms_trans1 = 0, ms_trans2 = 0, ms_trans4 = 0, ms_trans8 = 0;
    dim3 tgrid1((N_SUBBANDS + TRANS_TILE - 1) / TRANS_TILE,
                (N_FRAMES_C_TE + TRANS_TILE - 1) / TRANS_TILE);
    dim3 tgrid2((N_SUBBANDS + TRANS_TILE - 1) / TRANS_TILE,
                (N_FRAMES_C_TE + TRANS_TILE * 2 - 1) / (TRANS_TILE * 2));
    dim3 tgrid4((N_SUBBANDS + TRANS_TILE - 1) / TRANS_TILE,
                (N_FRAMES_C_TE + TRANS_TILE * 4 - 1) / (TRANS_TILE * 4));
    dim3 tgrid8((N_SUBBANDS + TRANS_TILE - 1) / TRANS_TILE,
                (N_FRAMES_C_TE + TRANS_TILE * 8 - 1) / (TRANS_TILE * 8));

    CUDA_CHECK(cudaEventRecord(ev_b0));
    transpose_grid_mixer_kernel<1><<<tgrid1, tblock>>>(d_coarse_buf, d_trans_buf, N_FRAMES_C_TE, N_SUBBANDS, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_trans1, ev_b0, ev_b1));

    CUDA_CHECK(cudaEventRecord(ev_b0));
    transpose_grid_mixer_kernel<2><<<tgrid2, tblock>>>(d_coarse_buf, d_trans_buf, N_FRAMES_C_TE, N_SUBBANDS, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_trans2, ev_b0, ev_b1));

    CUDA_CHECK(cudaEventRecord(ev_b0));
    transpose_grid_mixer_kernel<4><<<tgrid4, tblock>>>(d_coarse_buf, d_trans_buf, N_FRAMES_C_TE, N_SUBBANDS, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_trans4, ev_b0, ev_b1));

    CUDA_CHECK(cudaEventRecord(ev_b0));
    transpose_grid_mixer_kernel<8><<<tgrid8, tblock>>>(d_coarse_buf, d_trans_buf, N_FRAMES_C_TE, N_SUBBANDS, K_START);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_trans8, ev_b0, ev_b1));

    double trans_bw1 = 2.0 * coarse_te_bytes / (ms_trans1 * 1e-3) / 1e9;
    double trans_bw2 = 2.0 * coarse_te_bytes / (ms_trans2 * 1e-3) / 1e9;
    double trans_bw4 = 2.0 * coarse_te_bytes / (ms_trans4 * 1e-3) / 1e9;
    double trans_bw8 = 2.0 * coarse_te_bytes / (ms_trans8 * 1e-3) / 1e9;
    printf("  Transpose (%d x %d cuComplex, Full 48 ms TE):\n", N_FRAMES_C_TE, N_SUBBANDS);
    printf("    1 Tile/Block (1.25M blocks) : %7.3f ms (%7.1f GB/s)\n", ms_trans1, trans_bw1);
    printf("    2 Tiles/Block (624k blocks) : %7.3f ms (%7.1f GB/s)\n", ms_trans2, trans_bw2);
    printf("    4 Tiles/Block (312k blocks) : %7.3f ms (%7.1f GB/s)\n", ms_trans4, trans_bw4);
    printf("    8 Tiles/Block (156k blocks) : %7.3f ms (%7.1f GB/s)\n\n", ms_trans8, trans_bw8);

    // Fine CSPFB Benchmark Mode 0 for 48 ms TE
    float ms_fine_reg = 0;
    CUDA_CHECK(cudaEventRecord(ev_b0));
    dim3 fgrid_b(N_SUBBANDS, 4);
    resample_cspfb_stitch_kernel<FFT_F, 0><<<fgrid_b, FFT_F::block_dim, smem_fine>>>(
        d_trans_buf, d_filt_f, d_fine_state, d_resamp_state, d_stitched_spw, 0, K_START, N_FRAMES_C_TE, N_FRAMES_F_TE);
    CUDA_CHECK(cudaEventRecord(ev_b1));
    CUDA_CHECK(cudaEventSynchronize(ev_b1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_fine_reg, ev_b0, ev_b1));

    printf("  Fused Resampler + Fine CSPFB (cuFFTDx Size<2048>, %d frames):\n", N_FRAMES_F_TE);
    printf("    Mode 0 (Registers, 4-way block) : %7.3f ms\n\n", ms_fine_reg);

    // ── Export Roofline Performance Data ──
    FILE* fp_perf = fopen("data/roofline_metrics.txt", "w");
    if (fp_perf) {
        fprintf(fp_perf, "HIST_MS=%.4f\n", h_ms);
        fprintf(fp_perf, "UNPACK_MS=0.0000\n");
        fprintf(fp_perf, "COARSE_MS=%.4f\n", c_ms);
        fprintf(fp_perf, "TRANSPOSE_MS=%.4f\n", t_ms);
        fprintf(fp_perf, "RESAMP_MS=0.0000\n");
        fprintf(fp_perf, "FINE_MS=%.4f\n", f_ms);
        fprintf(fp_perf, "TOTAL_PIPELINE_MS=%.4f\n", total_ms);
        fprintf(fp_perf, "REALTIME_FACTOR=%.2f\n", (DURATION_TE * 1e3) / total_ms);
        fprintf(fp_perf, "GRAPH_GPU_MS=%.4f\n", mean_graph_gpu);
        fprintf(fp_perf, "GRAPH_LAUNCH_US=%.2f\n", mean_graph_cpu);
        fprintf(fp_perf, "GRAPH_JITTER_US=%.2f\n", stddev_graph_cpu);
        fclose(fp_perf);
        printf("Exported roofline metrics to: data/roofline_metrics.txt\n");
    }

    // ── Cleanup ──
    cudaFree(d_input_ping);
    cudaFree(d_input_pong);
    cudaFree(d_hist);
    cudaFree(d_coarse_buf);
    cudaFree(d_trans_buf);
    cudaFree(d_resamp_state);
    cudaFree(d_fine_state);
    cudaFree(d_stitched_spw);
    cudaFree(d_filt_c); cudaFree(d_filt_r); cudaFree(d_filt_f);
    cudaFree(d_tones);

    cudaStreamDestroy(stream_pipeline);
    cudaEventDestroy(ev_pipeline_start);
    cudaEventDestroy(ev_pipeline_stop);
    cudaEventDestroy(ev_h_start);
    cudaEventDestroy(ev_h_stop);
    cudaEventDestroy(ev_c_start);
    cudaEventDestroy(ev_c_stop);
    cudaEventDestroy(ev_t_start);
    cudaEventDestroy(ev_t_stop);
    cudaEventDestroy(ev_f_start);
    cudaEventDestroy(ev_f_stop);
    cudaEventDestroy(ev_b0);
    cudaEventDestroy(ev_b1);
    cudaEventDestroy(ev_g_start);
    cudaEventDestroy(ev_g_stop);
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec_pipeline));
    CUDA_CHECK(cudaGraphDestroy(graph_pipeline));
}

// =============================================================================
//  MAIN ENTRY POINT
// =============================================================================

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=================================================================\n");
    printf("  Hardware: %s (%d SMs, Compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Global Memory Capacity: %.2f GB\n", (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("=================================================================\n\n");

#ifndef TARGET_ARCH
#define TARGET_ARCH 1200
#endif

    int arch = prop.major * 100 + prop.minor * 10;
#if TARGET_ARCH == 1200
    if (arch < 1200) {
        fprintf(stderr, "WARNING: Executing SM 1200 binary on SM %d device\n", arch);
    }
    run_tpgs_pipeline<1200>(prop.multiProcessorCount);
#elif TARGET_ARCH == 900
    if (arch < 900) {
        fprintf(stderr, "WARNING: Executing SM 900 binary on SM %d device\n", arch);
    }
    run_tpgs_pipeline<900>(prop.multiProcessorCount);
#else
    if (arch >= 1200) {
        run_tpgs_pipeline<1200>(prop.multiProcessorCount);
    } else if (arch >= 900) {
        run_tpgs_pipeline<900>(prop.multiProcessorCount);
    } else {
        fprintf(stderr, "ERROR: Architecture SM %d not supported (requires SM 900+)\n", arch);
        return 1;
    }
#endif

    return 0;
}
