// =============================================================================
//  FILE        : tpgs_params.h
//  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
//  DESCRIPTION : Mathematical invariants, multi-rate geometry, pre-data dimensions,
//                cuFFTDx transform configurations, and constant memory layouts.
//  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
//  VERSION     : 0.1.0
//  DATE        : October 2026
//  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
// =============================================================================

#pragma once

#include <stdint.h>
#include <cmath>
#include <cuComplex.h>

namespace tpgs {

constexpr const char* PIPELINE_VERSION         = "0.1.0";

// ── 1. Global ADC and Observation Timing Invariants ──────────────────────────
constexpr double FS_ADC                 = 40.0e9;       // 40.0 Gsps (40,000 MSPS)
constexpr double NYQUIST_BW             = FS_ADC / 2.0; // 20.0 GHz
constexpr double DURATION_TE            = 0.048;        // 48.0 ms Timing Event (TE) epoch
constexpr double DURATION_CHUNK         = 0.016;        // 16.0 ms processing chunk
constexpr int    NUM_CHUNKS_PER_TE      = 3;            // Exactly 3 chunks per 48 ms TE

constexpr long long TOTAL_SAMPLES_TE    = (long long)(FS_ADC * DURATION_TE);    // 1,920,000,000 samples
constexpr long long TOTAL_SAMPLES_CHUNK = (long long)(FS_ADC * DURATION_CHUNK); // 640,000,000 samples

// 6-bit packing: 4 samples in 3 bytes (24 bits)
constexpr int    SAMPLES_PER_PACK_GROUP = 4;
constexpr int    BYTES_PER_PACK_GROUP   = 3;

// ── 2. Stage 1: Coarse OSPFB Parameters (cuFFTDx Size<2048>) ─────────────────
constexpr int    M_C                    = 2048;         // Coarse FFT channels
constexpr int    D_C                    = 1250;         // Coarse decimation factor (OS = 1.6384)
constexpr int    K1_C                   = 4;            // Coarse filter taps per branch
constexpr int    N_TAPS_C               = M_C * K1_C;   // Total coarse taps = 8,192
constexpr int    N_NYQUIST_C            = M_C / 2;      // 1024 active positive frequency subbands

constexpr double DELTA_F_COARSE         = FS_ADC / M_C; // 19.53125 MHz
constexpr double FS_COARSE              = FS_ADC / D_C; // 32.000 MSPS

constexpr int    N_FRAMES_C_CHUNK       = (int)(TOTAL_SAMPLES_CHUNK / D_C); // 512,000 frames / 16 ms
constexpr int    N_FRAMES_C_TE          = N_FRAMES_C_CHUNK * NUM_CHUNKS_PER_TE; // 1,536,000 frames / 48 ms

// Coarse Pre-Data: FIR boundary history prefix
constexpr int    COARSE_PREDATA_SAMPLES = N_TAPS_C;     // 8,192 ADC samples (6.0 KB)
constexpr int    COARSE_PREDATA_BYTES   = (COARSE_PREDATA_SAMPLES / SAMPLES_PER_PACK_GROUP) * BYTES_PER_PACK_GROUP;

// Coarse Dual-FFT Unpack Buffer Sizing (eliminates magic numbers 9472 & 592)
constexpr int    FFTS_PER_BLOCK_C              = 2;            // 2 coarse FFTs per block
constexpr int    COARSE_DUAL_FRAME_SPAN        = (FFTS_PER_BLOCK_C - 1) * D_C + N_TAPS_C; // 9,442 samples
constexpr int    COARSE_UNPACK_SAMPLES_PER_TH  = 16;           // 4 x float4 per thread
constexpr int    COARSE_UNPACK_BYTES_PER_TH    = 12;           // 4 x 24-bit chunks (12 bytes) per thread
constexpr int    COARSE_UNPACK_THREADS         = (COARSE_DUAL_FRAME_SPAN + COARSE_UNPACK_SAMPLES_PER_TH - 1) / COARSE_UNPACK_SAMPLES_PER_TH; // 592 threads
constexpr int    COARSE_UNPACK_SMEM_FLOATS     = COARSE_UNPACK_THREADS * COARSE_UNPACK_SAMPLES_PER_TH; // 9,472 floats
constexpr size_t COARSE_UNPACK_SMEM_BYTES      = (size_t)COARSE_UNPACK_SMEM_FLOATS * sizeof(float); // 37,888 bytes

// ── 3. Subband Selection: Central 16 GHz out of 20 GHz ───────────────────────
constexpr double BW_SELECTED            = 16.0e9;       // 16.0 GHz central bandwidth
// Subband 102 starts at: 102 * 19.53125 MHz = 1992.1875 MHz (2.0 GHz)
constexpr int    K_START                = 102;
// 820 subbands * 19.53125 MHz = 16.015625 GHz coverage (102 to 921)
constexpr int    N_SUBBANDS             = 820;
constexpr int    K_END                  = K_START + N_SUBBANDS; // 922

// ── 4. Constant Memory Phase Correction (64.0 KB Limit) ──────────────────────
// D_C / M_C = 1250 / 2048 = 625 / 1024 -> exact periodicity q = 1024 frames
constexpr int    PHASE_PERIOD_P         = 625;
constexpr int    PHASE_PERIOD_Q         = 1024;
constexpr int    PHASE_PERIOD_MASK      = PHASE_PERIOD_Q - 1;   // 1023 (bitwise modulo Q)
constexpr float  TWO_PI_F               = 6.28318530717958647692f;

// Active elements per thread covering subbands K_START..K_END (102..921)
// With STRIDE = 512, e=0 covers 0..511, e=1 covers 512..1023. K_END=922 fits in 2 branches.
constexpr int    EPT_ACTIVE_COARSE      = (K_END + (M_C / 4) - 1) / (M_C / 4); // 2

constexpr int    PHASE_TABLE_ENTRIES    = 8 * N_SUBBANDS; // Unused by analytical SFU rotation
constexpr size_t PHASE_TABLE_BYTES      = PHASE_TABLE_ENTRIES * sizeof(float2);

// ── 5. Stage 1.5: Polyphase Rational Resampler (108 / 125) ───────────────────
constexpr int    I_RESAMP               = 108;          // Interpolation factor
constexpr int    D_RESAMP               = 125;          // Decimation factor
constexpr int    KR_RESAMP              = 8;            // Taps per phase
constexpr int    N_TAPS_RESAMP          = I_RESAMP * KR_RESAMP; // 864 taps (3,456 Bytes)

constexpr double FS_RESAMP              = FS_COARSE * (double)I_RESAMP / (double)D_RESAMP; // Exactly 27.648 MSPS

// 512,000 * 108 / 125 = 442,368 resampled samples per subband per 16 ms chunk
constexpr int    N_RESAMP_SAMPLES_CHUNK = (int)((long long)N_FRAMES_C_CHUNK * I_RESAMP / D_RESAMP); // 442,368
constexpr int    N_RESAMP_SAMPLES_TE    = N_RESAMP_SAMPLES_CHUNK * NUM_CHUNKS_PER_TE; // 1,327,104

// Resampler Pre-Data: 7 prior coarse samples per subband
constexpr int    RESAMP_PREDATA_SAMPLES = KR_RESAMP - 1; // 7 coarse samples per subband

// ── 6. Stage 2: Fine CSPFB Parameters (cuFFTDx Size<2048>) ───────────────────
constexpr int    M_F                    = 2048;         // Fine FFT channels (Power of Two!)
constexpr int    D_F                    = 2048;         // Critically sampled (D_F = M_F)
constexpr int    K2_F                   = 5;            // Fine filter taps per branch
constexpr int    N_TAPS_FINE            = M_F * K2_F;   // Total fine taps = 10,240 (40,960 Bytes)

constexpr double DELTA_F_FINE           = FS_RESAMP / M_F; // Exactly 13500.0 Hz (13.5000 kHz)

constexpr int    N_FRAMES_F_CHUNK       = N_RESAMP_SAMPLES_CHUNK / M_F; // 216 fine frames / 16 ms
constexpr int    N_FRAMES_F_TE          = N_FRAMES_F_CHUNK * NUM_CHUNKS_PER_TE; // 648 fine frames / 48 ms

// Fine Pre-Data: Ring buffer state preservation (4 fine frames per subband)
constexpr int    FINE_PREDATA_FRAMES    = K2_F - 1;     // 4 fine frames
constexpr int    FINE_PREDATA_SAMPLES   = FINE_PREDATA_FRAMES * M_F; // 8,192 samples per subband

// ── 7. Frequency Alignment & Seamless Direct Array Stitching ─────────────────
// Coarse to fine bin ratio: 19531.25 / 13.5 = 1446 + 41/54
constexpr int    GRID_NUM               = 41;
constexpr int    GRID_DEN               = 54;
constexpr int    HALF_M_F               = M_F / 2;      // 1024

// Pre-rotation angular step per grid remainder unit: 2*pi / (M_F * GRID_DEN)
// Note: (2*pi * f_shift / FS_COARSE) simplifies identically to: -2*pi * frac_offset / M_F
//                                                            = -THETA_GRID_UNIT * rem
constexpr double THETA_GRID_UNIT        = (2.0 * M_PI) / ((double)M_F * (double)GRID_DEN);

// Channels kept per subband to cover exactly 19.53125 MHz
constexpr int    N_FINE_KEEP_BASE       = 1446;
// Total stitched channels across the central 16.0 GHz band: 16 GHz / 13.5 kHz
constexpr int    TOTAL_STITCHED_CHANNELS= 1185185;      // 1,185,185 contiguous 13.5 kHz bins

// ── 8. Multi-Tone Stimulus Testbed Configuration ─────────────────────────────
struct CWTone {
    int    subband;     // Subband index (0..819, relative to K_START)
    int    fine_offset; // Offset within passband (-700..+700)
    double freq_hz;     // Exact RF frequency in Hz
    float  amp;         // Amplitude relative to quantization levels
};

// Reference interior subbands for automated verification
constexpr int NUM_REF_SUBBANDS = 5;
constexpr int REF_SUBBANDS[NUM_REF_SUBBANDS] = { 10, 200, 410, 600, 810 };

constexpr float NOISE_SIGMA     = 8.0f;  // Standard operating noise level in 6-bit
constexpr float QUANT_DELTA     = 1.0f;
constexpr unsigned long long CURAND_SEED = 123456789ULL;

// ── 9. Filter File Paths ─────────────────────────────────────────────────────
constexpr const char* FILTER_COARSE_PATH  = "filters/h_coarse_2048_remez.bin";
constexpr const char* FILTER_RESAMP_PATH  = "filters/h_resamp_108_125_remez.bin";
constexpr const char* FILTER_FINE_PATH    = "filters/h_fine_2048_remez.bin";
constexpr const char* OUTPUT_SPECTRUM_BIN = "data/power_spectrum_16ghz.bin";

} // namespace tpgs
