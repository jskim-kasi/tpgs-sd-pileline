// =============================================================================
//  FILE        : packet_wsu.h
//  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
//  DESCRIPTION : ALMA WSU ICD 2048-byte packet structure and high-performance
//                Option B 3-byte packing/unpacking device routines.
//  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
//  VERSION     : 0.1.0
//  DATE        : October 2026
//  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
// =============================================================================

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#ifndef ETHER_ADDR_LEN
#define ETHER_ADDR_LEN 6
#endif

namespace tpgs {

#define WSU_ETHERTYPE                     0xFEED
#define WSU_META_FRAME_SIZE               128
#define WSU_SAMPLE_FRAME_SIZE             64
#define WSU_FRAMES_PER_PACKET             30
#define WSU_PACKET_SIZE                   2048 /* META_FRAME_SIZE + (30 * 64) */
#define WSU_SAMPLES_PER_FRAME             80
#define WSU_SAMPLE_BITS                   6
#define WSU_SAMPLE_PAYLOAD_SIZE           60   /* 80 samples * 6 bits / 8 = 60 bytes */
#define WSU_SAMPLES_PER_PACKET            2400 /* 30 * 80 */
#define WSU_HISTOGRAM_BINS                32

#pragma pack(push, 1)

// Meta_Frame Structure (128 Bytes / 1024 Bits)
struct wsu_meta_frame {
    uint8_t  d_mac[ETHER_ADDR_LEN];     // 0..5: Destination MAC Address
    uint8_t  s_mac[ETHER_ADDR_LEN];     // 6..11: Source MAC Address
    uint16_t ethertype;                // 12..13: Ethertype: 0xFEED (Big-Endian)
    uint8_t  unassigned0[2];            // 14..15: Unassigned

    uint8_t  pad_id;                    // 16: Pad ID
    uint8_t  band_id;                   // 17: Band ID
    uint16_t antenna_id;                // 18..19: Antenna ID
    uint8_t  sb_pol_id;                 // 20: Pol and Sideband
    uint8_t  unassigned1[3];            // 21..23: Unassigned

    uint8_t  te_index[6];               // 24..29: 48-bit TE Index
    uint8_t  packets_from_te[6];        // 30..35: 48-bit Packet counter from TE marker

    uint16_t sum_of_samples;           // 36..37: Sum of sample values
    uint16_t sum_of_squares;           // 38..39: Sum of squares of samples
    uint16_t valid_sample_frame_count; // 40..41: Number of valid frames (typically 30)
    uint8_t  unassigned2;               // 42: Unassigned
    uint8_t  epoch_id;                  // 43: Epoch ID

    uint16_t histogram[WSU_HISTOGRAM_BINS]; // 44..107: 32 bins x 16-bit
    uint8_t  unassigned3[20];           // 108..127: Padding
};

// Sample_Frame Structure (64 Bytes / 512 Bits)
struct wsu_sample_frame {
    uint32_t header;                    // 32-bit Header: Flags + Sample_Frames_from_TE
    uint8_t  samples[WSU_SAMPLE_PAYLOAD_SIZE]; // 80 x 6-bit samples in 60 bytes
};

// Full Packet Structure (2048 Bytes)
struct wsu_packet {
    struct wsu_meta_frame   meta;
    struct wsu_sample_frame sample_frames[WSU_FRAMES_PER_PACKET];
};

#pragma pack(pop)

static_assert(sizeof(struct wsu_meta_frame)   == WSU_META_FRAME_SIZE,   "wsu_meta_frame size mismatch");
static_assert(sizeof(struct wsu_sample_frame) == WSU_SAMPLE_FRAME_SIZE, "wsu_sample_frame size mismatch");
static_assert(sizeof(struct wsu_packet)       == WSU_PACKET_SIZE,       "wsu_packet size mismatch");

// =============================================================================
//  FAST DEVICE PACKING & UNPACKING (OPTION B: 4 SAMPLES IN 3 BYTES)
// =============================================================================

// Pack 4 signed 6-bit samples into 3 consecutive bytes (little-endian layout)
__device__ inline void pack_4_samples_6bit(uint8_t* dst, int s0, int s1, int s2, int s3)
{
    // Clamp to signed 6-bit range [-32, 31]
    s0 = max(-32, min(31, s0));
    s1 = max(-32, min(31, s1));
    s2 = max(-32, min(31, s2));
    s3 = max(-32, min(31, s3));

    uint32_t u0 = (uint32_t)(s0 & 0x3F);
    uint32_t u1 = (uint32_t)(s1 & 0x3F);
    uint32_t u2 = (uint32_t)(s2 & 0x3F);
    uint32_t u3 = (uint32_t)(s3 & 0x3F);

    uint32_t chunk = u0 | (u1 << 6) | (u2 << 12) | (u3 << 18);
    dst[0] = (uint8_t)(chunk & 0xFF);
    dst[1] = (uint8_t)((chunk >> 8) & 0xFF);
    dst[2] = (uint8_t)((chunk >> 16) & 0xFF);
}

// Unpack a single 6-bit signed sample from packed 3-byte stream
__device__ inline float unpack_1_sample_6bit(const uint8_t* __restrict__ raw_stream, long long sample_idx)
{
    long long byte_offset = (sample_idx >> 2) * 3;
    int pos = (int)(sample_idx & 3);
    const uint8_t* ptr = raw_stream + byte_offset;

    uint32_t chunk = (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8) | ((uint32_t)ptr[2] << 16);
    uint32_t raw6 = (chunk >> (pos * 6)) & 0x3F;

    // Sign extension from bit 5: if raw6 >= 32, value is raw6 - 64
    int q = (raw6 & 0x20) ? ((int)raw6 - 64) : (int)raw6;
    return (float)q;
}

// Unpack 4 samples at once when thread has byte-aligned chunk access
__device__ inline void unpack_4_samples_6bit(const uint8_t* __restrict__ chunk_ptr, float* out4)
{
    uint32_t chunk = (uint32_t)chunk_ptr[0] | ((uint32_t)chunk_ptr[1] << 8) | ((uint32_t)chunk_ptr[2] << 16);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t raw6 = (chunk >> (i * 6)) & 0x3F;
        int q = (raw6 & 0x20) ? ((int)raw6 - 64) : (int)raw6;
        out4[i] = (float)q;
    }
}

} // namespace tpgs
