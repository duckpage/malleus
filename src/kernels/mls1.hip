// MLS1 GEMV/GEMM: one bit per weight, one e4m3 scale per 32.
//
// q:  [N, K / 8] bytes, eight signs per byte, bit i of the row is the sign of weight k (set = negative).
// sc: [N, K / 32] e4m3 bytes, one scale per 32 weights — the amplitude is all a sign-only code carries, so it is
//     tracked where it moves rather than once per 128 like MLS2/3/4.
// xr: rotated activations [T, K] (the same Hadamard the converter applied to the rows).
//
// A lane owns 32 consecutive weights: four bytes of codes, read as one dword at `lane * 4`, and one scale byte. The
// row is K / 8 bytes, a multiple of 4 for every K the engine stores (a multiple of 128), so the dword never crosses
// into the next lane's codes.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float mls1_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

/// The e4m3 (bias 7, 3 mantissa bits) scale of one group, decoded without a hardware cast.
__device__ __forceinline__ float mls1_scale(uint8_t b) {
    const int e = (b & 0x7f) >> 3, m = b & 7;
    const float v = e == 0 ? (float)m * (1.0f / 512.0f) : (1.0f + (float)m / 8.0f) * exp2f((float)(e - 7));
    return (b & 0x80) ? -v : v;
}

template <int R, int TT>
__device__ __forceinline__ void mls1_core(float* out, const float* xr, const uint8_t* q, const uint8_t* sc, int T,
                                          int N, int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * R;
    const int t0 = blockIdx.y * TT;
    if (n0 >= N) return;
    const int G = K / 32;
    const size_t row_bytes = (size_t)K / 8;
    float acc[TT][R];
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
    for (int k = lane * 32; k < K; k += 1024) {
        float wf[R][32];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r < N ? n0 + r : N - 1;
            const uint32_t bits = *(const uint32_t*)(q + (size_t)n * row_bytes + (size_t)k / 8);
            const float s = mls1_scale(sc[(size_t)n * G + k / 32]);
#pragma unroll
            for (int i = 0; i < 32; ++i) wf[r][i] = ((bits >> i) & 1u) ? -s : s;
        }
#pragma unroll
        for (int a = 0; a < TT; ++a) {
            if (t0 + a >= T) break;
            const float4* xp = (const float4*)(xr + (size_t)(t0 + a) * K + k);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float4 xv = xp[c];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[a][r] += xv.x * wf[r][c * 4] + xv.y * wf[r][c * 4 + 1] + xv.z * wf[r][c * 4 + 2] +
                                 xv.w * wf[r][c * 4 + 3];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = mls1_warp_sum(acc[a][r]);
            if (lane == 0 && t0 + a < T && n0 + r < N) {
                float* o = out + (size_t)(t0 + a) * N + n0 + r;
                *o = accumulate ? *o + s : s;
            }
        }
}

/// One token: grid (ceil(N / 32), 1, 1), block 256.
extern "C" __global__ void gemv_mls1(float* out, const float* xr, const uint8_t* q, const uint8_t* sc, int T, int N,
                                     int K, int accumulate) {
    mls1_core<4, 1>(out, xr, q, sc, T, N, K, accumulate);
}

/// Several tokens per block: grid (ceil(N / 16), ceil(T / 8), 1), block 256.
extern "C" __global__ void gemm_mls1(float* out, const float* xr, const uint8_t* q, const uint8_t* sc, int T, int N,
                                     int K, int accumulate) {
    mls1_core<2, 8>(out, xr, q, sc, T, N, K, accumulate);
}
