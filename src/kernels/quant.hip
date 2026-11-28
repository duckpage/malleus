#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// Randomized Hadamard on 128-wide blocks: out = H (signs * x) / sqrt(128). One 128-thread block per chunk.
extern "C" __global__ void __launch_bounds__(128) hadamard128(float* out, const float* x, const int8_t* signs, int K) {
    __shared__ float s[128];
    const size_t base = (size_t)blockIdx.x * 128;
    const int i = threadIdx.x;
    const int col = (int)(base % K) + i;
    s[i] = x[base + i] * (float)signs[col % 128];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        const int j = i ^ h;
        const float a = s[i], b = s[j];
        __syncthreads();
        s[i] = (i & h) ? b - a : a + b;
        __syncthreads();
    }
    out[base + i] = s[i] * 0.08838834764831845f;
}

// MLS4 GEMV/GEMM. q: [N, K/2] bytes (low nibble = even k), int4 two's complement; sc: [N, K/128] fp16.
// xr: rotated activations [T, K]. Each warp handles R rows; lanes cover 32 consecutive weights per step.
template <int R, int TT>
__device__ __forceinline__ void mls4_core(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N,
                                          int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * R;
    const int t0 = blockIdx.y * TT;
    if (n0 >= N) return;
    const int G = K / 128;
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
            const uint4 v = *(const uint4*)(q + (size_t)n * (K / 2) + k / 2);
            const float s = __half2float(sc[(size_t)n * G + k / 128]);
            const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) wf[r][i * 8 + j] = (float)(((int32_t)(w4[i] << (28 - 4 * j))) >> 28) * s;
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
                    acc[a][r] += xv.x * wf[r][c * 4] + xv.y * wf[r][c * 4 + 1] + xv.z * wf[r][c * 4 + 2] + xv.w * wf[r][c * 4 + 3];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = warp_sum(acc[a][r]);
            if (lane == 0 && t0 + a < T && n0 + r < N) {
                float* o = out + (size_t)(t0 + a) * N + n0 + r;
                *o = accumulate ? *o + s : s;
            }
        }
}

extern "C" __global__ void gemv_mls4(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N, int K,
                                     int accumulate) {
    mls4_core<4, 1>(out, xr, q, sc, T, N, K, accumulate);
}

extern "C" __global__ void gemm_mls4(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N, int K,
                                     int accumulate) {
    mls4_core<2, 8>(out, xr, q, sc, T, N, K, accumulate);
}

// MLS8: q [N, K] int8 on rotated input, sc [N] fp32 per row.
//
// No FMA contraction here: the compiler contracts the products of `<4, 1>` and of `<4, 2>` differently (33 fmac and
// 30 adds against 34 and 73 in the gfx1151 object), so the same row came out one ulp apart depending on how many
// tokens the block held, and a speculative verify did not reproduce the serial step. With the contraction off every
// instance does the same multiplications and additions in the order written.
template <int R, int TT>
__device__ __forceinline__ void mls8_core(float* out, const float* xr, const int8_t* q, const float* sc, int T, int N,
                                          int K, int accumulate) {
#pragma clang fp contract(off)
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * R;
    const int t0 = blockIdx.y * TT;
    if (n0 >= N) return;
    float acc[TT][R];
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
    for (int k = lane * 16; k < K; k += 512) {
        float wf[R][16];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r < N ? n0 + r : N - 1;
            const uint4 v = *(const uint4*)(q + (size_t)n * K + k);
            const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wf[r][i * 4 + j] = (float)(((int32_t)(w4[i] << (24 - 8 * j))) >> 24);
        }
#pragma unroll
        for (int a = 0; a < TT; ++a) {
            if (t0 + a >= T) break;
            const float4* xp = (const float4*)(xr + (size_t)(t0 + a) * K + k);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 xv = xp[c];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[a][r] += xv.x * wf[r][c * 4] + xv.y * wf[r][c * 4 + 1] + xv.z * wf[r][c * 4 + 2] + xv.w * wf[r][c * 4 + 3];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = warp_sum(acc[a][r]);
            if (lane == 0 && t0 + a < T && n0 + r < N) {
                float* o = out + (size_t)(t0 + a) * N + n0 + r;
                const float v = s * sc[n0 + r];
                *o = accumulate ? *o + v : v;
            }
        }
}

extern "C" __global__ void gemv_mls8(float* out, const float* xr, const int8_t* q, const float* sc, int T, int N, int K,
                                     int accumulate) {
    mls8_core<4, 1>(out, xr, q, sc, T, N, K, accumulate);
}

extern "C" __global__ void gemm_mls8(float* out, const float* xr, const int8_t* q, const float* sc, int T, int N, int K,
                                     int accumulate) {
    mls8_core<2, 8>(out, xr, q, sc, T, N, K, accumulate);
}

typedef unsigned int q_u2 __attribute__((ext_vector_type(2)));
typedef unsigned int q_u4 __attribute__((ext_vector_type(4)));
typedef float q_f4 __attribute__((ext_vector_type(4)));

// MLS8 for up to TT tokens (the lm_head and the PLE projections in a speculative verify), each weight read once.
//
// It is `mls8_core` with all the tokens in one block, the template `gemv_mls8` decodes a single token with: a
// verify row then gets, bit for bit, what the same token gets when it is decoded alone. The kernel this replaces
// added the sixteen products of a step one at a time where `mls8_core` adds them four at a time, and that
// rounding was enough for the generated text to change with the drafts (and writing the same expression in a
// second kernel was not enough: the compiler contracted it differently).
#define GEMV8_TN(TT)                                                                                                  \
    extern "C" __global__ void gemv_mls8_t##TT(float* out, const float* xr, const int8_t* q, const float* sc, int T,  \
                                               int N, int K, int accumulate) {                                        \
        mls8_core<4, TT>(out, xr, q, sc, T, N, K, accumulate);                                                        \
    }
GEMV8_TN(2)
GEMV8_TN(3)
GEMV8_TN(4)
GEMV8_TN(5)
GEMV8_TN(6)

typedef _Float16 q_h2 __attribute__((ext_vector_type(2)));

// MLS4 for up to TT tokens (decode, speculative verify), each weight read once: warp = 4 rows, lane = 32 weights per
// 1024-wide K chunk (K a multiple of 32: lanes past K idle in the last chunk). The chunk of every token is converted
// once per block to LDS as fp16 pairs (j, j + 4) of each
// 8-weight word; nibble pairs become exact fp16 integers via 0x6400 | (n ^ 8) - 1032 and go through packed fp16 dot
// products with fp32 accumulation.
// Segmented outputs (o1 != null): rows [0, b1) go to out, [b1, b2) to o1, [b2, b3) to o2, [b3, N) to o3, each
// [T, rows of the segment] (projections sharing an input, merged into one matrix).
// rs != null: xr is not rotated yet; the randomized Hadamard of every 128-block (signs rs) is applied while staging
// (16 lanes per block: 8 elements each, then 4 shuffle stages), saving the separate rotation kernel.
template <int TT, int R, bool EXT = false>
__device__ __forceinline__ void gemv4_hl(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N,
                                         int K, int accumulate, float* o1 = nullptr, float* o2 = nullptr,
                                         float* o3 = nullptr, int b1 = 0, int b2 = 0, int b3 = 0,
                                         const int8_t* rs = nullptr, const int* xrow = nullptr,
                                         const int* orow = nullptr, q_u4* xs_ext = nullptr) {
    // EXT: the staging buffer is the caller's, so a kernel that picks TT at run time pays for one buffer, not for
    // the sum of all its instantiations.
    q_u4 (*xs)[128];
    if constexpr (EXT) {
        xs = (q_u4 (*)[128])xs_ext;
    } else {
        __shared__ __attribute__((aligned(16))) q_u4 xs_s[TT][128];
        xs = xs_s;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * R;
    const int G = K / 128;
    float acc[TT][R];
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
    const q_h2 bias = {(_Float16)1032.f, (_Float16)1032.f};
    for (int kc = 0; kc < K; kc += 1024) {
        __syncthreads();
        for (int it = threadIdx.x; it < TT * 128; it += 256) {
            const int a = it / 128, w = it % 128;
            const bool in = kc + 8 * w < K;
            // xrow: token a's input is row xrow[a] of xr (the rows of one expert, gathered without a copy).
            const int xa = xrow ? xrow[min(a, T - 1)] : min(a, T - 1);
            const q_f4* xp = (const q_f4*)(xr + (size_t)xa * K + min(kc + 8 * w, K - 8));
            q_f4 lo = xp[0], hi = xp[1];
            if (rs) {
                // Lanes 16c .. 16c + 15 hold block c (w % 16 == lane % 16: 256 and 128 are multiples of 32).
                const int e0 = 8 * (w % 16);
                float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) v[j] *= (float)rs[e0 + j];
#pragma unroll
                for (int h = 1; h < 8; h <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if (!(j & h)) {
                            const float x0 = v[j], x1 = v[j + h];
                            v[j] = x0 + x1;
                            v[j + h] = x0 - x1;
                        }
#pragma unroll
                for (int m = 1; m < 16; m <<= 1) {
                    const bool upper = w & m;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float o = __shfl_xor(v[j], m, 32);
                        v[j] = upper ? o - v[j] : v[j] + o;
                    }
                }
                constexpr float S = 0.08838834764831845f;
                lo = (q_f4){v[0] * S, v[1] * S, v[2] * S, v[3] * S};
                hi = (q_f4){v[4] * S, v[5] * S, v[6] * S, v[7] * S};
            }
            if (!in) continue;
            const q_h2 p0 = {(_Float16)lo.x, (_Float16)hi.x}, p1 = {(_Float16)lo.y, (_Float16)hi.y};
            const q_h2 p2 = {(_Float16)lo.z, (_Float16)hi.z}, p3 = {(_Float16)lo.w, (_Float16)hi.w};
            q_u4 u;
            u.x = __builtin_bit_cast(unsigned, p0);
            u.y = __builtin_bit_cast(unsigned, p1);
            u.z = __builtin_bit_cast(unsigned, p2);
            u.w = __builtin_bit_cast(unsigned, p3);
            xs[a][w] = u;
        }
        __syncthreads();
        const int k = kc + lane * 32;
        if (n0 >= N || k >= K) continue;
        q_u4 v[R];
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = min(n0 + r, N - 1);
            v[r] = *(const q_u4*)(q + (size_t)n * (K / 2) + k / 2);
            s[r] = __half2float(sc[(size_t)n * G + k / 128]);
        }
        float part[TT][R];
#pragma unroll
        for (int a = 0; a < TT; ++a)
#pragma unroll
            for (int r = 0; r < R; ++r) part[a][r] = 0.f;
        // The weights of this chunk stay in registers (`wh`) and the token words are read one token at a time: with
        // `xw[TT]` held for the whole loop the register count grew by TT * 4, which at TT = 16 pushed the kernel
        // into a spill and cost 13 % per verified token. Each `acc[a][r]` sums over (i, p) in the same order as
        // before, so the result is bit-identical; only the loops that were already independent are reordered.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q_h2 wh[R][4];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const unsigned w = v[r][i] ^ 0x88888888u;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned bits = ((w >> (4 * p)) & 0x000F000Fu) | 0x64006400u;
                    wh[r][p] = __builtin_bit_cast(q_h2, bits) - bias;
                }
            }
#pragma unroll
            for (int a = 0; a < TT; ++a) {
                const q_u4 xw = xs[a][lane * 4 + i];
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        // bit_cast of a vector element reads element 0: go through a scalar.
                        const unsigned xp = xw[p];
                        part[a][r] =
                            __builtin_amdgcn_fdot2(wh[r][p], __builtin_bit_cast(q_h2, xp), part[a][r], false);
                    }
            }
        }
#pragma unroll
        for (int a = 0; a < TT; ++a)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[a][r] += part[a][r] * s[r];
    }
    if (n0 >= N) return;
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_sum(acc[a][r]);
            if (lane == 0 && a < T && n0 + r < N) {
                const int n = n0 + r;
                float* o;
                if (!o1) o = out + (size_t)(orow ? orow[a] : a) * N + n;
                else if (n < b1) o = out + (size_t)a * b1 + n;
                else if (n < b2) o = o1 + (size_t)a * (b2 - b1) + n - b1;
                else if (n < b3) o = o2 + (size_t)a * (b3 - b2) + n - b2;
                else o = o3 + (size_t)a * (N - b3) + n - b3;
                *o = accumulate ? *o + v : v;
            }
        }
}

#define GEMV4_HL(TT)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        gemv_mls4_l##TT(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N, int K, int acc, \
                        const int8_t* rs) {                                                                           \
        gemv4_hl<TT, 4>(out, xr, q, sc, T, N, K, acc, nullptr, nullptr, nullptr, 0, 0, 0, rs);                        \
    }                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        gemv_mls4_m##TT(float* out, float* o1, float* o2, float* o3, int b1, int b2, int b3, const float* xr,         \
                        const uint8_t* q, const __half* sc, int T, int N, int K, const int8_t* rs) {                  \
        gemv4_hl<TT, 4>(out, xr, q, sc, T, N, K, 0, o1, o2, o3, b1, b2, b3, rs);                                      \
    }
GEMV4_HL(1)
extern "C" __global__ void __launch_bounds__(256) gemv_mls4_r1(float* out, const float* xr, const uint8_t* q,
                                                              const __half* sc, int T, int N, int K, int acc,
                                                              const int8_t* rs) {
    gemv4_hl<1, 1>(out, xr, q, sc, T, N, K, acc, nullptr, nullptr, nullptr, 0, 0, 0, rs);
}
// One row per warp for TT tokens (a verify on few rows, N <= 4096): with four rows per warp a 3072-row matrix gave
// the GPU 96 blocks and the verify of six ran at half the speed of a single r1 step.
// RROWS rows per warp: 1 starves the LDS (six tokens' inputs read per 512 bytes of weights), 4 starves the GPU.
#define RROWS 2
#define GEMV4_R(TT)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        gemv_mls4_r##TT(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N, int K, int acc, \
                        const int8_t* rs) {                                                                           \
        gemv4_hl<TT, RROWS>(out, xr, q, sc, T, N, K, acc, nullptr, nullptr, nullptr, 0, 0, 0, rs);                    \
    }
GEMV4_R(2)
GEMV4_R(3)
GEMV4_R(4)
GEMV4_R(5)
GEMV4_R(6)
GEMV4_HL(2)
GEMV4_HL(3)
GEMV4_HL(4)
GEMV4_HL(5)
GEMV4_HL(6)

// MoE router for one token per block: top-k of the E logits, weights softmax-renormalized over the k chosen (equal to
// softmax, top-k, divide by the sum). Slot k is the shared expert (id E) weighted by sigmoid(logit E).
extern "C" __global__ void __launch_bounds__(256) moe_route(int* ids, float* w, const float* logits, int E, int topk) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int chosen[64];
    const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* l = logits + (size_t)t * (E + 1);
    for (int j = 0; j < topk; ++j) {
        float v = -INFINITY;
        int idx = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += 256) {
            bool taken = false;
            for (int c = 0; c < j; ++c) taken |= chosen[c] == e;
            const float x = l[e];
            if (!taken && (x > v || (x == v && e < idx))) v = x, idx = e;
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(idx, o, 32);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (lane == 0) bv[warp] = v, bi[warp] = idx;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < 8; ++q)
                if (bv[q] > bv[0] || (bv[q] == bv[0] && bi[q] < bi[0])) bv[0] = bv[q], bi[0] = bi[q];
            chosen[j] = bi[0];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float m = l[chosen[0]];
        float s = 0.f;
        for (int j = 0; j < topk; ++j) s += expf(l[chosen[j]] - m);
        for (int j = 0; j < topk; ++j) {
            ids[t * (topk + 1) + j] = chosen[j];
            w[t * (topk + 1) + j] = expf(l[chosen[j]] - m) / s;
        }
        ids[t * (topk + 1) + topk] = E;
        w[t * (topk + 1) + topk] = 1.f / (1.f + expf(-l[E]));
    }
}

// moe_route with one wave per token (E <= 32 * 32, topk <= 32): every lane keeps E / 32 logits, each round takes the
// wave maximum (lowest index on ties) without barriers.
extern "C" __global__ void __launch_bounds__(32) moe_route_w(int* ids, float* w, const float* logits, int E, int topk) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* l = logits + (size_t)t * (E + 1);
    float v[32];
    const int per = (E + 31) / 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = i < per && lane + 32 * i < E ? l[lane + 32 * i] : -INFINITY;
    float m0 = 0.f, sum = 0.f;
    int chosen[32];
    float cv[32];
    for (int j = 0; j < topk; ++j) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (v[i] > bv) bv = v[i], bi = lane + 32 * i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if ((bi & 31) == lane) {
#pragma unroll
            for (int i = 0; i < 32; ++i)
                if (lane + 32 * i == bi) v[i] = -INFINITY;
        }
        if (j == 0) m0 = bv;
        chosen[j] = bi;
        cv[j] = bv;
        sum += expf(bv - m0);
    }
    if (lane == 0) {
        for (int j = 0; j < topk; ++j) {
            ids[t * (topk + 1) + j] = chosen[j];
            w[t * (topk + 1) + j] = expf(cv[j] - m0) / sum;
        }
        ids[t * (topk + 1) + topk] = E;
        w[t * (topk + 1) + topk] = 1.f / (1.f + expf(-l[E]));
    }
}

// Ling-3.0-flash router: sigmoid scores, group top-k with an additive bias (`topk_method: noaux_tc`).
// One block of 256 threads per token, same output geometry as moe_route: topk + 1 slots, the last one the shared
// expert. The two halves of the rule are the part that is easy to get wrong and the reason this is a tested kernel:
// the groups and the experts are chosen on `sr = sigmoid(logit) + expert_bias`, but the weights come from the
// plain `s = sigmoid(logit)` without the bias, renormalized over the chosen ones and multiplied by `scaling`.
// The shared expert of Ling has no gate at all: it is added with weight 1 (docs/model-ling3-flash.md, § MoE).
extern "C" __global__ void __launch_bounds__(256) moe_route_g(int* ids, float* w, const float* logits,
                                                              const float* bias, int E, int topk, int ng, int tg,
                                                              float scaling, int ld, int shared) {
    __shared__ float sr[512], sv[512], gs[8];
    __shared__ int gsel[8], chosen[32];
    const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    // `ld` is the row stride of the logits (E + 1 where the router has a shared-expert gate next to them, E where it
    // has none) and `shared` writes that extra slot (weight 1, id E) after the routed ones.
    const float* l = logits + (size_t)t * ld;
    for (int e = threadIdx.x; e < E; e += 256) {
        const float s = 1.f / (1.f + expf(-l[e]));
        sv[e] = s;
        sr[e] = s + bias[e];
    }
    __syncthreads();
    // Group score: the sum of the two best of each group of E / ng experts, one warp per group (ng <= 8).
    const int per = E / ng;
    if (warp < ng) {
        const int base = warp * per;
        float m1 = -INFINITY, m2 = -INFINITY;
        for (int i = lane; i < per; i += 32) {
            const float v = sr[base + i];
            if (v > m1) {
                m2 = m1;
                m1 = v;
            } else if (v > m2) {
                m2 = v;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float a = __shfl_xor(m1, o, 32), b = __shfl_xor(m2, o, 32);
            // Merging two (max, second max) pairs: the new max is the larger max, the new second is the largest of
            // what is left.
            if (a > m1) {
                m2 = fmaxf(m1, b);
                m1 = a;
            } else {
                m2 = fmaxf(m2, a);
            }
        }
        if (lane == 0) gs[warp] = m1 + m2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // The tg groups with the highest score; a tie goes to the lower index, as the expert tie-break does.
        for (int q = 0; q < 8; ++q) gsel[q] = 0;
        for (int g = 0; g < tg; ++g) {
            int best = 0;
            float bv = -INFINITY;
            for (int q = 0; q < ng; ++q) {
                const int taken = gsel[q] == 1;
                if (!taken && (gs[q] > bv || (gs[q] == bv && q < best))) bv = gs[q], best = q;
            }
            gsel[best] = 1;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += 256) {
        if (!gsel[e / per]) sr[e] = -INFINITY;
    }
    __syncthreads();
    // topk experts by sr, lowest index on ties, exactly the selection rule of moe_route.
    __shared__ float bv8[8];
    __shared__ int bi8[8];
    for (int j = 0; j < topk; ++j) {
        float v = -INFINITY;
        int idx = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += 256) {
            bool taken = false;
            for (int c = 0; c < j; ++c) taken |= chosen[c] == e;
            const float x = sr[e];
            if (!taken && (x > v || (x == v && e < idx))) v = x, idx = e;
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(idx, o, 32);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (lane == 0) bv8[warp] = v, bi8[warp] = idx;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < 8; ++q) {
                if (bv8[q] > bv8[0] || (bv8[q] == bv8[0] && bi8[q] < bi8[0])) bv8[0] = bv8[q], bi8[0] = bi8[q];
            }
            chosen[j] = bi8[0];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Weights from the scores without the bias, renormalized over what was chosen, then scaled.
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) sum += sv[chosen[j]];
        const int stride = shared ? topk + 1 : topk;
        for (int j = 0; j < topk; ++j) {
            ids[t * stride + j] = chosen[j];
            w[t * stride + j] = sv[chosen[j]] / (sum + 1e-20f) * scaling;
        }
        if (shared) {
            ids[t * stride + topk] = E;
            w[t * stride + topk] = 1.f;
        }
    }
}

// gate|up rows of the expert of each (token, slot) pair: q [E + 1, N, K / 2], one pair per blockIdx.y.
extern "C" __global__ void __launch_bounds__(256) moe_gate_up(float* out, const float* xr, const uint8_t* q,
                                                              const __half* sc, const int* ids, int slots, int N, int K,
                                                              const int8_t* rs) {
    const int p = blockIdx.y;
    const size_t e = ids[p];
    gemv4_hl<1, 4>(out + (size_t)p * N, xr + (size_t)(p / slots) * K, q + e * N * (K / 2), sc + e * N * (K / 128), 1, N,
                   K, 0, nullptr, nullptr, nullptr, 0, 0, 0, rs);
}

// silu(gate) * up of each pair (gu rows [gate I | up I]), then the randomized Hadamard of every 128-block.
// silu(gate) * up, rotated for the down projection (the expert weights are in the rotated basis, so the
// butterfly here and the signs of the quantizer are the same transform). `lim` is the per-layer SwiGLU clamp of
// ADR 0004 — silu(gate).clamp(max = L) * up.clamp(-L, L) — and 0 means no clamp, which is every Qwen layer and
// the first Ling layers.
extern "C" __global__ void __launch_bounds__(128) moe_act(float* out, const float* gu, const int8_t* signs, int I,
                                                          float lim) {
    __shared__ float s[128];
    const int nb = I / 128, p = blockIdx.x / nb, b = blockIdx.x % nb, i = threadIdx.x;
    const float g = gu[(size_t)p * 2 * I + b * 128 + i], u0 = gu[(size_t)p * 2 * I + I + b * 128 + i];
    const float gs = lim > 0.f ? fminf(g, lim) : g;
    const float u = lim > 0.f ? fminf(fmaxf(u0, -lim), lim) : u0;
    s[i] = gs / (1.f + expf(-gs)) * u * (float)signs[i];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        const int j = i ^ h;
        const float a = s[i], c = s[j];
        __syncthreads();
        s[i] = (i & h) ? c - a : a + c;
        __syncthreads();
    }
    out[(size_t)p * I + b * 128 + i] = s[i] * 0.08838834764831845f;
}

// out[t, n] += Σ_s w[t, s] · down_{ids[t, s]}[n] · a[t, s]. q [E + 1, N, K / 2], K a multiple of 512; warp = 4 rows,
// lane = 16 weights per 512-wide step.
extern "C" __global__ void __launch_bounds__(256) moe_down(float* out, const float* a, const uint8_t* q, const __half* sc,
                                                           const int* ids, const float* w, int slots, int N, int K,
                                                           int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * 4, t = blockIdx.y;
    if (n0 >= N) return;
    const int G = K / 128;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < slots; ++s) {
        const size_t e = ids[t * slots + s];
        const uint8_t* qe = q + e * N * (K / 2);
        const __half* se = sc + e * N * G;
        const float* x = a + (size_t)(t * slots + s) * K;
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k = lane * 16; k < K; k += 512) {
            float xv[16];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const q_f4 v = *(const q_f4*)(x + k + 4 * c);
                xv[4 * c] = v.x, xv[4 * c + 1] = v.y, xv[4 * c + 2] = v.z, xv[4 * c + 3] = v.w;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int n = min(n0 + r, N - 1);
                const q_u2 v = *(const q_u2*)(qe + (size_t)n * (K / 2) + k / 2);
                float d = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const unsigned wd = j < 8 ? v.x : v.y;
                    d += xv[j] * (float)(((int)(wd << (28 - 4 * (j % 8)))) >> 28);
                }
                part[r] += d * __half2float(se[(size_t)n * G + k / 128]);
            }
        }
        const float ws = w[t * slots + s];
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[r] += ws * part[r];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0 && n0 + r < N) out[(size_t)t * N + n0 + r] = accumulate ? out[(size_t)t * N + n0 + r] + v : v;
    }
}

// Groups the P (token, slot) pairs by expert (one 1024-thread block): offs[E1 + 1] row offsets, perm[sorted] = pair,
// pos[pair] = sorted row, tiles = (expert, first row) of every BM-row tile, ntiles their count. E1 <= 4096.
// BM2 > 0: only full BM-row tiles go to `tiles`; the rest of each expert goes in BM2-row tiles at tiles + 2 * OFF2,
// counted in ntiles[1] (a kernel with smaller tiles takes them: less padding).
extern "C" __global__ void __launch_bounds__(1024) moe_sort(int* offs, int* perm, int* pos, int* tiles, int* ntiles,
                                                            const int* ids, int P, int E1, int BM, int BM2 = 0,
                                                            int OFF2 = 0) {
    __shared__ int cnt[4096], start[4096];
    for (int e = threadIdx.x; e < E1; e += 1024) cnt[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += 1024) atomicAdd(&cnt[ids[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, n = 0, n2 = 0;
        int* t2 = tiles + 2 * OFF2;
        for (int e = 0; e < E1; ++e) {
            start[e] = o;
            offs[e] = o;
            int r = 0;
            if (BM2 > 0) {
                for (; r + BM <= cnt[e]; r += BM) tiles[2 * n] = e, tiles[2 * n + 1] = r, ++n;
                for (; r < cnt[e]; r += BM2) t2[2 * n2] = e, t2[2 * n2 + 1] = r, ++n2;
            } else {
                for (; r < cnt[e]; r += BM) tiles[2 * n] = e, tiles[2 * n + 1] = r, ++n;
            }
            o += cnt[e];
            cnt[e] = 0;
        }
        offs[E1] = o;
        ntiles[0] = n;
        // Both counters are written in both modes: the grouped GEMM launches its second list unconditionally and
        // bounds it with `ntiles[1]`, so leaving it untouched means the caller reads whatever the buffer held before
        // — a recycled buffer holds the previous request's count, and the tiles that follow are its tiles (issue
        // #167: a GPU page fault in moe_w4a8 on the second of two prefills).
        ntiles[1] = (BM2 > 0) ? n2 : 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += 1024) {
        const int e = ids[p], r = start[e] + atomicAdd(&cnt[e], 1);
        perm[r] = p;
        pos[p] = r;
    }
}

// xg[r] = xq[perm[r] / slots] (int8 rows of K bytes, K multiple of 16) and its row scale.
extern "C" __global__ void __launch_bounds__(128) moe_gather(int8_t* xg, float* sg, const int8_t* xq, const float* xs,
                                                             const int* perm, int slots, int K) {
    const int r = blockIdx.x, t = perm[r] / slots;
    for (int k = threadIdx.x * 16; k < K; k += 128 * 16)
        *(q_u4*)(xg + (size_t)r * K + k) = *(const q_u4*)(xq + (size_t)t * K + k);
    if (threadIdx.x == 0) sg[r] = xs[t];
}

// x[t, n] += Σ_s w[t, s] · y[pos[t, s], n].
// x[t] += sum_s w[t, s] * y[pos[t, s]]: one block per token and 1024 columns, four columns per thread so the
// rows of `y` (the expert outputs) are read as float4. The routing scalars stay in registers instead of being
// re-read for every column: this is a pure reduction over `slots` rows of N f32, and it was measuring ~80 GB/s
// (quant.hip), which is a third of what the machine streams.
extern "C" __global__ void __launch_bounds__(256) moe_combine(float* x, const float* y, const float* w, const int* pos,
                                                              int slots, int N, int accumulate = 1) {
    const int t = blockIdx.x;
    const int n = (blockIdx.y * 256 + threadIdx.x) * 4;
    if (n >= N) return;
    const int* ps = pos + (size_t)t * slots;
    const float* ws = w + (size_t)t * slots;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < slots; ++s) {
        const float g = ws[s];
        const float4 v = *(const float4*)(y + (size_t)ps[s] * N + n);
        acc.x += g * v.x;
        acc.y += g * v.y;
        acc.z += g * v.z;
        acc.w += g * v.w;
    }
    float4* o = (float4*)(x + (size_t)t * N + n);
    if (accumulate) {
        const float4 d = *o;
        acc.x += d.x;
        acc.y += d.y;
        acc.z += d.z;
        acc.w += d.w;
    }
    *o = acc;
}

// Same reduction with `y` in fp16 (MALLEUS_MOE_Y_HALF): the intermediate is written once and read once per layer and
// is the widest buffer of the prefill, so its bytes are what this kernel is limited by. Four columns per thread stay
// one 8-byte load.
extern "C" __global__ void __launch_bounds__(256) moe_combine_yh(float* x, const _Float16* y, const float* w,
                                                               const int* pos, int slots, int N, int accumulate = 1) {
    const int t = blockIdx.x;
    const int n = (blockIdx.y * 256 + threadIdx.x) * 4;
    if (n >= N) return;
    const int* ps = pos + (size_t)t * slots;
    const float* ws = w + (size_t)t * slots;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < slots; ++s) {
        const float g = ws[s];
        const uint2 raw = *(const uint2*)(y + (size_t)ps[s] * N + n);
        const _Float16* v = (const _Float16*)&raw;
        acc.x += g * (float)v[0];
        acc.y += g * (float)v[1];
        acc.z += g * (float)v[2];
        acc.w += g * (float)v[3];
    }
    float4* o = (float4*)(x + (size_t)t * N + n);
    if (accumulate) {
        const float4 d = *o;
        acc.x += d.x;
        acc.y += d.y;
        acc.z += d.z;
        acc.w += d.w;
    }
    *o = acc;
}

// Routed experts of a few tokens (a decode step, a speculative verify), each used expert read once: `tiles` holds
// the experts `moe_sort` found (one tile of up to 6 rows each, blockIdx.y), `perm` the sorted (token, slot) pairs.
// The rows of an expert go through gemv4_hl<rows, 4> with their input rows gathered by index, and every row sums in
// the same order as a single-row GEMV would: a verify row equals the serial step bit for bit, and the expert's
// weights cross the bus once instead of once per token that picked it.
//   gate|up: out [P, N] from x [T, K] (unrotated; rs folds the rotation), row of pair p = p / slots
//   down:    out [P, N] (expert-sorted rows) from x [P, K] (already rotated by moe_act)
template <bool GU, int MAXT>
__device__ __forceinline__ void moe_grp_core(float* out, const float* x, const uint8_t* q, const __half* sc,
                                             const int* offs, const int* perm, const int* tiles, const int* ntiles,
                                             int slots, int N, int K, const int8_t* rs) {
    __shared__ int xrow[8], orow[8];
    const int ti = blockIdx.y;
    if (ti >= ntiles[0]) return;
    const int e = tiles[2 * ti], r0 = tiles[2 * ti + 1];
    const int rows = min(MAXT, offs[e + 1] - offs[e] - r0);
    if (threadIdx.x < 8) {
        const int pr = perm[offs[e] + r0 + min((int)threadIdx.x, rows - 1)];
        // gate|up: one output row per pair; down: the expert-sorted row `moe_combine` finds through `pos`.
        xrow[threadIdx.x] = GU ? pr / slots : pr;
        orow[threadIdx.x] = GU ? pr : offs[e] + r0 + min((int)threadIdx.x, rows - 1);
    }
    __syncthreads();
    // Four rows per warp (one per warp measured slower here: 8 rows a block is too little work per staging) and the instantiation
    // for this expert's row count, staging in one shared buffer: the per-row sums depend on neither, so a verify row
    // equals the single-row step bit for bit, and an expert one token picked computes one row.
    __shared__ __attribute__((aligned(16))) q_u4 xbuf[MAXT][128];
    const uint8_t* qe = q + (size_t)e * N * (K / 2);
    const __half* se = sc + (size_t)e * N * (K / 128);
    const int8_t* r = GU ? rs : nullptr;
#define MOE_GRP(TT)                                                                                                   \
    gemv4_hl<TT, 4, true>(out, x, qe, se, rows, N, K, 0, nullptr, nullptr, nullptr, 0, 0, 0, r, xrow, orow, &xbuf[0][0])
    if constexpr (MAXT == 1) {
        MOE_GRP(1);
    } else {
        switch (rows) {
            case 1: MOE_GRP(1); break;
            case 2: MOE_GRP(2); break;
            case 3: MOE_GRP(3); break;
            case 4: MOE_GRP(4); break;
            case 5: MOE_GRP(5); break;
            default: MOE_GRP(6); break;
        }
    }
#undef MOE_GRP
}
#define MOE_GRP_KERNELS(SUF, MAXT)                                                                                     \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        moe_gate_up_grp##SUF(float* out, const float* x, const uint8_t* q, const __half* sc, const int* offs,          \
                             const int* perm, const int* tiles, const int* ntiles, int slots, int N, int K,            \
                             const int8_t* rs) {                                                                      \
        moe_grp_core<true, MAXT>(out, x, q, sc, offs, perm, tiles, ntiles, slots, N, K, rs);                          \
    }                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        moe_down_grp##SUF(float* out, const float* x, const uint8_t* q, const __half* sc, const int* offs,             \
                          const int* perm, const int* tiles, const int* ntiles, int N, int K) {                       \
        moe_grp_core<false, MAXT>(out, x, q, sc, offs, perm, tiles, ntiles, 1, N, K, nullptr);                        \
    }
// grp1: a decode step (every expert has one row); grp: a verify (up to 6 rows an expert). Same arithmetic per row.
MOE_GRP_KERNELS(1, 1)
MOE_GRP_KERNELS(, 6)
