#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Inkling-2's short convolution over the key and value streams (issue #62). A causal depthwise conv of kernel 4,
// with the three past tokens carried in a state buffer so decode and prefill share one path. The layer's own
// convolution (`attn_sconv`, `mlp_sconv`) is kernel 4 as well and runs through the same code.
//
// Two variants, as the reference graph has them:
//   static   the taps are the same for every token:      y = (x + conv) / sqrt(5)
//   dynamic  the taps are the static ones plus a per-token, per-head projection:  y = (x + conv) / 3
// The divisors differ because the reference divides by sqrt(conv_kernel_size + 1) in the static variant and by
// sqrt(2 * conv_kernel_size + 1) in the dynamic one; the golden comparison is what settles it.
typedef unsigned short ik_bf16;

__device__ __forceinline__ float ik_bf2f(ik_bf16 b) {
    return __uint_as_float(((unsigned)b) << 16);
}

// y[t][c] = (x[t][c] + sum_j w[c][j] * x[t-3+j][c]) / div, with x[-1..-3] from `past` ([3, C], oldest first).
// w is [C, 4] bf16 in the checkpoint's [C, 1, 4] order; dyn_w is [T, heads * 4] f32, or null in the static variant.
extern "C" __global__ void __launch_bounds__(256) inkling2_sconv(float* y, const float* x, const ik_bf16* w,
                                                                const float* dyn_w, const float* past, int T, int C,
                                                                int heads, float div) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)T * C) return;
    const int t = (int)(idx / C), c = (int)(idx % C);
    const int h = c / (C / heads);
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const int p = t - lag;
        const float v = p >= 0 ? x[(size_t)p * C + c] : past[(size_t)(p + 3) * C + c];
        // The static variant's weight is [C, 4] with tap 0 oldest; the dynamic variant's static_w is [4, C] with
        // tap 0 the current token (the reference's `static_weight.T`). The lag runs 0 = current .. 3 = three back.
        float wj = dyn_w ? ik_bf2f(w[(size_t)lag * C + c]) : ik_bf2f(w[(size_t)c * 4 + (3 - lag)]);
        if (dyn_w) wj += dyn_w[(size_t)t * heads * 4 + (size_t)h * 4 + lag];
        acc += wj * v;
    }
    y[idx] = (x[idx] + acc) / div;
}

// The per-head transforms Inkling-2 applies before and around its attention (issue #62).
//
// A "row" below is one head vector: `dim` values of one (token, head). The reference scales by a learned per-(head,
// dim) scale, clamps the L2 norm to a threshold, and (full layers) multiplies by tau, the log-scaled position
// factor `1 + alpha * log(max(1, (pos + 1) / floor))`.

// y = x * min(1, thr / (|x| + 1e-6)) * scale[i], then * tau[t] when tau is not null. Row-major [T, heads, dim].
extern "C" __global__ void __launch_bounds__(128) inkling2_l2_clip(float* y, const float* x, const float* scale,
                                                                  const float* tau, int T, int heads, int dim,
                                                                  float thr) {
    __shared__ float ss[128];
    const int row = blockIdx.x;  // token * heads + head
    const int r = threadIdx.x;
    // Lanes past `dim` (the relative-bias stream is 16 wide) contribute nothing to the tree below and write
    // nothing back: returning early used to leave their `ss` slots to whatever the previous launch left there.
    const bool live = r < dim;
    const int t = row / heads;
    const float v = live ? x[(size_t)row * dim + r] * scale[(size_t)(row % heads) * dim + r] : 0.f;
    ss[r] = v * v;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (r < o) ss[r] += ss[r + o];
        __syncthreads();
    }
    const float norm = sqrtf(ss[0]);
    const float k = min(1.f, thr / (norm + 1e-6f)) * (tau ? tau[t] : 1.f);
    if (live) y[(size_t)row * dim + r] = v * k;
}

// Unweighted RMSNorm over the last dim: y = x / sqrt(mean(x^2) + eps).
extern "C" __global__ void __launch_bounds__(128) inkling2_rms_unnorm(float* y, const float* x, int rows, int dim,
                                                                     float eps) {
    __shared__ float ss[128];
    const int row = blockIdx.x;
    const int r = threadIdx.x;
    const bool live = r < dim;
    const float v = live ? x[(size_t)row * dim + r] : 0.f;
    ss[r] = v * v;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (r < o) ss[r] += ss[r + o];
        __syncthreads();
    }
    if (live) y[(size_t)row * dim + r] = v * rsqrtf(ss[0] / dim + eps);
}

// bias[head][q][s] = rel[q][head] . proj[:, s - q] when 0 <= q - s < rel_extent, else 0.
// rel is [T, heads, d_rel], proj is [d_rel, rel_extent] row-major, bias is [heads, T, T].
extern "C" __global__ void __launch_bounds__(256) inkling2_rel_bias(float* bias, const float* rel, const float* proj,
                                                                   int T, int heads, int d_rel, int extent) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)heads * T * T) return;
    const int h = (int)(idx / ((long long)T * T));
    const int q = (int)((idx % ((long long)T * T)) / T);
    const int s = (int)(idx % T);
    const int d = q - s;
    float acc = 0.f;
    if (d >= 0 && d < extent) {
        const float* rp = rel + ((size_t)q * heads + h) * d_rel;
        for (int i = 0; i < d_rel; ++i) acc += rp[i] * proj[(size_t)i * extent + d];
    }
    bias[idx] = acc;
}

// Causal attention with the relative bias, the learned sink and the output gate: one block per (token, head).
// q, k, v, out and gate are [T, heads|nkv, dim] (kh = h / (heads / nkv) for k and v); bias is [heads, T, T]; `sink`
// is [T, heads] or null. The window is the number of keys a token sees (0 = all of them, the full layers), and the
// scores live in shared memory, so a full layer with a long context needs the tiling of `attention_split` — this
// kernel is the correctness path used to check the pieces, not the long-context one.
extern "C" __global__ void __launch_bounds__(128) inkling2_attn(float* out, const float* q, const float* k,
                                                                const float* v, const float* bias, const float* sink,
                                                                const float* gate, int T, int heads, int nkv, int dim,
                                                                int window, float scale) {
    __shared__ float red[128];
    __shared__ float sc[4096];
    __shared__ float sinkv;
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const int kh = h / (heads / nkv);
    const int r = threadIdx.x;
    const int w0 = window > 0 ? max(0, t - window + 1) : 0;
    const int n = t - w0 + 1;
    if (r == 0) sinkv = sink ? sink[(size_t)t * heads + h] : -INFINITY;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        const int s = w0 + i;
        red[r] = q[((size_t)t * heads + h) * dim + r] * k[((size_t)s * nkv + kh) * dim + r];
        __syncthreads();
        for (int o = 64; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        if (r == 0) sc[i] = red[0] * scale + bias[((size_t)h * T + t) * T + s];
        __syncthreads();
    }
    if (r == 0) {
        float mx = sink ? sinkv : -INFINITY;
        for (int i = 0; i < n; ++i) mx = fmaxf(mx, sc[i]);
        float sum = 0.f;
        for (int i = 0; i < n; ++i) {
            sc[i] = expf(sc[i] - mx);
            sum += sc[i];
        }
        if (sink) sum += expf(sinkv - mx);
        for (int i = 0; i < n; ++i) sc[i] /= sum;
    }
    __syncthreads();
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc += sc[i] * v[((size_t)(w0 + i) * nkv + kh) * dim + r];
    const float g = gate ? gate[((size_t)t * heads + h) * dim + r] : 1.f;
    out[((size_t)t * heads + h) * dim + r] = acc / (1.f + expf(-g));
}

// The attention residual (issue #62): a learned query attends over the running stack of block states plus the
// current partial state. `x` is [T, n, hidden] (the stack of `n` states at every position), `q` is [nq, hidden]
// (1 query per layer, 3 in the full layers), and `out` is [T, nq, hidden]. Every state row is normed with an
// unweighted RMSNorm first, and the softmax is over the n states.
extern "C" __global__ void __launch_bounds__(256) inkling2_attn_res(float* out, const float* q, const float* x, int T,
                                                                   int n, int hidden, int nq, float eps) {
    __shared__ float red[256];
    __shared__ float sc[16];
    __shared__ float fac[16];
    const int t = blockIdx.x / nq, qi = blockIdx.x % nq;
    const int r = threadIdx.x;
    for (int j = 0; j < n; ++j) {
        float s = 0.f;
        for (int i = r; i < hidden; i += 256) {
            const float v = x[((size_t)j * T + t) * hidden + i];
            s += v * v;
        }
        red[r] = s;
        __syncthreads();
        for (int o = 128; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        if (r == 0) fac[j] = rsqrtf(red[0] / (float)hidden + eps);
        __syncthreads();
    }
    for (int j = 0; j < n; ++j) {
        float s = 0.f;
        for (int i = r; i < hidden; i += 256) {
            s += q[qi * hidden + i] * x[((size_t)j * T + t) * hidden + i] * fac[j];
        }
        red[r] = s;
        __syncthreads();
        for (int o = 128; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        if (r == 0) sc[j] = red[0] * rsqrtf((float)hidden);
        __syncthreads();
    }
    if (r == 0) {
        float mx = sc[0];
        for (int j = 1; j < n; ++j) mx = fmaxf(mx, sc[j]);
        float sum = 0.f;
        for (int j = 0; j < n; ++j) {
            sc[j] = expf(sc[j] - mx);
            sum += sc[j];
        }
        for (int j = 0; j < n; ++j) sc[j] /= sum;
    }
    __syncthreads();
    for (int i = r; i < hidden; i += 256) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) acc += sc[j] * x[((size_t)j * T + t) * hidden + i] * fac[j];
        out[((size_t)t * nq + qi) * hidden + i] = acc;
    }
}

// MoE router: sigmoid scores, the bias added only for the selection, the weights renormalized over the k chosen
// scores and scaled by `route_scale`. No shared expert (issue #62). ids and w are [T, topk].
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_route(int* ids, float* w, const float* logits,
                                                                    const float* bias, int E, int topk,
                                                                    float route_scale) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int chosen[64];
    const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* l = logits + (size_t)t * E;
    for (int j = 0; j < topk; ++j) {
        float v = -INFINITY;
        int idx = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += 256) {
            bool taken = false;
            for (int c = 0; c < j; ++c) taken |= chosen[c] == e;
            const float x = 1.f / (1.f + expf(-l[e])) + bias[e];
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
        float s = 0.f, sc[64];
        for (int j = 0; j < topk; ++j) sc[j] = 1.f / (1.f + expf(-l[chosen[j]])), s += sc[j];
        for (int j = 0; j < topk; ++j) {
            ids[t * topk + j] = chosen[j];
            w[t * topk + j] = sc[j] / (s + 1e-20f) * route_scale;
        }
    }
}

// The expert activation: softcap(gate) * sigmoid(gate) * softcap(up), with softcap(x) = tanh(x / c) * c, then the
// randomized Hadamard of every 128-block like moe_act (the down weights are in the rotated basis).
//
// The checkpoint stores `w13` interleaved (`g0, u0, g1, u1, ...`): the reference deinterleaves it with
// `Interleave(dim=1)` before splitting the gate and the up. The MLS4 file keeps the checkpoint's row order, so the
// gate of element `j` is row `2j` and the up is row `2j+1`.
extern "C" __global__ void __launch_bounds__(128) inkling2_moe_act(float* out, const float* gu, const int8_t* signs,
                                                                  int I, float gate_cap, float up_cap) {
    __shared__ float s[128];
    const int nb = I / 128, p = blockIdx.x / nb, b = blockIdx.x % nb, i = threadIdx.x;
    const int pos = b * 128 + i;
    const float g = gu[(size_t)p * 2 * I + 2 * pos], u = gu[(size_t)p * 2 * I + 2 * pos + 1];
    const float gc = tanhf(g / gate_cap) * gate_cap;
    const float uc = tanhf(u / up_cap) * up_cap;
    s[i] = gc / (1.f + expf(-g)) * uc * (float)signs[i];
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

// The loop's cross-pass controller (issue #62): the message of the previous pass is mixed into this pass's input.
//   gate   = down . concat(rms(cur), rms(message))      [rank]
//   gain   = tanh(gain_w . gate)                        [hidden]
//   forget = tanh(forget_w . gate)                      [hidden]
//   out    = cur - forget * cur + gain * message
// One block per token; the weights are bf16 like the checkpoint's, the arithmetic is fp32.
extern "C" __global__ void __launch_bounds__(256) inkling2_control(float* out, const float* cur, const float* msg,
                                                                  const ik_bf16* down, const ik_bf16* gain_w,
                                                                  const ik_bf16* forget_w, int hidden, int rank,
                                                                  float eps) {
    __shared__ float red[256];
    __shared__ float gate[128];
    __shared__ float nc[4096];
    __shared__ float nm[4096];
    const int t = blockIdx.x, r = threadIdx.x;
    const float* cp = cur + (size_t)t * hidden;
    const float* mp = msg + (size_t)t * hidden;
    // rms of both inputs, then the concatenated normed pair.
    float sc = 0.f, sm = 0.f;
    for (int i = r; i < hidden; i += 256) sc += cp[i] * cp[i], sm += mp[i] * mp[i];
    red[r] = sc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (r < o) red[r] += red[r + o];
        __syncthreads();
    }
    const float fc = rsqrtf(red[0] / (float)hidden + eps);
    __syncthreads();
    red[r] = sm;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (r < o) red[r] += red[r + o];
        __syncthreads();
    }
    const float fm = rsqrtf(red[0] / (float)hidden + eps);
    for (int i = r; i < hidden; i += 256) nc[i] = cp[i] * fc, nm[i] = mp[i] * fm;
    __syncthreads();
    // gate = down . [nc | nm], one output per thread up to rank.
    for (int j = r; j < rank; j += 256) {
        const ik_bf16* w = down + (size_t)j * 2 * hidden;
        float acc = 0.f;
        for (int i = 0; i < hidden; ++i) acc += ik_bf2f(w[i]) * nc[i] + ik_bf2f(w[hidden + i]) * nm[i];
        gate[j] = acc;
    }
    __syncthreads();
    // out = cur - tanh(forget) * cur + tanh(gain) * message.
    for (int i = r; i < hidden; i += 256) {
        float g = 0.f, f = 0.f;
        for (int j = 0; j < rank; ++j) {
            g += ik_bf2f(gain_w[(size_t)i * rank + j]) * gate[j];
            f += ik_bf2f(forget_w[(size_t)i * rank + j]) * gate[j];
        }
        out[(size_t)t * hidden + i] = cp[i] * (1.f - tanhf(f)) + tanhf(g) * mp[i];
    }
}

// The final logit softcap: y = tanh(x / cap) * cap. One thread per element.
extern "C" __global__ void inkling2_softcap(float* y, const float* x, long long n, float cap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i] / cap) * cap;
}

// ---- the decode path: one token at a time, against the keys and values of the slots it has already seen ----

// The short convolution of one token, with the three-row history (`past`, [3, C], oldest first) shifted in place.
// Same arithmetic as inkling2_sconv at T=1; the shift is what makes one launch per token enough.
extern "C" __global__ void __launch_bounds__(256) inkling2_sconv_dec(float* y, const float* x, const ik_bf16* w,
                                                                   const float* dyn_w, float* past, int C, int heads,
                                                                   float div) {
    const int c = (int)blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const int h = c / (C / heads);
    // Tap 0 is the current token; taps 1..3 are the three rows of the history, newest first.
    const float v[4] = {x[c], past[(size_t)2 * C + c], past[(size_t)C + c], past[c]};
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        float wj = dyn_w ? ik_bf2f(w[(size_t)lag * C + c]) : ik_bf2f(w[(size_t)c * 4 + (3 - lag)]);
        if (dyn_w) wj += dyn_w[(size_t)h * 4 + lag];
        acc += wj * v[lag];
    }
    y[c] = (x[c] + acc) / div;
    past[c] = past[(size_t)C + c];
    past[(size_t)C + c] = past[(size_t)2 * C + c];
    past[(size_t)2 * C + c] = x[c];
}

// The relative-position bias of one query row at `pos`, over the `nk` keys starting at `k0`: bias is [heads, nk].
// The keys outside the layer's relative extent get 0, like the whole-prompt kernel.
extern "C" __global__ void __launch_bounds__(256) inkling2_rel_bias_dec(float* bias, const float* rel,
                                                                      const float* proj, int pos, int k0, int nk,
                                                                      int heads, int d_rel, int extent) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)heads * nk) return;
    const int h = (int)(idx / nk), s = (int)(idx % nk);
    const int d = pos - (k0 + s);
    float acc = 0.f;
    if (d >= 0 && d < extent) {
        const float* rp = rel + (size_t)h * d_rel;
        for (int i = 0; i < d_rel; ++i) acc += rp[i] * proj[(size_t)i * extent + d];
    }
    bias[idx] = acc;
}

// One query against the cached keys: k and v are [max_ctx, nkv, dim] with `nk` valid rows from `k0`, q, gate and
// out are [heads, dim], bias is [heads, nk] and sink is [heads]. The softmax runs online (max and running sum
// rescaled per key), so the cache length is not bounded by the shared memory the whole-prompt kernel needs.
extern "C" __global__ void __launch_bounds__(128) inkling2_attn_dec(float* out, const float* q, const float* k,
                                                                  const float* v, const float* bias, const float* sink,
                                                                  const float* gate, int k0, int nk, int heads,
                                                                  int nkv, int dim, float scale) {
    __shared__ float red[128];
    const int h = blockIdx.x, kh = h / (heads / nkv), r = threadIdx.x;
    const float qv = q[(size_t)h * dim + r];
    const float g = gate ? gate[(size_t)h * dim + r] : 1.f;
    // The sink is the score of a virtual key whose value is zero: it only enters the denominator.
    float m = sink ? sink[h] : -INFINITY, l = sink ? 1.f : 0.f, acc = 0.f;
    for (int i = 0; i < nk; ++i) {
        const size_t s = (size_t)(k0 + i) * nkv + kh;
        red[r] = qv * k[s * dim + r];
        __syncthreads();
        for (int o = 64; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        const float sc = red[0] * scale + bias[(size_t)h * nk + i];
        const float mn = fmaxf(m, sc), k1 = __expf(m - mn), e = __expf(sc - mn);
        l = l * k1 + e;
        acc = acc * k1 + e * v[s * dim + r];
        m = mn;
        __syncthreads();
    }
    out[(size_t)h * dim + r] = acc / l / (1.f + expf(-g));
}

// MoE combine: y[t][i] += w[t] * x[t][i]. The route weight of one expert for every token, applied as the expert's
// output is added into the layer's accumulator.
extern "C" __global__ void inkling2_moe_acc(float* y, const float* x, const float* w, long long n, int h) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += w[i / h] * x[i];
}
