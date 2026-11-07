// Pre- and post-processing - identical to trainer/colab/scripts/ml_tools.py and trainer/convert/decode.py.
// Plain C++ without ESP dependencies: also tested on the PC (firmware/tools/test_pc).
#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>

namespace pd {

constexpr int GRAY = 114;

// ------------------------------------------------------------------ letterbox
struct Letterbox {
    float s;      // scale original -> letterbox
    int nw, nh;   // size of the image inside the letterbox
    int dx, dy;   // offset (border left/top)
};

// like ml_tools.letterbox_params (w, h = width/height of the original)
inline Letterbox letterbox_params(int w, int h, int imgsz)
{
    Letterbox l;
    float sw = (float)imgsz / w, sh = (float)imgsz / h;
    l.s = sw < sh ? sw : sh;
    l.nw = (int)lrintf(w * l.s);
    l.nh = (int)lrintf(h * l.s);
    l.dx = (imgsz - l.nw) / 2;
    l.dy = (imgsz - l.nh) / 2;
    return l;
}

// bilinear like cv2.resize(INTER_LINEAR): source sx = (x+0.5)*fx - 0.5, clamped at the border.
struct Axis {
    int i0, i1;   // source indices
    float w1;     // weight for i1 (w0 = 1 - w1)
};

inline void axis_resize(Axis *a, int n_dst, int n_src)
{
    float f = (float)n_src / n_dst;
    for (int i = 0; i < n_dst; i++) {
        float s = (i + 0.5f) * f - 0.5f;
        if (s < 0) s = 0;
        int i0 = (int)s;
        float w = s - i0;
        if (i0 >= n_src - 1) { i0 = n_src - 1; w = 0; }
        a[i].i0 = i0;
        a[i].i1 = i0 + 1 < n_src ? i0 + 1 : i0;
        a[i].w1 = w;
    }
}

// rgb: W x H x 3 (uint8, RGB). Destination: imgsz x imgsz x 3. 'map' converts the uint8 value to the destination
// type (e.g. int8 via a table for the model or uint8 unchanged for tests).
// ax/ay: work memory for at least imgsz entries.
// Integer (weights in 1/256) and row by row: every needed source row is scaled horizontally once
// (reads the image in order -> fast from PSRAM), then two rows are blended.
constexpr int LB_MAX = 640;   // largest letterbox width
template <typename T, typename F>
void letterbox(const uint8_t *rgb, int W, int H, int imgsz, T *dst, F map, Axis *ax, Axis *ay)
{
    static uint16_t row_a[LB_MAX * 3], row_b[LB_MAX * 3];
    static uint16_t wx1[LB_MAX];
    Letterbox l = letterbox_params(W, H, imgsz);
    if (l.nw > LB_MAX) l.nw = LB_MAX;
    T gray = map((uint8_t)GRAY);
    int n = imgsz * imgsz * 3;
    for (int i = 0; i < n; i++) dst[i] = gray;
    axis_resize(ax, l.nw, W);
    axis_resize(ay, l.nh, H);
    for (int x = 0; x < l.nw; x++) wx1[x] = (uint16_t)(ax[x].w1 * 256.0f + 0.5f);

    // horizontal: source row q -> buf (values * 256)
    auto horizontal = [&](int q, uint16_t *buf) {
        const uint8_t *r = rgb + (size_t)q * W * 3;
        for (int x = 0; x < l.nw; x++) {
            const uint8_t *p0 = r + ax[x].i0 * 3, *p1 = r + ax[x].i1 * 3;
            uint16_t w1 = wx1[x], w0 = (uint16_t)(256 - w1);
            buf[x * 3] = (uint16_t)(p0[0] * w0 + p1[0] * w1);
            buf[x * 3 + 1] = (uint16_t)(p0[1] * w0 + p1[1] * w1);
            buf[x * 3 + 2] = (uint16_t)(p0[2] * w0 + p1[2] * w1);
        }
    };
    int in_a = -1, in_b = -1;   // which source row is in row_a / row_b
    for (int y = 0; y < l.nh; y++) {
        int q0 = ay[y].i0, q1 = ay[y].i1;
        // reuse the buffers when the row is already there
        if (in_a != q0) {
            if (in_b == q0) { for (int i = 0; i < l.nw * 3; i++) row_a[i] = row_b[i]; }
            else horizontal(q0, row_a);
            in_a = q0;
        }
        if (in_b != q1) {
            if (q1 == q0) { for (int i = 0; i < l.nw * 3; i++) row_b[i] = row_a[i]; }
            else horizontal(q1, row_b);
            in_b = q1;
        }
        uint32_t wy1 = (uint32_t)(ay[y].w1 * 256.0f + 0.5f), wy0 = 256 - wy1;
        T *out = dst + ((size_t)(y + l.dy) * imgsz + l.dx) * 3;
        for (int i = 0; i < l.nw * 3; i++) {
            uint32_t v = (row_a[i] * wy0 + row_b[i] * wy1 + 32768u) >> 16;
            out[i] = map((uint8_t)(v > 255 ? 255 : v));
        }
    }
}

// ------------------------------------------------------------------ crop around a die
// Like ml_tools.crop: square around the box center, side = max(w,h)*(1+2*margin), bilinear,
// outside gray 114. Box in original pixels; the buffer image may be smaller
// (k = buffer pixels per original pixel, e.g. 0.5 at half resolution).
// Integer: source coordinates in 1/65536 pixel, weights in 1/256.
//
// Usable row by row (for the block mode of the JPEG decoder, which delivers the image in strips):
//   CropPlan p = crop_plan(...);  for each destination row v: provide source rows crop_source_row(p, v) and +1
//   and call crop_row(p, v, row0, row1, W, dst, map)
//   (row pointer nullptr = row is outside the image -> gray).
struct CropPlan {
    int g;            // destination size g x g
    int32_t d;        // step per destination pixel (1/65536 buffer pixel)
    int32_t sx0, sy0; // source position of the first destination pixel
};

inline CropPlan crop_plan(float k, float x0, float y0, float x1, float y1, int g, float margin)
{
    float bw = x1 - x0, bh = y1 - y0;
    float side = (bw > bh ? bw : bh) * (1.0f + 2.0f * margin);
    float cx = (x0 + x1) * 0.5f, cy = (y0 + y1) * 0.5f;
    float s = side / g;
    // buffer pixel for u: sx(u) = ((cx + (u + 0.5 - g/2) * s - 0.5) + 0.5) * k - 0.5  = a + u * d
    float ax0 = ((cx + (0.5f - g * 0.5f) * s - 0.5f) + 0.5f) * k - 0.5f;
    float ay0 = ((cy + (0.5f - g * 0.5f) * s - 0.5f) + 0.5f) * k - 0.5f;
    CropPlan p;
    p.g = g;
    p.d = (int32_t)lrintf(s * k * 65536.0f);
    p.sx0 = (int32_t)lrintf(ax0 * 65536.0f);
    p.sy0 = (int32_t)lrintf(ay0 * 65536.0f);
    return p;
}

// upper of the two source rows for destination row v (may be < 0 or >= H)
inline int crop_source_row(const CropPlan &p, int v) { return (p.sy0 + v * p.d) >> 16; }

template <typename T, typename F>
void crop_row(const CropPlan &p, int v, const uint8_t *r0, const uint8_t *r1, int W, T *dst, F map)
{
    int32_t sy = p.sy0 + v * p.d;
    uint32_t wy1 = ((uint32_t)sy >> 8) & 255, wy0 = 256 - wy1;
    T *out = dst + (size_t)v * p.g * 3;
    int32_t sx = p.sx0;
    for (int u = 0; u < p.g; u++, sx += p.d, out += 3) {
        int ix = sx >> 16;                       // arithmetic -> rounds down also when negative
        uint32_t wx1 = ((uint32_t)sx >> 8) & 255, wx0 = 256 - wx1;
        if (r0 && r1 && ix >= 0 && ix + 1 < W) {  // fast path: all 4 neighbors inside the image
            const uint8_t *a = r0 + ix * 3, *b = r1 + ix * 3;
            for (int c = 0; c < 3; c++) {
                uint32_t top = a[c] * wx0 + a[c + 3] * wx1, bot = b[c] * wx0 + b[c + 3] * wx1;
                out[c] = map((uint8_t)((top * wy0 + bot * wy1 + 32768u) >> 16));
            }
        } else {                                  // border: neighbors outside = gray 114
            for (int c = 0; c < 3; c++) {
                uint32_t px[4];
                for (int j = 0; j < 4; j++) {
                    int xx = ix + (j & 1);
                    const uint8_t *r = (j >> 1) ? r1 : r0;
                    px[j] = (!r || xx < 0 || xx >= W) ? (uint32_t)GRAY : r[xx * 3 + c];
                }
                uint32_t top = px[0] * wx0 + px[1] * wx1, bot = px[2] * wx0 + px[3] * wx1;
                out[c] = map((uint8_t)((top * wy0 + bot * wy1 + 32768u) >> 16));
            }
        }
    }
}

// whole image in memory (W x H x 3)
template <typename T, typename F>
void crop(const uint8_t *rgb, int W, int H, float k, float x0, float y0, float x1, float y1,
          int g, float margin, T *dst, F map)
{
    CropPlan p = crop_plan(k, x0, y0, x1, y1, g, margin);
    const size_t rs = (size_t)W * 3;
    for (int v = 0; v < g; v++) {
        int iy = crop_source_row(p, v);
        const uint8_t *r0 = (iy >= 0 && iy < H) ? rgb + (size_t)iy * rs : nullptr;
        const uint8_t *r1 = (iy + 1 >= 0 && iy + 1 < H) ? rgb + (size_t)(iy + 1) * rs : nullptr;
        crop_row(p, v, r0, r1, W, dst, map);
    }
}

// Crops from an image that arrives in strips (JPEG block mode): call strip() for every strip;
// only the current and the previous strip are needed. strip() returns true = all crops complete ->
// the rest of the image need not be decoded. Result identical to crop() on the whole image.
template <typename T, int MAXN>
struct CropStream {
    CropPlan plan[MAXN];
    int v[MAXN];
    T *dst[MAXN];
    int n = 0, W = 0, H = 0;

    void start(int count, int width, int height) { n = count < MAXN ? count : MAXN; W = width; H = height; }
    void set(int i, const CropPlan &p, T *d) { plan[i] = p; v[i] = 0; dst[i] = d; }

    // cur: rows y0 .. y0+sh-1, prev: rows y0-sh .. y0-1 (may be nullptr for the first strip)
    template <typename F>
    bool strip(const uint8_t *cur, const uint8_t *prev, int y0, int sh, bool last, F map)
    {
        const size_t rs = (size_t)W * 3;
        const int end = y0 + sh;
        auto row = [&](int r) -> const uint8_t * {
            if (r < 0 || r >= H) return nullptr;
            if (r >= y0) return cur + (size_t)(r - y0) * rs;
            return prev ? prev + (size_t)(r - y0 + sh) * rs : nullptr;
        };
        bool all = true;
        for (int i = 0; i < n; i++) {
            while (v[i] < plan[i].g) {
                int iy = crop_source_row(plan[i], v[i]);
                if (iy + 1 >= end && !last) break;   // the lower source row comes with the next strip
                crop_row(plan[i], v[i], row(iy), row(iy + 1), W, dst[i], map);
                v[i]++;
            }
            if (v[i] < plan[i].g) all = false;
        }
        return all;
    }
};

// table uint8 (0..255) -> int8 for a model input with exponent e: q = round(v/255 / 2^e)
inline void input_table(int8_t tab[256], int exponent)
{
    float f = ldexpf(1.0f, -exponent) / 255.0f;
    for (int v = 0; v < 256; v++) {
        int q = (int)lrintf(v * f);
        tab[v] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
}

// ------------------------------------------------------------------ decode the detector outputs
struct Box {
    float x0, y0, x1, y1;   // original pixels
    float score;
};

struct Level {               // one output pair (box, score) in NHWC int8 format
    const int8_t *box;
    int box_exp;
    const int8_t *score;
    int score_exp;
    int h, w, channels;      // channels = 16 (padded)
    int stride;
};

inline float iou(const Box &a, const Box &b)
{
    float ix = fminf(a.x1, b.x1) - fmaxf(a.x0, b.x0);
    float iy = fminf(a.y1, b.y1) - fmaxf(a.y0, b.y0);
    if (ix <= 0 || iy <= 0) return 0;
    float inter = ix * iy;
    float u = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return u > 0 ? inter / u : 0;
}

// cand: work memory (max_cand), out: result (max_out). Returns: number of boxes after NMS.
// *n_cand receives the number of candidates before NMS (for the monitoring).
inline int decode(const Level *lv, int n_levels, int imgsz, int orig_w, int orig_h, float threshold,
                  float nms_iou, Box *cand, int max_cand, Box *out, int max_out, int *n_cand)
{
    Letterbox l = letterbox_params(orig_w, orig_h, imgsz);
    float logit = logf(threshold / (1.0f - threshold));
    int n = 0;
    for (int e = 0; e < n_levels; e++) {
        const Level &L = lv[e];
        float fs = ldexpf(1.0f, L.score_exp), fb = ldexpf(1.0f, L.box_exp);
        for (int y = 0; y < L.h; y++) {
            for (int x = 0; x < L.w; x++) {
                size_t i = ((size_t)y * L.w + x) * L.channels;
                float sc = L.score[i] * fs;
                if (sc <= logit) continue;
                float lft = L.box[i] * fb, top = L.box[i + 1] * fb, rgt = L.box[i + 2] * fb, bot = L.box[i + 3] * fb;
                float cx = (x + 0.5f) * L.stride, cy = (y + 0.5f) * L.stride;
                Box r;
                r.x0 = (cx - lft * L.stride - l.dx) / l.s;
                r.y0 = (cy - top * L.stride - l.dy) / l.s;
                r.x1 = (cx + rgt * L.stride - l.dx) / l.s;
                r.y1 = (cy + bot * L.stride - l.dy) / l.s;
                r.score = 1.0f / (1.0f + expf(-sc));
                if (n < max_cand) {
                    cand[n++] = r;
                } else {   // full: replace the weakest
                    int m = 0;
                    for (int j = 1; j < n; j++) if (cand[j].score < cand[m].score) m = j;
                    if (cand[m].score < r.score) cand[m] = r;
                }
            }
        }
    }
    if (n_cand) *n_cand = n;
    // sort by score descending (insertion sort, n is small)
    for (int i = 1; i < n; i++) {
        Box t = cand[i];
        int j = i - 1;
        while (j >= 0 && cand[j].score < t.score) { cand[j + 1] = cand[j]; j--; }
        cand[j + 1] = t;
    }
    int m = 0;
    for (int i = 0; i < n && m < max_out; i++) {
        bool ok = true;
        for (int j = 0; j < m; j++) if (iou(cand[i], out[j]) >= nms_iou) { ok = false; break; }
        if (ok) out[m++] = cand[i];
    }
    return m;
}

// softmax over the first n values of an int8 output. Returns: best class, *p = probability.
inline int choose_class(const int8_t *o, int exponent, int n, float *p)
{
    float f = ldexpf(1.0f, exponent);
    int best = 0;
    for (int i = 1; i < n; i++) if (o[i] > o[best]) best = i;
    float mx = o[best] * f, sum = 0;
    for (int i = 0; i < n; i++) sum += expf(o[i] * f - mx);
    if (p) *p = 1.0f / sum;
    return best;
}

// ------------------------------------------------------------------ poker hand (the web page has the same names)
enum Hand {
    H_NONE = 0,   // not 5 clear dice
    H_NOTHING, H_PAIR, H_TWO_PAIRS, H_THREE, H_SMALL_STRAIGHT, H_LARGE_STRAIGHT,
    H_FULL_HOUSE, H_FOUR, H_FIVE,
};

// English name of a hand ("" = no hand)
inline const char *hand_name(int h)
{
    static const char *const names[10] = {"", "Nothing", "One Pair", "Two Pairs", "Three of a Kind", "Small Straight",
                                          "Large Straight", "Full House", "Four of a Kind", "Five of a Kind"};
    return h >= 0 && h < 10 ? names[h] : "";
}

// ranks: rank per die (0 = lowest, -1 = unclear), n_ranks = number of different ranks (6).
inline int score_hand(const int *ranks, int n, int n_ranks)
{
    if (n != 5) return H_NONE;
    int count[16] = {0};
    for (int i = 0; i < n; i++) {
        if (ranks[i] < 0 || ranks[i] >= 16) return H_NONE;
        count[ranks[i]]++;
    }
    int a = 0, b = 0;   // largest and second largest count
    for (int r = 0; r < 16; r++) {
        if (count[r] > a) { b = a; a = count[r]; }
        else if (count[r] > b) b = count[r];
    }
    if (a == 5) return H_FIVE;
    if (a == 4) return H_FOUR;
    if (a == 3 && b == 2) return H_FULL_HOUSE;
    if (a == 3) return H_THREE;
    if (a == 2 && b == 2) return H_TWO_PAIRS;
    if (a == 2) return H_PAIR;
    int lo = 99, hi = -1;
    for (int r = 0; r < 16; r++) if (count[r]) { if (r < lo) lo = r; if (r > hi) hi = r; }
    if (hi - lo == 4) return hi == n_ranks - 1 ? H_LARGE_STRAIGHT : H_SMALL_STRAIGHT;
    return H_NOTHING;
}

}  // namespace pd
