// PC test: crops from strips (JPEG block mode of the firmware) == crop from the whole image, bit-exact.
// Called by compare.py; by hand (Linux/WSL/MSYS2, in firmware/tools/test_pc):  g++ -O2 -std=c++17 -I ../../main strips.cpp -o strips && ./strips
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "preprocess.hpp"

int main()
{
    srand(7);
    long diff = 0, cases = 0, early = 0;
    for (int t = 0; t < 800; t++) {
        int W = (rand() % 3 == 0) ? 640 : (rand() % 2 ? 1024 : 1600), H = W * 3 / 4, sh = (rand() % 2) ? 8 : 16;
        std::vector<uint8_t> img((size_t)W * H * 3);
        for (auto &v : img) v = rand() & 255;
        int n = 1 + rand() % 6, g = (rand() % 2) ? 64 : 96;
        pd::CropStream<uint8_t, 10> st;
        st.start(n, W, H);
        std::vector<std::vector<uint8_t>> a(n, std::vector<uint8_t>(g * g * 3)), b(n, std::vector<uint8_t>(g * g * 3));
        float r[10][4];
        for (int i = 0; i < n; i++) {
            float bw = 20 + rand() % 160, bh = 20 + rand() % 160;
            float x0 = (rand() % (W + 200)) - 100.3f, y0 = (rand() % (H + 200)) - 100.7f;
            r[i][0] = x0; r[i][1] = y0; r[i][2] = x0 + bw; r[i][3] = y0 + bh;
            st.set(i, pd::crop_plan(1.0f, x0, y0, x0 + bw, y0 + bh, g, 0.25f), b[i].data());
        }
        auto id = [](uint8_t v) { return v; };
        // strips one after another, with own buffers (like in the decoder: only current + previous)
        std::vector<uint8_t> s0((size_t)W * sh * 3), s1((size_t)W * sh * 3);
        int count = H / sh;
        for (int s = 0; s < count; s++) {
            std::vector<uint8_t> &cur = (s & 1) ? s1 : s0, &prev = (s & 1) ? s0 : s1;
            memcpy(cur.data(), img.data() + (size_t)s * sh * W * 3, (size_t)W * sh * 3);
            if (st.strip(cur.data(), s ? prev.data() : nullptr, s * sh, sh, s == count - 1, id)) { early += s < count - 1; break; }
        }
        for (int i = 0; i < n; i++) {
            pd::crop(img.data(), W, H, 1.0f, r[i][0], r[i][1], r[i][2], r[i][3], g, 0.25f, a[i].data(), id);
            for (size_t k = 0; k < a[i].size(); k++) diff += a[i][k] != b[i][k];
            cases++;
        }
    }
    printf("%ld crops, %ld differing values, %ld images finished early\n", cases, diff, early);
    return diff ? 1 : 0;
}
