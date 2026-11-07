// PC test: the pre- and post-processing of the firmware (main/preprocess.hpp), called by compare.py.
//   compare letterbox <in.rgb> <W> <H> <imgsz> <out.rgb>
//   compare crop      <in.rgb> <W> <H> <k> <x0> <y0> <x1> <y1> <g> <margin> <out.rgb>
//   compare decode    <imgsz> <W> <H> <threshold> <nms> { <h> <w> <ch> <stride> <box_exp> <score_exp> <box.bin> <score.bin> } x3
//   compare hand      <n_ranks> <r1> <r2> <r3> <r4> <r5>       -> "<code>|<name>"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "preprocess.hpp"

static std::vector<uint8_t> read_file(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(2);
    fclose(f);
    return v;
}

static void write_file(const char *p, const void *d, size_t n)
{
    FILE *f = fopen(p, "wb");
    fwrite(d, 1, n, f);
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "see the head of compare.cpp\n"); return 1; }
    auto same = [](uint8_t v) { return v; };
    if (!strcmp(argv[1], "letterbox") && argc == 7) {
        auto in = read_file(argv[2]);
        int W = atoi(argv[3]), H = atoi(argv[4]), g = atoi(argv[5]);
        std::vector<uint8_t> out(g * g * 3);
        std::vector<pd::Axis> ax(g), ay(g);
        pd::letterbox(in.data(), W, H, g, out.data(), same, ax.data(), ay.data());
        write_file(argv[6], out.data(), out.size());
        return 0;
    }
    if (!strcmp(argv[1], "crop") && argc == 13) {
        auto in = read_file(argv[2]);
        int W = atoi(argv[3]), H = atoi(argv[4]), g = atoi(argv[10]);
        std::vector<uint8_t> out(g * g * 3);
        pd::crop(in.data(), W, H, atof(argv[5]), atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), g,
                 atof(argv[11]), out.data(), same);
        write_file(argv[12], out.data(), out.size());
        return 0;
    }
    if (!strcmp(argv[1], "decode") && argc == 7 + 3 * 8) {
        int imgsz = atoi(argv[2]), W = atoi(argv[3]), H = atoi(argv[4]);
        float threshold = atof(argv[5]), nms = atof(argv[6]);
        pd::Level lv[3];
        std::vector<uint8_t> data[6];
        for (int e = 0; e < 3; e++) {
            char **a = argv + 7 + e * 8;
            lv[e].h = atoi(a[0]); lv[e].w = atoi(a[1]); lv[e].channels = atoi(a[2]); lv[e].stride = atoi(a[3]);
            lv[e].box_exp = atoi(a[4]); lv[e].score_exp = atoi(a[5]);
            data[2 * e] = read_file(a[6]);
            data[2 * e + 1] = read_file(a[7]);
            lv[e].box = (const int8_t *)data[2 * e].data();
            lv[e].score = (const int8_t *)data[2 * e + 1].data();
        }
        static pd::Box cand[256], out[64];
        int nc = 0;
        int n = pd::decode(lv, 3, imgsz, W, H, threshold, nms, cand, 256, out, 64, &nc);
        printf("%d %d\n", n, nc);
        for (int i = 0; i < n; i++) printf("%.3f %.3f %.3f %.3f %.5f\n", out[i].x0, out[i].y0, out[i].x1, out[i].y1, out[i].score);
        return 0;
    }
    if (!strcmp(argv[1], "hand") && argc == 8) {
        int r[5];
        for (int i = 0; i < 5; i++) r[i] = atoi(argv[3 + i]);
        int h = pd::score_hand(r, 5, atoi(argv[2]));
        printf("%d|%s\n", h, pd::hand_name(h));
        return 0;
    }
    fprintf(stderr, "wrong arguments\n");
    return 1;
}
