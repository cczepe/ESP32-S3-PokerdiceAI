// Turns single frames into a calm result ("n of m frames equal").
// Plain C++ - testable on the PC.
#pragma once
#include <string.h>
#include "preprocess.hpp"

namespace pd {

constexpr int STAB_MAX_M = 8;
constexpr int STAB_MAX_N = 10;

struct Signature {
    int n;                      // number of dice
    int k[STAB_MAX_N];          // classes, sorted by rank (high -> low, unclear last)
    bool equals(const Signature &o) const
    {
        if (n != o.n) return false;
        for (int i = 0; i < n; i++) if (k[i] != o.k[i]) return false;
        return true;
    }
};

// sort classes by rank descending (rank -1 = unclear to the end)
inline void sort_by_rank(int *k, int n, const int *rank)
{
    for (int i = 1; i < n; i++) {
        int t = k[i], j = i - 1;
        while (j >= 0 && rank[k[j]] < rank[t]) { k[j + 1] = k[j]; j--; }
        k[j + 1] = t;
    }
}

enum StabStatus { S_SEARCHING = 0, S_ROLLING, S_STABLE };

struct Stabilizer {
    Signature history[STAB_MAX_M];
    int pos = 0, filled = 0;
    Signature reported{};       // last reported result
    bool has_reported = false;
    int restless = 99;          // frames since the last report that looked different

    // classes: n classes (any order), p: probabilities.
    // Returns the status; *is_new = true when a new stable result has just appeared.
    StabStatus step(const int *classes, const float *p, int n, const int *rank, int expected,
                    int class_unsure, int stable_n, int stable_m, float p_min, bool *is_new, Signature *current)
    {
        *is_new = false;
        if (stable_m > STAB_MAX_M) stable_m = STAB_MAX_M;
        if (stable_m < 1) stable_m = 1;
        if (stable_n > stable_m) stable_n = stable_m;
        Signature s;
        s.n = n > STAB_MAX_N ? STAB_MAX_N : n;
        bool clear = true;
        for (int i = 0; i < s.n; i++) {
            s.k[i] = classes[i];
            if (classes[i] == class_unsure || rank[classes[i]] < 0 || p[i] < p_min) clear = false;
        }
        sort_by_rank(s.k, s.n, rank);
        history[pos] = s;
        pos = (pos + 1) % STAB_MAX_M;
        if (filled < STAB_MAX_M) filled++;
        if (current) *current = s;

        int equal = 0, count = filled < stable_m ? filled : stable_m;
        for (int i = 0; i < count; i++) {
            int j = (pos - 1 - i + STAB_MAX_M) % STAB_MAX_M;
            if (history[j].equals(s)) equal++;
        }
        bool as_reported = has_reported && s.equals(reported);
        if (!as_reported && restless < 99) restless++;

        if (s.n == 0) return S_SEARCHING;
        if (equal >= stable_n && s.n == expected && clear) {
            // new result, or the same after an interruption (rolled again)
            if (!as_reported || restless >= 2) {
                reported = s;
                has_reported = true;
                restless = 0;
                *is_new = true;
            }
            return S_STABLE;
        }
        if (s.n < expected && equal >= stable_n) return S_SEARCHING;   // calm, but dice are missing
        return S_ROLLING;
    }

    void reset() { pos = filled = 0; has_reported = false; restless = 99; }
};

}  // namespace pd
