#include "game_ggml/decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace game_ggml {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf =  std::numeric_limits<float>::infinity();
}

std::vector<std::uint8_t> decode_soft_boundaries(
    const float * probs, std::size_t len,
    const std::uint8_t * barriers, const std::uint8_t * mask,
    float threshold, int radius)
{
    // Matches modules/decoding.py::decode_soft_boundaries + find_local_extremum:
    //   - Mask invalid frames to +inf so they never count as maxima.
    //   - Force barrier frames to +inf so they always win the window.
    //   - A frame is a boundary iff it's the max in [-r, +r] AND >= threshold.
    //   - Barrier frames bypass the threshold (arg-max makes them maxima
    //     regardless because +inf compares as max).
    std::vector<float> v(len);
    for (std::size_t i = 0; i < len; ++i) {
        if (mask && !mask[i])       v[i] = kPosInf;
        else if (barriers && barriers[i]) v[i] = kPosInf;
        else                         v[i] = probs[i];
    }

    std::vector<std::uint8_t> out(len, 0);
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(len); ++i) {
        // Argmax within the ±radius window (edges clamped); take leftmost
        // maximum to mirror numpy/PyTorch argmax on ties.
        const std::ptrdiff_t lo = std::max<std::ptrdiff_t>(0, i - radius);
        const std::ptrdiff_t hi = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(len) - 1, i + radius);
        float best = v[lo];
        std::ptrdiff_t arg = lo;
        for (std::ptrdiff_t k = lo + 1; k <= hi; ++k) {
            if (v[k] > best) { best = v[k]; arg = k; }
        }
        if (arg != i) continue;
        const bool is_barrier = (barriers && barriers[i]);
        const bool meets_thr  = (probs[i] >= threshold);
        if (is_barrier || meets_thr) out[i] = 1;
        if (mask && !mask[i]) out[i] = 0;
    }
    return out;
}

GaussianBlurredResult decode_gaussian_blurred_probs(
    const float * probs, std::size_t n, std::size_t bins,
    float min_val, float max_val, float deviation, float threshold)
{
    // Matches modules/decoding.py::decode_gaussian_blurred_probs.
    GaussianBlurredResult r;
    r.values.assign(n, 0.0f);
    r.presence.assign(n, 0);

    const int   N = static_cast<int>(bins);
    const float step = (max_val - min_val) / (N - 1);
    const int   width = static_cast<int>(std::ceil(deviation / step));

    std::vector<float> centers(N);
    for (int k = 0; k < N; ++k) centers[k] = min_val + step * k;

    for (std::size_t i = 0; i < n; ++i) {
        const float * row = probs + i * bins;
        int   argmax = 0;
        float vmax   = row[0];
        for (int k = 1; k < N; ++k) {
            if (row[k] > vmax) { vmax = row[k]; argmax = k; }
        }
        const int lo = std::max(0, argmax - width);
        const int hi = std::min(N, argmax + width + 1);
        float wsum = 0.0f;
        float psum = 0.0f;
        for (int k = lo; k < hi; ++k) {
            wsum += row[k];
            psum += row[k] * centers[k];
        }
        r.values[i]   = psum / (wsum + 1e-8f);
        r.presence[i] = (vmax >= threshold) ? 1 : 0;
    }
    return r;
}

std::vector<std::int32_t> boundaries_to_regions(
    const std::uint8_t * boundaries, const std::uint8_t * mask,
    std::size_t len)
{
    // regions = cumsum(boundaries.long()) + 1, masked to 0 on padding.
    std::vector<std::int32_t> r(len, 0);
    std::int32_t running = 1;
    for (std::size_t i = 0; i < len; ++i) {
        running += (boundaries[i] ? 1 : 0);
        const bool valid = (!mask) || (mask[i] != 0);
        r[i] = valid ? running : 0;
    }
    return r;
}

std::vector<std::uint8_t> format_boundaries(
    const float * durations, std::size_t n_words,
    std::size_t n_frames, float timestep)
{
    // Matches modules/functional.py::format_boundaries:
    //   cumsum = durations.cumsum() / timestep
    //   boundary_indices = round(cumsum).long()[:-1]
    //   boundaries[t] = (t == any boundary_index)
    std::vector<std::uint8_t> b(n_frames, 0);
    float acc = 0.0f;
    for (std::size_t i = 0; i + 1 < n_words; ++i) {   // last element dropped
        acc += durations[i];
        const int idx = static_cast<int>(std::lround(acc / timestep));
        if (idx >= 0 && static_cast<std::size_t>(idx) < n_frames) {
            b[idx] = 1;
        }
    }
    return b;
}

}  // namespace game_ggml
