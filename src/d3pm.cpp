#include "d3pm.h"

#include "rng.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace game_ggml::internal {

float d3pm_time_schedule(float t) {
    // p = (1 + cos(t*pi)) / 2  — matches modules/d3pm.py::d3pm_time_schedule.
    return 0.5f * (1.0f + std::cos(t * static_cast<float>(3.14159265358979323846)));
}

void remove_boundaries(
    const std::uint8_t * boundaries, std::size_t n,
    float p,
    IRandomSource & rng,
    std::uint8_t * out)
{
    // PyTorch: rnd = torch.rand_like(boundaries, dtype=float32);
    //         remain = (rnd <= 1 - p) & boundaries
    std::vector<float> r(n);
    rng.uniform(r.data(), n);
    const float keep = 1.0f - p;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (boundaries[i] && (r[i] <= keep)) ? 1 : 0;
    }
}

void remove_mutable_boundaries(
    const std::uint8_t * boundaries,
    const std::uint8_t * immutable,
    std::size_t n,
    float p,
    IRandomSource & rng,
    std::uint8_t * out)
{
    // PyTorch logic: only random-drop mutable boundaries with rescaled p so
    // expected remaining count matches uniform-p over all boundaries.
    //   mutable = boundaries & ~immutable
    //   N = #boundaries        M = #mutable
    //   P = min(1, N*p / M)   per-sample drop probability for mutable ones
    //   remain = (mutable_remain_with_P) | immutable
    std::size_t N = 0, M = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool b = boundaries[i] != 0;
        const bool im = immutable[i] != 0;
        if (b)        ++N;
        if (b && !im) ++M;
    }
    float P = 1.0f;
    if (M > 0) P = std::min(1.0f, static_cast<float>(N) * p / static_cast<float>(M));

    std::vector<float> r(n);
    rng.uniform(r.data(), n);
    const float keep = 1.0f - P;
    for (std::size_t i = 0; i < n; ++i) {
        const bool b  = boundaries[i] != 0;
        const bool im = immutable[i] != 0;
        const bool mut_b = b && !im;
        std::uint8_t keep_mut = (mut_b && r[i] <= keep) ? 1 : 0;
        out[i] = (im && b) ? 1 : keep_mut;
    }
}

}  // namespace game_ggml::internal
