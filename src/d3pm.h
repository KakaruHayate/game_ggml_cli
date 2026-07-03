#pragma once

// D3PM-specific helpers (time schedule, randomised boundary dropping).
// Internal use.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace game_ggml::internal { class IRandomSource; }

namespace game_ggml::internal {

// Cosine schedule: returns (1 + cos(t*pi)) / 2 in [0, 1].
float d3pm_time_schedule(float t);

// Remove boundaries uniformly at random with probability `p`.  Fills
// `out` (length == n); semantics match modules/d3pm.py::remove_boundaries.
void remove_boundaries(
    const std::uint8_t * boundaries, std::size_t n,
    float p,
    IRandomSource & rng,
    std::uint8_t * out);

// Remove only *mutable* (not immutable) boundaries, with adjusted p so
// the expected count of remaining boundaries equals the uniform-p case.
// Matches modules/d3pm.py::remove_mutable_boundaries.
void remove_mutable_boundaries(
    const std::uint8_t * boundaries,
    const std::uint8_t * immutable,
    std::size_t n,
    float p,
    IRandomSource & rng,
    std::uint8_t * out);

}  // namespace game_ggml::internal
