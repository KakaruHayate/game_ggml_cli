#pragma once

// Decoding helpers for boundary + pitch.  These operate on plain CPU
// buffers (outputs of the ggml graphs); they do not touch the graph
// themselves.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace game_ggml {

// Decode gaussian-softened boundary probabilities into hard indicators.
//
//   * `probs`       length T, sigmoid-activated boundary probabilities.
//   * `barriers`    optional length T; any true entries are forced to
//                   boundaries (before radius selection).
//   * `mask`        optional length T valid mask; positions with mask=false
//                   are excluded.
//   * `threshold`   probability floor for local maxima.
//   * `radius`      a frame is kept iff it's the maximum within [-r, +r].
//
// Returns length-T uint8 (0 / 1) — matches torch.bool semantics.
std::vector<std::uint8_t> decode_soft_boundaries(
    const float * probs,
    std::size_t   len,
    const std::uint8_t * barriers,
    const std::uint8_t * mask,
    float threshold,
    int   radius);

// Decode gaussian-blurred probabilities into (pitch, presence) pairs.
//
//   * `probs`       row-major [N, bins] sigmoid-activated.
//   * `n`, `bins`   outer / inner lengths.
//   * `min_val`/`max_val`   bin edges in value units (e.g. MIDI semitones).
//   * `deviation`   Gaussian width in value units; limits the averaging
//                   window to ±ceil(deviation / step).
//   * `threshold`   presence (voiced) gate on max-probability per frame.
//
// Populates `out_values` and `out_presence`, both length `n`.
struct GaussianBlurredResult {
    std::vector<float>         values;
    std::vector<std::uint8_t>  presence;
};

GaussianBlurredResult decode_gaussian_blurred_probs(
    const float * probs,
    std::size_t   n,
    std::size_t   bins,
    float         min_val,
    float         max_val,
    float         deviation,
    float         threshold);

// Convert per-frame boolean boundaries to region IDs starting from 1.
// Boundaries are "starts-of-new-region" markers; position 0 always belongs
// to region 1.  Padding positions (mask=false) receive 0.
std::vector<std::int32_t> boundaries_to_regions(
    const std::uint8_t * boundaries,
    const std::uint8_t * mask,
    std::size_t len);

// Convert known per-word durations (seconds) to boolean boundaries at frame
// positions.  Boundaries mark the *last* frame of each word, matching the
// PyTorch `format_boundaries` helper.
std::vector<std::uint8_t> format_boundaries(
    const float * durations, std::size_t n_words,
    std::size_t   n_frames,
    float         timestep);

}  // namespace game_ggml
