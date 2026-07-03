#pragma once

// RMS-silence-based waveform slicer.  Port of GAME's `inference/slicer2.py`.
// Each chunk carries its offset back to the original waveform so the caller
// can glue per-chunk inferences into a global timeline.

#include <cstddef>
#include <string>
#include <vector>

namespace game_ggml::cli {

struct SliceChunk {
    double             offset_seconds;
    std::vector<float> waveform;   // mono samples
};

struct SlicerConfig {
    int    sample_rate   = 44100;
    double threshold_db  = -40.0;  // silence if rms < 10^(threshold_db/20)
    int    min_length_ms = 5000;
    int    min_interval_ms = 300;
    int    hop_ms        = 20;
    int    max_sil_kept_ms = 5000;
};

// Slice a mono float waveform into speech chunks.  The slicer assumes the
// caller has already validated the sample rate.  Returns at least one chunk;
// for very short clips the entire input is returned as a single chunk.
std::vector<SliceChunk> slice_waveform(
    const float * samples, std::size_t n,
    const SlicerConfig & cfg);

}  // namespace game_ggml::cli
