#pragma once

// Mel spectrogram extractor matching lib.feature.mel.StretchableMelSpectrogram
// in the GAME training code.  This is the canonical front-end used by both
// the library API and internal unit tests.

#include <cstddef>
#include <memory>
#include <vector>

namespace game_ggml {

// Construction parameters — defaults mirror the 1.0-small checkpoint:
// 44100 Hz mono in, 80-bin mel, hop=441 → timestep = 10 ms.
struct MelConfig {
    int   sample_rate = 44100;
    int   n_fft       = 2048;
    int   win_length  = 2048;
    int   hop_length  = 441;
    int   n_mels      = 80;
    float fmin        = 0.0f;
    float fmax        = 8000.0f;
    float clip_val    = 1e-5f;   // dynamic-range compression floor
};

class MelExtractor {
public:
    explicit MelExtractor(const MelConfig & cfg);
    ~MelExtractor();
    MelExtractor(MelExtractor &&) noexcept;
    MelExtractor & operator=(MelExtractor &&) noexcept;
    MelExtractor(const MelExtractor &) = delete;
    MelExtractor & operator=(const MelExtractor &) = delete;

    const MelConfig & config() const noexcept;

    // Number of output frames for a given waveform length.  Matches
    // StretchableMelSpectrogram's reflect-pad + center=False convention.
    int num_frames(std::size_t n_samples) const noexcept;

    // Extract a mel spectrogram.  `waveform` is 44100 Hz mono float in
    // [-1, 1].  Returns a row-major [T, n_mels] buffer.
    std::vector<float> forward(const float * waveform, std::size_t n_samples) const;

    std::vector<float> forward(const std::vector<float> & waveform) const {
        return forward(waveform.data(), waveform.size());
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace game_ggml
