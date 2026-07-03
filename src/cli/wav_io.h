#pragma once

// Simple WAV loader living in the CLI layer (dr_wav-backed).
// The core library is not coupled to dr_wav.

#include <cstddef>
#include <string>
#include <vector>

namespace game_ggml::cli {

struct WavFile {
    std::vector<float> samples;    // always mono, downmixed if necessary
    int sample_rate = 0;
    int channels    = 0;           // original channel count
};

// Load a WAV file and downmix to mono float32.  Throws `InvalidWav` when the
// file can't be parsed or when `expected_sample_rate` is nonzero and differs.
WavFile load_wav_mono_f32(const std::string & path, int expected_sample_rate = 0);

// Minimal PCM-16 mono WAV writer — used in tests to fabricate inputs.
void write_wav_mono_i16(const std::string & path, const float * samples,
                        std::size_t n, int sample_rate);

}  // namespace game_ggml::cli
