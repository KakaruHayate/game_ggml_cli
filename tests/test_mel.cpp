// Mel spectrogram + WAV loader parity tests (task 6).

#include <gtest/gtest.h>

#include "support/reference_io.h"
#include "game_ggml/mel.h"
#include "game_ggml/errors.h"
#include "../src/cli/wav_io.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
using namespace game_ggml;
using namespace game_ggml::test;

TEST(Mel, MatchesPyTorchOnNoise) {
    auto wav_path = ref_data_path("mel", "mel_wav");
    auto out_path = ref_data_path("mel", "mel_output");
    if (!fs::exists(wav_path) || !fs::exists(out_path)) {
        GTEST_SKIP() << "mel reference dumps missing; run dump_reference.py --category mel";
    }
    auto wav_ref = load_ref(wav_path);
    auto ref_out = load_ref(out_path);

    ASSERT_EQ(wav_ref.shape.size(), 1u);
    const int n = static_cast<int>(wav_ref.shape[0]);

    MelConfig cfg;   // defaults match the small checkpoint
    MelExtractor mel(cfg);
    auto got = mel.forward(wav_ref.as_f32(), n);

    ASSERT_EQ(ref_out.shape.size(), 2u);
    const int T = static_cast<int>(ref_out.shape[0]);
    const int N = static_cast<int>(ref_out.shape[1]);
    ASSERT_EQ(got.size(), static_cast<std::size_t>(T) * N);

    const float err = compare_f32(ref_out, got, /*rtol=*/1e-3f, /*atol=*/1e-3f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[mel] T=%d n_mels=%d max_abs_err=%.3e\n", T, N, err);
}

TEST(WavIO, SineRoundtrip) {
    const int sr = 44100;
    const int n  = sr;  // 1 second
    std::vector<float> sine(n);
    for (int i = 0; i < n; ++i) {
        sine[i] = 0.5f * std::sin(2.0f * 3.14159265358979f * 440.0f * i / sr);
    }
    auto tmp = fs::temp_directory_path() / "game_ggml_wav_test.wav";
    cli::write_wav_mono_i16(tmp.string(), sine.data(), n, sr);

    auto wav = cli::load_wav_mono_f32(tmp.string(), sr);
    EXPECT_EQ(wav.sample_rate, sr);
    ASSERT_EQ(wav.samples.size(), static_cast<std::size_t>(n));
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(wav.samples[i] - sine[i]));
    }
    EXPECT_LT(max_err, 2e-4f);   // 1/32767 i16 rounding noise
    std::fprintf(stderr, "[wav] max_abs_err=%.3e\n", max_err);
    fs::remove(tmp);
}

TEST(WavIO, WrongSampleRateThrows) {
    const int sr = 22050;
    std::vector<float> dummy(sr, 0.0f);
    auto tmp = fs::temp_directory_path() / "game_ggml_wav_wrongsr.wav";
    cli::write_wav_mono_i16(tmp.string(), dummy.data(), dummy.size(), sr);
    EXPECT_THROW(cli::load_wav_mono_f32(tmp.string(), 44100), game_ggml::InvalidWav);
    fs::remove(tmp);
}
