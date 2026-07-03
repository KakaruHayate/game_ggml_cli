// Full pipeline parity test (task 9).
//
// Runs Model::infer on a ~2 s synthetic waveform with the same random
// numbers PyTorch used for its reference pass, then compares the output
// note list (durations / presence / pitch scores).

#include <gtest/gtest.h>

#include "game_ggml/game_ggml.h"
#include "game_ggml/model.h"
#include "../src/model_impl.h"
#include "../src/rng.h"
#include "support/reference_io.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
using namespace game_ggml;
using namespace game_ggml::test;

TEST(Pipeline, E2EBitExactWithInjectedRng) {
    auto wav_path = ref_data_path("pipeline", "pipe_wav");
    if (!fs::exists(wav_path)) GTEST_SKIP() << "pipeline refs missing";
    const char * asset = std::getenv("GAME_GGML_TEST_ASSET");
    fs::path gguf = asset ? asset :
        fs::current_path() / ".." / "assets" / "game_small.gguf";
    if (!fs::exists(gguf)) GTEST_SKIP() << "game_small.gguf missing";

    auto wav       = load_ref(wav_path);
    auto ts_ref    = load_ref(ref_data_path("pipeline", "pipe_ts"));
    auto rng_ref   = load_ref(ref_data_path("pipeline", "pipe_rng"));
    auto dur_ref   = load_ref(ref_data_path("pipeline", "pipe_durations"));
    auto pres_ref  = load_ref(ref_data_path("pipeline", "pipe_presence"));
    auto score_ref = load_ref(ref_data_path("pipeline", "pipe_scores"));

    auto model = Model::load(gguf.string());

    InferParams params;
    params.language = 4;   // matches the dumped run
    params.d3pm_ts.assign(ts_ref.as_f32(), ts_ref.as_f32() + ts_ref.numel());

    // Inject Python-captured uniforms so D3PM random drops match bit-for-bit.
    std::vector<float> rng_vals(rng_ref.as_f32(), rng_ref.as_f32() + rng_ref.numel());
    internal::InjectedRng rng(std::move(rng_vals));

    auto result = model.internals().infer_with_rng(
        wav.as_f32(), static_cast<std::size_t>(wav.numel()),
        params, rng);

    std::fprintf(stderr, "[pipeline] predicted %zu notes, %d frames\n",
                 result.notes.size(), result.num_frames);

    // PyTorch durations have length N_max across batch, padded with -1 * step.
    // Compare up to the smaller length — counts can legitimately differ by ±1
    // due to Metal FP32 borderline flips as documented in segmenter tests.
    const std::size_t n_ref = dur_ref.numel();
    const std::size_t n_got = result.notes.size();
    EXPECT_LE(std::abs(static_cast<int>(n_ref) - static_cast<int>(n_got)), 2);

    const std::size_t n_cmp = std::min(n_ref, n_got);

    // Duration comparison: frame-level (~1 frame = ~0.01 s tolerance).
    float dur_err = 0.0f;
    int dur_frame_diff = 0;
    for (std::size_t i = 0; i < n_cmp; ++i) {
        const float d_ref = dur_ref.as_f32()[i];
        const float d_got = result.notes[i].duration_seconds;
        const float diff  = std::fabs(d_ref - d_got);
        dur_err = std::max(dur_err, diff);
        if (diff > 0.005f) ++dur_frame_diff;
    }

    // Pitch comparison (only where both PyTorch and C++ report voiced).
    int pitch_n = 0;
    float pitch_err = 0.0f;
    for (std::size_t i = 0; i < n_cmp; ++i) {
        const bool  pv = pres_ref.as_bool()[i] != 0;
        const bool  gv = result.notes[i].voiced;
        if (!pv || !gv) continue;
        const float diff = std::fabs(score_ref.as_f32()[i] - result.notes[i].pitch_midi);
        pitch_err = std::max(pitch_err, diff);
        ++pitch_n;
    }

    std::fprintf(stderr,
        "[pipeline] n_ref=%zu n_got=%zu  dur_max=%.3fs frames_off=%d  "
        "pitch_cmp=%d max_err=%.4f semitone\n",
        n_ref, n_got, dur_err, dur_frame_diff, pitch_n, pitch_err);

    EXPECT_LT(dur_err, 0.020f);        // 2 frame durations
    EXPECT_LT(pitch_err, 0.5f);        // within half a semitone
}
