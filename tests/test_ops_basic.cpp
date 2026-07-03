// Numerical-parity tests for the basic ops (task 3) against PyTorch dumps.
//
// These tests require the .bin files produced by
//   python ggml_backend/scripts/dump_reference.py --category basic --out tests/data/
// to be present under ggml_backend/tests/data/basic/.
// If the files are missing the tests are skipped (so CI that hasn't yet run
// the dump step still produces a clean signal).

#include <gtest/gtest.h>

#include <ggml.h>

#include "support/ggml_test_env.h"
#include "support/reference_io.h"
#include "../src/ops_basic.h"

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace {

using namespace game_ggml::test;
namespace ops = game_ggml::internal::ops;

class OpsBasic : public GgmlEnv {
protected:
    // If the basic reference dump isn't present, skip the test — but print an
    // instructive line so the user knows how to produce it.
    void require_dumps_or_skip() {
        auto p = ref_data_path("basic", "rms_norm_input");
        if (!fs::exists(p)) {
            GTEST_SKIP() << "reference dump not found at " << p
                         << "; run scripts/dump_reference.py --category basic --out tests/data/";
        }
    }
};

// --- RMSNorm --------------------------------------------------------------

TEST_F(OpsBasic, RmsNorm) {
    require_dumps_or_skip();
    auto x_ref = load_ref(ref_data_path("basic", "rms_norm_input"));
    auto w_ref = load_ref(ref_data_path("basic", "rms_norm_weight"));
    auto y_ref = load_ref(ref_data_path("basic", "rms_norm_output"));

    // PyTorch [B, T, D] -> ggml ne0=D, ne1=T, ne2=B
    ASSERT_EQ(x_ref.shape.size(), 3u);
    const int64_t B = x_ref.shape[0], T = x_ref.shape[1], D = x_ref.shape[2];

    auto * x = new_input_f32({D, T, B}, "x");
    auto * w = new_input_f32({D}, "weight");
    auto * y = ops::rms_norm(ctx(), x, w, 1e-6f);
    set_output(y);

    upload_raw(x, x_ref.as_f32(), x_ref.numel() * sizeof(float));
    upload_raw(w, w_ref.as_f32(), w_ref.numel() * sizeof(float));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-5f, /*atol=*/1e-6f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_basic.rms_norm] max_abs_err = %.3e\n", err);
}

// --- Linear (with bias) ---------------------------------------------------

TEST_F(OpsBasic, Linear) {
    require_dumps_or_skip();
    auto x_ref = load_ref(ref_data_path("basic", "linear_input"));
    auto w_ref = load_ref(ref_data_path("basic", "linear_weight"));
    auto b_ref = load_ref(ref_data_path("basic", "linear_bias"));
    auto y_ref = load_ref(ref_data_path("basic", "linear_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1], I = x_ref.shape[2];
    const int64_t O = w_ref.shape[0];
    ASSERT_EQ(w_ref.shape[1], I);

    auto * x = new_input_f32({I, T, B}, "x");
    auto * w = new_input_f32({I, O}, "w");
    auto * b = new_input_f32({O}, "b");
    auto * y = ops::linear(ctx(), x, w, b);
    set_output(y);

    upload_raw(x, x_ref.as_f32(), x_ref.numel() * sizeof(float));
    upload_raw(w, w_ref.as_f32(), w_ref.numel() * sizeof(float));
    upload_raw(b, b_ref.as_f32(), b_ref.numel() * sizeof(float));
    compute();

    auto out = download_f32(y);
    // Metal FP32 matmul precision on small-magnitude results is ~1e-3 due to
    // GPU fma accumulator limits (documented in llama.cpp's test suite).
    // This tolerance still catches correctness errors (wrong matmul direction
    // would give >0.1 errors) while passing legitimate GPU noise.
    float err = compare_f32(y_ref, out, /*rtol=*/1e-3f, /*atol=*/1e-3f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_basic.linear] max_abs_err = %.3e\n", err);
}

// --- Linear (no bias) -----------------------------------------------------

TEST_F(OpsBasic, LinearNoBias) {
    require_dumps_or_skip();
    auto x_ref = load_ref(ref_data_path("basic", "linear_nb_input"));
    auto w_ref = load_ref(ref_data_path("basic", "linear_nb_weight"));
    auto y_ref = load_ref(ref_data_path("basic", "linear_nb_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1], I = x_ref.shape[2];
    const int64_t O = w_ref.shape[0];

    auto * x = new_input_f32({I, T, B});
    auto * w = new_input_f32({I, O});
    auto * y = ops::linear(ctx(), x, w, nullptr);
    set_output(y);

    upload_raw(x, x_ref.as_f32(), x_ref.numel() * sizeof(float));
    upload_raw(w, w_ref.as_f32(), w_ref.numel() * sizeof(float));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-3f, /*atol=*/1e-3f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_basic.linear_nb] max_abs_err = %.3e\n", err);
}

// --- LayerScale -----------------------------------------------------------

TEST_F(OpsBasic, LayerScale) {
    require_dumps_or_skip();
    auto x_ref = load_ref(ref_data_path("basic", "layer_scale_input"));
    auto s_ref = load_ref(ref_data_path("basic", "layer_scale_scale"));
    auto y_ref = load_ref(ref_data_path("basic", "layer_scale_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1], D = x_ref.shape[2];

    auto * x = new_input_f32({D, T, B});
    auto * s = new_input_f32({D});
    auto * y = ops::layer_scale(ctx(), x, s);
    set_output(y);

    upload_raw(x, x_ref.as_f32(), x_ref.numel() * sizeof(float));
    upload_raw(s, s_ref.as_f32(), s_ref.numel() * sizeof(float));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-6f, /*atol=*/1e-7f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_basic.layer_scale] max_abs_err = %.3e\n", err);
}

// --- Embedding ------------------------------------------------------------

TEST_F(OpsBasic, Embedding) {
    require_dumps_or_skip();
    auto w_ref   = load_ref(ref_data_path("basic", "embedding_weight"));
    auto idx_ref = load_ref(ref_data_path("basic", "embedding_indices"));
    auto y_ref   = load_ref(ref_data_path("basic", "embedding_output"));

    const int64_t V = w_ref.shape[0], D = w_ref.shape[1];
    const int64_t B = idx_ref.shape[0], T = idx_ref.shape[1];

    // Weight in ggml: ne0=D, ne1=V (PyTorch [V, D] row-major).
    // Indices are int32 and ggml_get_rows accepts 1-D indices per row.  For
    // (B, T) we flatten to a 1D tensor of length B*T; the output comes back
    // as [D, B*T] and we compare flat since the reference is stored flat too.
    auto * w   = new_input_f32({D, V}, "weight");
    auto * idx = new_input_i32({B * T}, "indices");
    auto * y   = ops::embedding(ctx(), w, idx);
    set_output(y);

    upload_raw(w, w_ref.as_f32(), w_ref.numel() * sizeof(float));
    upload_raw(idx, idx_ref.as_i32(), idx_ref.numel() * sizeof(int32_t));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-6f, /*atol=*/1e-7f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_basic.embedding] max_abs_err = %.3e\n", err);
}

// --- Cyclic region embedding ---------------------------------------------

TEST_F(OpsBasic, CyclicRegionEmbedding) {
    require_dumps_or_skip();
    auto w_ref       = load_ref(ref_data_path("basic", "cyclic_weight"));
    auto raw_idx_ref = load_ref(ref_data_path("basic", "cyclic_raw_indices"));
    auto mod_idx_ref = load_ref(ref_data_path("basic", "cyclic_mod_indices"));
    auto y_ref       = load_ref(ref_data_path("basic", "cyclic_output"));

    const int64_t cycle = w_ref.shape[0], D = w_ref.shape[1];
    const int64_t B = raw_idx_ref.shape[0], T = raw_idx_ref.shape[1];

    // Sanity: applying modulo in C++ matches the pre-computed modded indices.
    std::vector<int32_t> modded(B * T);
    for (size_t i = 0; i < modded.size(); ++i) {
        const int32_t v = raw_idx_ref.as_i32()[i];
        modded[i] = ((v % cycle) + cycle) % cycle;
    }
    for (size_t i = 0; i < modded.size(); ++i) {
        ASSERT_EQ(modded[i], mod_idx_ref.as_i32()[i]);
    }

    auto * w   = new_input_f32({D, cycle}, "weight");
    auto * idx = new_input_i32({B * T}, "mod_indices");
    auto * y   = ops::embedding(ctx(), w, idx);
    set_output(y);

    upload_raw(w, w_ref.as_f32(), w_ref.numel() * sizeof(float));
    upload_raw(idx, modded.data(), modded.size() * sizeof(int32_t));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-6f, /*atol=*/1e-7f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_basic.cyclic_embedding] max_abs_err = %.3e\n", err);
}

}  // namespace
