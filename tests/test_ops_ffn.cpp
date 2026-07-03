// Numerical-parity tests for GLUFFN + CgMLP (task 4).

#include <gtest/gtest.h>

#include <ggml.h>

#include "support/ggml_test_env.h"
#include "support/reference_io.h"
#include "../src/ops_ffn.h"

#include <filesystem>

namespace fs = std::filesystem;
using namespace game_ggml::test;
namespace ops = game_ggml::internal::ops;

namespace {

class OpsFfn : public GgmlEnv {
protected:
    void require_dumps_or_skip() {
        auto p = ref_data_path("ffn", "glu_input");
        if (!fs::exists(p)) {
            GTEST_SKIP() << "reference dump not found at " << p
                         << "; run scripts/dump_reference.py --category ffn --out tests/data/";
        }
    }
};

// ---- GLUFFN -------------------------------------------------------------

TEST_F(OpsFfn, GluFfn) {
    require_dumps_or_skip();
    auto x_ref = load_ref(ref_data_path("ffn", "glu_input"));
    auto w1    = load_ref(ref_data_path("ffn", "glu_w_ln1"));
    auto b1    = load_ref(ref_data_path("ffn", "glu_b_ln1"));
    auto w2    = load_ref(ref_data_path("ffn", "glu_w_ln2"));
    auto b2    = load_ref(ref_data_path("ffn", "glu_b_ln2"));
    auto y_ref = load_ref(ref_data_path("ffn", "glu_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1], D = x_ref.shape[2];
    const int64_t L2 = w1.shape[0];       // 2*L
    const int64_t L  = L2 / 2;

    auto * x  = new_input_f32({D, T, B}, "x");
    auto * W1 = new_input_f32({D,  L2},  "w_ln1");
    auto * B1 = new_input_f32({L2},      "b_ln1");
    auto * W2 = new_input_f32({L,  D},   "w_ln2");
    auto * B2 = new_input_f32({D},       "b_ln2");
    auto * y  = ops::glu_ffn(ctx(), x, W1, B1, W2, B2);
    set_output(y);

    upload_raw(x,  x_ref.as_f32(), x_ref.numel() * sizeof(float));
    upload_raw(W1, w1.as_f32(),    w1.numel() * sizeof(float));
    upload_raw(B1, b1.as_f32(),    b1.numel() * sizeof(float));
    upload_raw(W2, w2.as_f32(),    w2.numel() * sizeof(float));
    upload_raw(B2, b2.as_f32(),    b2.numel() * sizeof(float));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-3f, /*atol=*/1e-3f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_ffn.glu] max_abs_err = %.3e\n", err);
}

// ---- CgMLP --------------------------------------------------------------

TEST_F(OpsFfn, CgMLP) {
    require_dumps_or_skip();
    auto x_ref  = load_ref(ref_data_path("ffn", "cg_input"));
    auto w_pw1  = load_ref(ref_data_path("ffn", "cg_w_pw1"));
    auto b_pw1  = load_ref(ref_data_path("ffn", "cg_b_pw1"));
    auto w_norm = load_ref(ref_data_path("ffn", "cg_w_norm"));
    auto w_dw   = load_ref(ref_data_path("ffn", "cg_w_dw"));
    auto b_dw   = load_ref(ref_data_path("ffn", "cg_b_dw"));
    auto w_pw2  = load_ref(ref_data_path("ffn", "cg_w_pw2"));
    auto b_pw2  = load_ref(ref_data_path("ffn", "cg_b_pw2"));
    auto y_ref  = load_ref(ref_data_path("ffn", "cg_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1], D = x_ref.shape[2];
    // pw1 weight shape in PyTorch is (2L, D, 1); in GGUF/ggml ne=(1, D, 2L).
    ASSERT_EQ(w_pw1.shape.size(), 3u);
    const int64_t L2 = w_pw1.shape[0];  const int64_t L = L2 / 2;
    const int64_t K  = w_dw.shape[2];   // PyTorch Conv1d weight (L, 1, K) -> ne=(K, 1, L)

    auto * x = new_input_f32({D, T, B}, "x");

    // Weights: we upload them with the *PyTorch* row-major shape so the
    // memory layout reaching ggml matches what the converter would write.
    // w_pw1 PyTorch (2L, D, 1) -> ggml 3D tensor ne=(1, D, 2L).
    auto * W_pw1 = new_input_f32({1, D, L2},  "w_pw1");
    auto * B_pw1 = new_input_f32({L2},        "b_pw1");
    auto * W_norm= new_input_f32({L},         "w_norm");
    auto * W_dw  = new_input_f32({K, 1, L},   "w_dw");
    auto * B_dw  = new_input_f32({L},         "b_dw");
    auto * W_pw2 = new_input_f32({1, L, D},   "w_pw2");
    auto * B_pw2 = new_input_f32({D},         "b_pw2");

    auto * y = ops::cgmlp(ctx(), x,
        W_pw1, B_pw1, W_norm,
        W_dw,  B_dw,
        W_pw2, B_pw2,
        /*kernel_size=*/(int)K);
    set_output(y);

    upload_raw(x,     x_ref.as_f32(),  x_ref.numel()  * sizeof(float));
    upload_raw(W_pw1, w_pw1.as_f32(),  w_pw1.numel()  * sizeof(float));
    upload_raw(B_pw1, b_pw1.as_f32(),  b_pw1.numel()  * sizeof(float));
    upload_raw(W_norm,w_norm.as_f32(), w_norm.numel() * sizeof(float));
    upload_raw(W_dw,  w_dw.as_f32(),   w_dw.numel()   * sizeof(float));
    upload_raw(B_dw,  b_dw.as_f32(),   b_dw.numel()   * sizeof(float));
    upload_raw(W_pw2, w_pw2.as_f32(),  w_pw2.numel()  * sizeof(float));
    upload_raw(B_pw2, b_pw2.as_f32(),  b_pw2.numel()  * sizeof(float));
    compute();

    auto out = download_f32(y);
    // CgMLP stacks 3 linear-ish projections + conv + gelus; cumulative GPU
    // FP32 error is noticeably larger than a single matmul.
    float err = compare_f32(y_ref, out, /*rtol=*/1e-3f, /*atol=*/2e-3f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_ffn.cgmlp] max_abs_err = %.3e\n", err);
}

}  // namespace
