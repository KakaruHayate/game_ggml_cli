// Numerical-parity tests for RoPE variants (task 4).

#include <gtest/gtest.h>

#include <ggml.h>

#include "support/ggml_test_env.h"
#include "support/reference_io.h"
#include "../src/ops_rope.h"

#include <filesystem>

namespace fs = std::filesystem;
using namespace game_ggml::test;
namespace ops = game_ggml::internal::ops;

namespace {

class OpsRope : public GgmlEnv {
protected:
    void require_dumps_or_skip() {
        auto p = ref_data_path("rope", "single_x");
        if (!fs::exists(p)) {
            GTEST_SKIP() << "reference dump not found at " << p
                         << "; run scripts/dump_reference.py --category rope --out tests/data/";
        }
    }
};

// ---- Single-position RoPE (global arange) ------------------------------

TEST_F(OpsRope, SingleRope) {
    require_dumps_or_skip();
    auto x_ref   = load_ref(ref_data_path("rope", "single_x"));
    auto pos_ref = load_ref(ref_data_path("rope", "single_positions"));
    auto y_ref   = load_ref(ref_data_path("rope", "single_output"));

    // Python dumped tensor with shape [B, T, H, D] (already transposed so the
    // flat bytes match ggml ne = (D, H, T, B) — the order ggml_rope_ext
    // expects: ne[2] = n_tokens).
    ASSERT_EQ(x_ref.shape.size(), 4u);
    const int64_t B = x_ref.shape[0], T = x_ref.shape[1],
                  H = x_ref.shape[2], D = x_ref.shape[3];

    auto * x   = new_input_f32({D, H, T, B}, "x");
    auto * pos = new_input_i32({T},          "positions");
    auto * y   = ops::apply_rope(ctx(), x, pos, /*n_dims=*/D, /*theta=*/10000.0f);
    set_output(y);

    upload_raw(x,   x_ref.as_f32(), x_ref.numel() * sizeof(float));
    upload_raw(pos, pos_ref.as_i32(), pos_ref.numel() * sizeof(int32_t));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-4f, /*atol=*/1e-5f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_rope.single] max_abs_err = %.3e\n", err);
}

// ---- Region RoPE (local mode) ------------------------------------------

TEST_F(OpsRope, RegionRopeLocal) {
    require_dumps_or_skip();
    auto x_ref   = load_ref(ref_data_path("rope", "local_x"));
    auto pos_ref = load_ref(ref_data_path("rope", "local_positions"));
    auto y_ref   = load_ref(ref_data_path("rope", "local_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1],
                  H = x_ref.shape[2], D = x_ref.shape[3];

    auto * x   = new_input_f32({D, H, T, B}, "x");
    auto * pos = new_input_i32({T},          "local_positions");
    auto * y   = ops::region_rope(ctx(), x,
        ops::RegionRopeMode::Local,
        /*global=*/nullptr, /*region=*/nullptr, /*local=*/pos,
        /*head_dim=*/(int)D);
    set_output(y);

    upload_raw(x,   x_ref.as_f32(),   x_ref.numel() * sizeof(float));
    upload_raw(pos, pos_ref.as_i32(), pos_ref.numel() * sizeof(int32_t));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-4f, /*atol=*/1e-5f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_rope.region_local] max_abs_err = %.3e\n", err);
}

// ---- Region RoPE (mixed mode) ------------------------------------------

TEST_F(OpsRope, RegionRopeMixed) {
    require_dumps_or_skip();
    auto x_ref       = load_ref(ref_data_path("rope", "mixed_x"));
    auto gpos_ref    = load_ref(ref_data_path("rope", "mixed_global_pos"));
    auto ridx_ref    = load_ref(ref_data_path("rope", "mixed_region_idx"));
    auto y_ref       = load_ref(ref_data_path("rope", "mixed_output"));

    const int64_t B = x_ref.shape[0], T = x_ref.shape[1],
                  H = x_ref.shape[2], D = x_ref.shape[3];

    auto * x    = new_input_f32({D, H, T, B}, "x");
    auto * gpos = new_input_i32({T},          "global_pos");
    auto * ridx = new_input_i32({T},          "region_idx");
    auto * y    = ops::region_rope(ctx(), x,
        ops::RegionRopeMode::Mixed,
        gpos, ridx, /*local=*/nullptr,
        /*head_dim=*/(int)D);
    set_output(y);

    upload_raw(x,    x_ref.as_f32(),    x_ref.numel()    * sizeof(float));
    upload_raw(gpos, gpos_ref.as_i32(), gpos_ref.numel() * sizeof(int32_t));
    upload_raw(ridx, ridx_ref.as_i32(), ridx_ref.numel() * sizeof(int32_t));
    compute();

    auto out = download_f32(y);
    float err = compare_f32(y_ref, out, /*rtol=*/1e-4f, /*atol=*/1e-5f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[ops_rope.region_mixed] max_abs_err = %.3e\n", err);
}

}  // namespace
