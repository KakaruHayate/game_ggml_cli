// Estimator E2E parity test (task 8).

#include <gtest/gtest.h>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include "support/reference_io.h"
#include "../src/backend.h"
#include "../src/gguf_io.h"
#include "../src/model_estimator.h"
#include "../src/ops_joint_attn.h"
#include "../src/tensor_utils.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
namespace gi = game_ggml::internal;
using namespace game_ggml::test;

TEST(Estimator, FullForwardParity) {
    auto est_x = ref_data_path("estimator", "est_x_est");
    if (!fs::exists(est_x)) GTEST_SKIP() << "estimator dumps missing";
    const char * asset = std::getenv("GAME_GGML_TEST_ASSET");
    fs::path gguf_path = asset ? asset :
        fs::current_path() / ".." / "assets" / "game_small.gguf";
    if (!fs::exists(gguf_path)) GTEST_SKIP() << "game_small.gguf missing";

    auto gguf = gi::GgufFile::open(gguf_path.string());
    auto cfg  = gi::load_config(gguf);
    auto backend = gi::init_best_backend();
    auto weights = gi::LoadedWeights::load_all(gguf, backend);
    auto est_w = gi::bind_estimator_weights(weights, cfg);

    auto x_ref       = load_ref(est_x);
    auto regions_ref = load_ref(ref_data_path("estimator", "est_regions"));
    auto logits_ref  = load_ref(ref_data_path("estimator", "est_logits"));

    ASSERT_EQ(x_ref.shape.size(), 2u);      // [T, D]
    const int64_t T = x_ref.shape[0], D = x_ref.shape[1];
    // Determine N = max region.
    int N = 0;
    for (int i = 0; i < T; ++i) N = std::max<int>(N, regions_ref.as_i32()[i]);
    const int S = N + T;
    ASSERT_EQ(D, cfg.embedding_dim);

    // Build graph.
    ggml_init_params ip{};
    ip.mem_size = 512 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * xest        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, T, 1);
    ggml_tensor * regions_mod = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * positions   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * region_ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * mask_fp16   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, 1);
    for (auto * t : {xest, regions_mod, positions, region_ids, mask_fp16}) ggml_set_input(t);

    auto outs = gi::build_estimator_graph(ctx, xest, regions_mod, positions,
        region_ids, mask_fp16, N, est_w, cfg);
    ggml_set_output(outs.pool_logits);
    ggml_build_forward_expand(graph, outs.pool_logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, graph));

    // Upload inputs.
    ggml_backend_tensor_set(xest, x_ref.as_f32(), 0, ggml_nbytes(xest));

    std::vector<int32_t> rmod(T);
    for (int i = 0; i < T; ++i) rmod[i] = regions_ref.as_i32()[i] % cfg.region_cycle_len;
    ggml_backend_tensor_set(regions_mod, rmod.data(), 0, rmod.size() * sizeof(int32_t));

    // Global positions: pool 0..N-1, x 0..T-1.
    std::vector<int32_t> gpos(S);
    for (int i = 0; i < N; ++i) gpos[i] = i;
    for (int i = 0; i < T; ++i) gpos[N + i] = i;
    ggml_backend_tensor_set(positions, gpos.data(), 0, gpos.size() * sizeof(int32_t));

    // Region indices for mixed RoPE: pool = zeros (R=1, use_pool_offset=false),
    // x = local_positions_within_region + R (= +1).
    std::vector<int32_t> ridx(S, 0);
    {
        int cur_region = 0;
        int cur_local  = 0;
        for (int i = 0; i < T; ++i) {
            const int r = regions_ref.as_i32()[i];
            if (r != cur_region) { cur_region = r; cur_local = 0; }
            ridx[N + i] = (r > 0) ? (cur_local + 1) : 0;
            ++cur_local;
        }
    }
    ggml_backend_tensor_set(region_ids, ridx.data(), 0, ridx.size() * sizeof(int32_t));

    // Attention mask.
    auto mask = game_ggml::internal::ops::build_joint_attn_mask_fp16(
        regions_ref.as_i32(), static_cast<int>(T), N);
    ggml_backend_tensor_set(mask_fp16, mask.data(), 0, mask.size() * sizeof(uint16_t));

    ASSERT_EQ(ggml_backend_graph_compute(backend, graph), GGML_STATUS_SUCCESS);

    const int bins = cfg.estimator_out_dim;
    std::vector<float> got(ggml_nelements(outs.pool_logits));
    ggml_backend_tensor_get(outs.pool_logits, got.data(), 0, got.size() * sizeof(float));
    ASSERT_EQ(got.size(), static_cast<std::size_t>(N * bins));

    const float err = compare_f32(logits_ref, got, /*rtol=*/3e-3f, /*atol=*/5e-3f);
    EXPECT_FALSE(std::isnan(err));
    std::fprintf(stderr, "[estimator] N=%d bins=%d max_abs_err=%.3e\n", N, bins, err);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    gi::free_backend(backend);
}
