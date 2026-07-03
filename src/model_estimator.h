#pragma once

// Estimator (JEBFBackbone) — consumes `x_est` from the encoder plus the
// predicted region assignment, and emits a per-region probability
// distribution over MIDI pitch bins.
//
// Uses joint attention between `pool` tokens (one per region) and `x`
// frame tokens, with a region-aware mask + mixed RoPE.

#include "game_ggml/config.h"
#include "ops_joint_attn.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal {

class LoadedWeights;

struct EstimatorWeights {
    ggml_tensor * w_input_proj     = nullptr;
    ggml_tensor * b_input_proj     = nullptr;
    ggml_tensor * w_pool_token_gen = nullptr;   // (R, D)
    ggml_tensor * w_region_embedding = nullptr; // CyclicRegion, (cycle, D)
    std::vector<ops::JEBFBlockWeights> layers;
    ggml_tensor * w_output_norm_x    = nullptr;
    ggml_tensor * w_output_norm_pool = nullptr;
    ggml_tensor * w_output_proj_x    = nullptr;
    ggml_tensor * b_output_proj_x    = nullptr;
    ggml_tensor * w_output_proj_pool = nullptr;
    ggml_tensor * b_output_proj_pool = nullptr;
};

EstimatorWeights bind_estimator_weights(
    const LoadedWeights & W,
    const GameModelConfig & cfg);

// Build the estimator graph and return the pool-branch logits (the one
// `forward_estimation` returns as the note distribution).
struct EstimatorOutputs {
    ggml_tensor * pool_logits;   // (bins, N, 1)
};

EstimatorOutputs build_estimator_graph(
    ggml_context * ctx,
    ggml_tensor * x_est,                // (D, T, 1)
    ggml_tensor * regions_mod3,         // int32 (T,)  -- regions % cycle_len
    ggml_tensor * positions,            // int32 (N+T,) -- (0..N-1, 0..T-1)
    ggml_tensor * region_indices,       // int32 (N+T,) -- (0*N, regions+1 or similar)
    ggml_tensor * attn_mask_fp16,       // (N+T, N+T, 1, 1)
    int N,
    const EstimatorWeights & W,
    const GameModelConfig & cfg);

}  // namespace game_ggml::internal
