#include "ops_rope.h"

#include "game_ggml/errors.h"

#include <ggml.h>

#include <cassert>

namespace game_ggml::internal::ops {

namespace {
// Thin wrapper hiding the YaRN parameters we never use.
ggml_tensor * rope_normal(ggml_context * ctx,
                          ggml_tensor * x,
                          ggml_tensor * positions,
                          int n_dims,
                          float theta) {
    return ggml_rope_ext(
        ctx, x, positions,
        /*freq_factors=*/nullptr,
        n_dims,
        /*mode=*/GGML_ROPE_TYPE_NORMAL,
        /*n_ctx_orig=*/0,
        /*freq_base=*/theta,
        /*freq_scale=*/1.0f,
        /*ext_factor=*/0.0f,
        /*attn_factor=*/1.0f,
        /*beta_fast=*/0.0f,
        /*beta_slow=*/0.0f);
}
}  // namespace

ggml_tensor * apply_rope(ggml_context * ctx,
                         ggml_tensor * x,
                         ggml_tensor * positions,
                         int n_dims,
                         float theta) {
    return rope_normal(ctx, x, positions, n_dims, theta);
}

ggml_tensor * region_rope(ggml_context * ctx,
                          ggml_tensor * x,
                          RegionRopeMode mode,
                          ggml_tensor * global_positions,
                          ggml_tensor * region_indices,
                          ggml_tensor * local_positions,
                          int head_dim,
                          float theta) {
    if (mode == RegionRopeMode::Local) {
        if (!local_positions) throw InvalidArgument("region_rope(Local): local_positions required");
        return rope_normal(ctx, x, local_positions, head_dim, theta);
    }
    if (mode == RegionRopeMode::Global) {
        if (!global_positions) throw InvalidArgument("region_rope(Global): global_positions required");
        return rope_normal(ctx, x, global_positions, head_dim, theta);
    }
    // --- Mixed: split head_dim in half, use global positions on the lower half
    //           and region indices on the upper half.
    if (!global_positions) throw InvalidArgument("region_rope(Mixed): global_positions required");
    if (!region_indices)   throw InvalidArgument("region_rope(Mixed): region_indices required");

    assert(head_dim % 2 == 0);
    const int half = head_dim / 2;
    const size_t esize = ggml_element_size(x);

    // View the two halves of x along ne[0].  Views share memory with x so
    // they have matching higher-dim strides.
    ggml_tensor * lo = ggml_view_4d(ctx, x,
        half, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        /*offset=*/0);
    ggml_tensor * hi = ggml_view_4d(ctx, x,
        half, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        /*offset=*/half * esize);

    // RoPE requires contiguous inputs.
    ggml_tensor * lo_c = ggml_cont(ctx, lo);
    ggml_tensor * hi_c = ggml_cont(ctx, hi);

    ggml_tensor * lo_r = rope_normal(ctx, lo_c, global_positions, half, theta);
    ggml_tensor * hi_r = rope_normal(ctx, hi_c, region_indices,   half, theta);

    // Concat along ne[0] to restore full head_dim.
    return ggml_concat(ctx, lo_r, hi_r, /*dim=*/0);
}

}  // namespace game_ggml::internal::ops
