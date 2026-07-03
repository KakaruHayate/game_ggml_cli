#pragma once

// Rotary positional embedding primitives.
//
// All modes below use the "paired / interleaved" encoding (GGML_ROPE_TYPE_NORMAL,
// pattern [cs cs cs ... cs]) which matches the PyTorch implementation in
// modules/backbones/rope.py.
//
// Shape convention for `x` (matches the attention sub-module):
//   ne = (head_dim, n_tokens, n_heads, batch)
//
// Positions tensors are int32 with shape (n_tokens,) — shared across heads and
// batch.  This matches `ggml_rope_ext`'s expectation.

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal::ops {

// Apply paired RoPE to `x` along its first dimension (head_dim) using the
// provided positions.  `n_dims` = number of dims to rotate (typically
// head_dim); remaining dims pass through unchanged.  `theta` = freq base.
ggml_tensor * apply_rope(ggml_context * ctx,
                         ggml_tensor * x,
                         ggml_tensor * positions,   // int32 [n_tokens]
                         int n_dims,
                         float theta = 10000.0f);

// RegionRoPE mode names for `region_rope` below.
enum class RegionRopeMode { Local, Global, Mixed };

// RegionRoPE: apply RoPE with one of three strategies:
//   - Local:  single RoPE pass over the full head_dim, using `local_positions`.
//   - Global: single RoPE pass over the full head_dim, using `global_positions`.
//   - Mixed:  split head_dim in two; first half rotated by `global_positions`,
//             second half by `region_indices`.  This is the mode actually
//             used by JointAttention in GAME.
//
// Returns a new tensor with the same shape as `x`.
ggml_tensor * region_rope(ggml_context * ctx,
                          ggml_tensor * x,
                          RegionRopeMode mode,
                          ggml_tensor * global_positions,   // int32 [n_tokens] — required for Global/Mixed
                          ggml_tensor * region_indices,     // int32 [n_tokens] — required for Mixed
                          ggml_tensor * local_positions,    // int32 [n_tokens] — required for Local
                          int head_dim,
                          float theta = 10000.0f);

}  // namespace game_ggml::internal::ops
