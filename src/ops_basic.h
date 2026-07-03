#pragma once

// Internal building blocks (graph builders) for elementary ops shared across
// encoder / segmenter / estimator.  Each function takes a ggml context and
// the relevant input / weight tensors and returns the result node.
//
// Not part of the public API.

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal::ops {

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight.
//   - `x`:      [ne0, ne1, ne2, ne3]  — normalized along ne0.
//   - `weight`: [ne0]                 — elementwise scale (PyTorch `weight`).
ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * x,
                       ggml_tensor * weight,
                       float eps = 1e-6f);

// Linear (fully-connected): y = x @ W.T + b.
//   - `x`:      [in, T, ...]          — PyTorch shape [..., in], i.e. ggml ne0=in.
//   - `weight`: [in, out]             — PyTorch weight (out, in) row-major == ggml ne0=in ne1=out.
//   - `bias`:   [out] or nullptr      — optional.
// Returns a tensor with ne0=out (matching PyTorch nn.Linear).
ggml_tensor * linear(ggml_context * ctx,
                     ggml_tensor * x,
                     ggml_tensor * weight,
                     ggml_tensor * bias);

// LayerScale: y = x * scale  (elementwise along ne0, broadcast on higher dims).
//   - `x`:      [D, ...]
//   - `scale`:  [D]
ggml_tensor * layer_scale(ggml_context * ctx,
                          ggml_tensor * x,
                          ggml_tensor * scale);

// Embedding lookup: y[i] = weight[idx[i]].
//   - `weight`:  [D, V]            — ggml ne0=D, ne1=V (PyTorch nn.Embedding weight row-major).
//   - `indices`: int32 [N...]      — any rank; output is [D, N...].
// Thin wrapper around ggml_get_rows, kept as a named helper for clarity and
// in case we later need to add dtype coercion.
ggml_tensor * embedding(ggml_context * ctx,
                        ggml_tensor * weight,
                        ggml_tensor * indices);

}  // namespace game_ggml::internal::ops
