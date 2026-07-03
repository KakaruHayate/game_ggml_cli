#include "ops_basic.h"

#include <ggml.h>

namespace game_ggml::internal::ops {

ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * x,
                       ggml_tensor * weight,
                       float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    // Broadcast-multiply by weight (ne0 matches; higher dims broadcast).
    return ggml_mul(ctx, n, weight);
}

ggml_tensor * linear(ggml_context * ctx,
                     ggml_tensor * x,
                     ggml_tensor * weight,
                     ggml_tensor * bias) {
    // ggml_mul_mat(A, B):  A(ne0=K, ne1=M)  B(ne0=K, ne1=N, ...) -> (ne0=M, ne1=N, ...)
    // For Linear with PyTorch weight shape (out, in) stored row-major, we
    // have ggml ne0=in, ne1=out.  Input x has ne0=in, so the result has
    // ne0=out — exactly what Linear produces.
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    if (bias) {
        // `ggml_add` broadcasts `b` (ne0=out, higher dims=1) across higher
        // dims of `y`; this works on both CPU and Metal backends.
        y = ggml_add(ctx, y, bias);
    }
    return y;
}

ggml_tensor * layer_scale(ggml_context * ctx,
                          ggml_tensor * x,
                          ggml_tensor * scale) {
    return ggml_mul(ctx, x, scale);
}

ggml_tensor * embedding(ggml_context * ctx,
                        ggml_tensor * weight,
                        ggml_tensor * indices) {
    return ggml_get_rows(ctx, weight, indices);
}

}  // namespace game_ggml::internal::ops
