#include "ggml_test_env.h"

#include "../../src/backend.h"

#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace game_ggml::test {

namespace {
constexpr size_t kCtxMemSize = 64 * 1024 * 1024;   // 64 MB of node metadata
}

void GgmlEnv::SetUp() {
    backend_ = game_ggml::internal::init_best_backend();

    ggml_init_params ip{};
    ip.mem_size   = kCtxMemSize;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;  // use gallocr to back the tensors on the device
    ctx_ = ggml_init(ip);
    ASSERT_NE(ctx_, nullptr);

    // Max-sized graph suffices for unit tests.  We can shrink this per-test
    // if it ever matters.
    graph_ = ggml_new_graph_custom(ctx_, /*size=*/4096, /*grads=*/false);
    ASSERT_NE(graph_, nullptr);
}

void GgmlEnv::TearDown() {
    if (gallocr_) { ggml_gallocr_free(gallocr_); gallocr_ = nullptr; }
    if (ctx_)     { ggml_free(ctx_);            ctx_     = nullptr; }
    if (backend_) { game_ggml::internal::free_backend(backend_); backend_ = nullptr; }
    inputs_.clear();
    graph_ = nullptr;
    output_set_ = false;
}

ggml_tensor * GgmlEnv::new_input_f32(std::initializer_list<int64_t> ne, const char * name) {
    std::vector<int64_t> dims(ne);
    while (dims.size() < 4) dims.push_back(1);
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, dims[0], dims[1], dims[2], dims[3]);
    // Trim the dims that are genuinely unused so shape reflects user's intent.
    // ggml tensors always carry 4 ne[] entries; nothing to do here beyond naming.
    if (name) ggml_set_name(t, name);
    ggml_set_input(t);
    inputs_.push_back(t);
    return t;
}

ggml_tensor * GgmlEnv::new_input_i32(std::initializer_list<int64_t> ne, const char * name) {
    std::vector<int64_t> dims(ne);
    while (dims.size() < 4) dims.push_back(1);
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, GGML_TYPE_I32, dims[0], dims[1], dims[2], dims[3]);
    if (name) ggml_set_name(t, name);
    ggml_set_input(t);
    inputs_.push_back(t);
    return t;
}

void GgmlEnv::set_output(ggml_tensor * out) {
    ASSERT_FALSE(output_set_) << "set_output called twice";
    ASSERT_NE(out, nullptr);
    ggml_set_output(out);
    ggml_set_name(out, "_out");
    ggml_build_forward_expand(graph_, out);

    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    ASSERT_NE(gallocr_, nullptr);
    const bool ok = ggml_gallocr_alloc_graph(gallocr_, graph_);
    ASSERT_TRUE(ok) << "ggml_gallocr_alloc_graph failed";
    output_set_ = true;
}

void GgmlEnv::upload_raw(ggml_tensor * t, const void * data, size_t nbytes) {
    ASSERT_TRUE(output_set_) << "upload must be called after set_output";
    ASSERT_EQ(ggml_nbytes(t), nbytes)
        << "upload size mismatch for tensor '" << (t->name[0] ? t->name : "?")
        << "': tensor=" << ggml_nbytes(t) << " got=" << nbytes;
    ggml_backend_tensor_set(t, data, 0, nbytes);
}

void GgmlEnv::compute() {
    ASSERT_TRUE(output_set_);
    const ggml_status s = ggml_backend_graph_compute(backend_, graph_);
    ASSERT_EQ(s, GGML_STATUS_SUCCESS) << "graph_compute failed with status " << static_cast<int>(s);
}

std::vector<float> GgmlEnv::download_f32(ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    return out;
}

std::vector<int32_t> GgmlEnv::download_i32(ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    std::vector<int32_t> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(int32_t));
    return out;
}

}  // namespace game_ggml::test
