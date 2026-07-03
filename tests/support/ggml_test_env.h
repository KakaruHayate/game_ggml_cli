#pragma once

// Common GoogleTest fixture: owns a ggml backend, a compute context, and a
// graph-allocator.  Provides helpers to upload inputs / download outputs.
//
// Typical usage:
//
//   class MyOp : public game_ggml::test::GgmlEnv {};
//   TEST_F(MyOp, SomeCase) {
//       auto * x = new_input_f32({B, T, D});
//       auto * w = new_input_f32({D});
//       auto * y = ops::rms_norm(ctx(), x, w);
//       set_output(y);
//       upload(x, x_data.data());
//       upload(w, w_data.data());
//       compute();
//       auto out = download_f32(y);
//       ...
//   }

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_backend *  ggml_backend_t;
typedef struct ggml_gallocr * ggml_gallocr_t;

namespace game_ggml::test {

class GgmlEnv : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    // Construct a new input tensor.  Allocated later by gallocr when we build
    // the graph.  Shape order follows ggml convention: first entry becomes
    // ne[0], i.e. the innermost dim.
    ggml_tensor * new_input_f32(std::initializer_list<int64_t> ne, const char * name = nullptr);
    ggml_tensor * new_input_i32(std::initializer_list<int64_t> ne, const char * name = nullptr);

    // Build the graph (allocates memory) and mark `out` as the single sink.
    void set_output(ggml_tensor * out);

    // Copy user data into an input tensor (must have been registered via
    // new_input_*).  `nbytes` must match the tensor's allocated size.
    void upload_raw(ggml_tensor * t, const void * data, size_t nbytes);

    template <class T>
    void upload(ggml_tensor * t, const std::vector<T> & v) {
        upload_raw(t, v.data(), v.size() * sizeof(T));
    }

    // Run the graph on the backend.  Call exactly once after set_output.
    void compute();

    // Read back the tensor values.  Must be called after compute().
    std::vector<float>   download_f32(ggml_tensor * t);
    std::vector<int32_t> download_i32(ggml_tensor * t);

    // Raw accessors.
    ggml_context * ctx() const noexcept { return ctx_; }
    ggml_backend_t backend() const noexcept { return backend_; }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context * ctx_     = nullptr;
    ggml_cgraph  * graph_   = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<ggml_tensor *> inputs_;
    bool output_set_ = false;
};

}  // namespace game_ggml::test
