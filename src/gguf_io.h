#pragma once

// Internal GGUF I/O helpers.  Not part of the public API.

#include "game_ggml/config.h"
#include "game_ggml/errors.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct gguf_context;

namespace game_ggml::internal {

// RAII wrapper around `gguf_context`.  Loads *metadata only* by default; pass
// `load_tensors = true` to also allocate the tensor table into `tensor_ctx`
// (used by the real model loader in later tasks).
class GgufFile {
public:
    // Open a GGUF file.  Throws `GgufError` on failure.  When `load_tensors`
    // is false the file is opened in "metadata only" mode; the full tensor
    // table is still enumerated (so tensor sizes/shapes are queryable) but
    // the heavy data isn't touched.
    static GgufFile open(const std::string & path);

    ~GgufFile();
    GgufFile(GgufFile &&) noexcept;
    GgufFile & operator=(GgufFile &&) noexcept;
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    // Raw handle.  The GGUF reader owns it; borrow for short lifetimes only.
    gguf_context * handle() const noexcept { return ctx_; }

    // File path it was loaded from, for diagnostics.
    const std::string & path() const noexcept { return path_; }

    // ---- KV accessors (all throw GgufError on missing / wrong-type) ----
    bool        has(const std::string & key) const;
    std::string get_string(const std::string & key) const;
    int64_t     get_int(const std::string & key) const;
    float       get_float(const std::string & key) const;
    bool        get_bool(const std::string & key) const;

    // Optional variants: return nullopt if the key is missing; throw GgufError
    // only when the key is present with the wrong type.
    std::optional<std::string> get_string_opt(const std::string & key) const;
    std::optional<int64_t>     get_int_opt(const std::string & key) const;
    std::optional<float>       get_float_opt(const std::string & key) const;
    std::optional<bool>        get_bool_opt(const std::string & key) const;

    // ---- Tensor table ----
    struct TensorInfo {
        std::string           name;
        int32_t               type;        // ggml_type value
        std::vector<int64_t>  shape;       // leading dim is innermost (ne[0])
        size_t                size_bytes;  // on-disk payload size
        size_t                offset;      // offset into the file's tensor-data region
    };
    std::vector<TensorInfo> list_tensors() const;

    // Full list of KV keys, in file order.  Used by the inspect subcommand.
    std::vector<std::string> list_keys() const;

    // Total parameter count (sum of numel across all tensors).
    size_t total_params() const;

private:
    GgufFile() = default;

    gguf_context * ctx_ = nullptr;
    std::string    path_;
};

// Parse a `GameModelConfig` from the KV data of an opened GGUF file.
// The `architecture` key is asserted to equal `game-me`.
GameModelConfig load_config(const GgufFile & file);

// Parse a flat-object JSON string of the form `{"en":1,"ja":2,...}` into a
// map.  Accepts optional whitespace and unescaped ASCII identifiers as keys.
// Any syntactic error throws `InvalidArgument`.  Exposed here so that
// test_gguf_io can exercise it in isolation.
std::map<std::string, int> parse_flat_int_object(const std::string & json);

}  // namespace game_ggml::internal
