#include "tensor_utils.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace game_ggml::internal {

namespace {
constexpr size_t kMetaCtxBytes = 2ull * 1024 * 1024;  // 2 MB suffices for 700-tensor metadata
}

LoadedWeights LoadedWeights::load_all(const GgufFile & gguf, ggml_backend_t backend) {
    // --- 1. Build a metadata-only ggml_context that mirrors the file's
    //        tensor table.  gguf_init_from_file with params.ctx != null will
    //        create ggml_tensor entries in that context but leave them
    //        un-backed (no_alloc=true).
    ggml_context * ctx = nullptr;
    {
        ggml_init_params ip{};
        ip.mem_size   = kMetaCtxBytes;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        ctx = ggml_init(ip);
        if (!ctx) throw GgufError("failed to create weight ggml_context");
    }

    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = &ctx;

    gguf_context * gctx = gguf_init_from_file(gguf.path().c_str(), gp);
    if (!gctx) {
        ggml_free(ctx);
        throw GgufError("failed to re-open GGUF for tensor loading: " + gguf.path());
    }

    // --- 2. Allocate backend buffer covering all tensors in ctx.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        gguf_free(gctx);
        ggml_free(ctx);
        throw GgufError("ggml_backend_alloc_ctx_tensors failed; out of memory?");
    }

    // --- 3. Upload tensor payloads directly from the file.
    FILE * f = std::fopen(gguf.path().c_str(), "rb");
    if (!f) {
        ggml_backend_buffer_free(buf);
        gguf_free(gctx);
        ggml_free(ctx);
        throw GgufError("failed to open GGUF payload file: " + gguf.path());
    }

    const size_t data_offset = gguf_get_data_offset(gctx);
    std::vector<std::uint8_t> scratch;

    LoadedWeights out;
    out.ctx_    = ctx;
    out.buffer_ = buf;

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            std::fclose(f);
            ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(ctx);
            throw GgufError(std::string("tensor '") + name + "' missing from ggml context");
        }
        const size_t bytes  = ggml_nbytes(t);
        const size_t offset = data_offset + gguf_get_tensor_offset(gctx, i);
        scratch.resize(bytes);
        if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0 ||
            std::fread(scratch.data(), 1, bytes, f) != bytes) {
            std::fclose(f);
            ggml_backend_buffer_free(buf);
            gguf_free(gctx);
            ggml_free(ctx);
            throw GgufError(std::string("short read for tensor '") + name + "'");
        }
        ggml_backend_tensor_set(t, scratch.data(), 0, bytes);
        out.tensors_.emplace(name, t);
    }

    std::fclose(f);
    gguf_free(gctx);

    return out;
}

LoadedWeights::~LoadedWeights() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_)    ggml_free(ctx_);
}

LoadedWeights::LoadedWeights(LoadedWeights && other) noexcept
    : ctx_(other.ctx_), buffer_(other.buffer_), tensors_(std::move(other.tensors_)) {
    other.ctx_ = nullptr;
    other.buffer_ = nullptr;
}

LoadedWeights & LoadedWeights::operator=(LoadedWeights && other) noexcept {
    if (this != &other) {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_)    ggml_free(ctx_);
        ctx_     = other.ctx_;
        buffer_  = other.buffer_;
        tensors_ = std::move(other.tensors_);
        other.ctx_ = nullptr;
        other.buffer_ = nullptr;
    }
    return *this;
}

ggml_tensor * LoadedWeights::get(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw GgufError("required tensor missing: " + name);
    }
    return it->second;
}

ggml_tensor * LoadedWeights::try_get(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

}  // namespace game_ggml::internal
