#include "gguf_io.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <utility>

namespace game_ggml::internal {

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

GgufFile GgufFile::open(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;    // don't allocate the tensor data region
    params.ctx      = nullptr; // we don't want ggml tensors materialised here

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        throw GgufError("failed to open GGUF file: " + path);
    }
    GgufFile f;
    f.ctx_  = ctx;
    f.path_ = path;
    return f;
}

GgufFile::~GgufFile() {
    if (ctx_) gguf_free(ctx_);
}

GgufFile::GgufFile(GgufFile && other) noexcept
    : ctx_(other.ctx_), path_(std::move(other.path_)) {
    other.ctx_ = nullptr;
}

GgufFile & GgufFile::operator=(GgufFile && other) noexcept {
    if (this != &other) {
        if (ctx_) gguf_free(ctx_);
        ctx_  = other.ctx_;
        path_ = std::move(other.path_);
        other.ctx_ = nullptr;
    }
    return *this;
}

bool GgufFile::has(const std::string & key) const {
    return gguf_find_key(ctx_, key.c_str()) >= 0;
}

// -- typed getters --

namespace {
    [[noreturn]] void wrong_type(const std::string & key, const char * wanted, const char * got) {
        throw GgufError("GGUF key '" + key + "' has wrong type: wanted " + wanted + ", got " + got);
    }
    [[noreturn]] void missing(const std::string & key) {
        throw GgufError("GGUF key '" + key + "' is missing");
    }
}

std::string GgufFile::get_string(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    if (t != GGUF_TYPE_STRING) wrong_type(key, "STRING", gguf_type_name(t));
    return std::string(gguf_get_val_str(ctx_, id));
}

int64_t GgufFile::get_int(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    switch (t) {
        case GGUF_TYPE_INT8:   return gguf_get_val_i8  (ctx_, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16 (ctx_, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32 (ctx_, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64 (ctx_, id);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8  (ctx_, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16 (ctx_, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32 (ctx_, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx_, id));
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(ctx_, id) ? 1 : 0;
        default: wrong_type(key, "INT*", gguf_type_name(t));
    }
}

float GgufFile::get_float(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    switch (t) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx_, id);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx_, id));
        case GGUF_TYPE_INT32:   return static_cast<float>(gguf_get_val_i32(ctx_, id));
        case GGUF_TYPE_INT64:   return static_cast<float>(gguf_get_val_i64(ctx_, id));
        default: wrong_type(key, "FLOAT*", gguf_type_name(t));
    }
}

bool GgufFile::get_bool(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    if (t != GGUF_TYPE_BOOL) wrong_type(key, "BOOL", gguf_type_name(t));
    return gguf_get_val_bool(ctx_, id);
}

std::optional<std::string> GgufFile::get_string_opt(const std::string & key) const {
    return has(key) ? std::optional<std::string>(get_string(key)) : std::nullopt;
}
std::optional<int64_t> GgufFile::get_int_opt(const std::string & key) const {
    return has(key) ? std::optional<int64_t>(get_int(key)) : std::nullopt;
}
std::optional<float> GgufFile::get_float_opt(const std::string & key) const {
    return has(key) ? std::optional<float>(get_float(key)) : std::nullopt;
}
std::optional<bool> GgufFile::get_bool_opt(const std::string & key) const {
    return has(key) ? std::optional<bool>(get_bool(key)) : std::nullopt;
}

// -- tensor table --

std::vector<GgufFile::TensorInfo> GgufFile::list_tensors() const {
    const int64_t n = gguf_get_n_tensors(ctx_);
    std::vector<TensorInfo> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        TensorInfo info;
        info.name = gguf_get_tensor_name(ctx_, i);
        info.type = static_cast<int32_t>(gguf_get_tensor_type(ctx_, i));
        info.size_bytes = gguf_get_tensor_size(ctx_, i);
        info.offset     = gguf_get_tensor_offset(ctx_, i);
        // GGUF tensors store ne[] inline; we query it via the ggml_tensor
        // structure created by gguf_init_from_file when params.ctx is null,
        // so instead we recover the shape from the public helper by
        // re-opening with a context.  As a lightweight alternative that
        // avoids re-opening, we walk the gguf internal struct via the
        // n_dims / ne[] read by the library during init — but the public
        // C API exposes these only through the ggml_tensor path.  For the
        // inspector we can still render useful information using
        // size_bytes + type, and shape is filled in the richer loader in
        // later tasks.  Leave `shape` empty here for now.
        out.push_back(std::move(info));
    }
    return out;
}

std::vector<std::string> GgufFile::list_keys() const {
    const int64_t n = gguf_get_n_kv(ctx_);
    std::vector<std::string> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_key(ctx_, i));
    }
    return out;
}

size_t GgufFile::total_params() const {
    // Accurate param count requires reading shape arrays; the simplest path
    // that works for all ggml types is to re-open with a ggml_context and
    // count via ggml_nelements().  We prefer not to pay that cost in the
    // inspector, so estimate from tensor sizes assuming FP32 (true for this
    // project's v1 schema).  If mixed types appear later we'll switch to
    // the exact path here.
    const int64_t n = gguf_get_n_tensors(ctx_);
    size_t params = 0;
    for (int64_t i = 0; i < n; ++i) {
        const auto t    = gguf_get_tensor_type(ctx_, i);
        const size_t sz = gguf_get_tensor_size(ctx_, i);
        // For FP32 assume 4 bytes/element.  For other types we'd need the
        // per-type block_size / type_size from ggml_type_traits.
        if (t == GGML_TYPE_F32) {
            params += sz / 4;
        } else {
            params += sz;
        }
    }
    return params;
}

// ---------------------------------------------------------------------------
// JSON parsing for flat {"str":int} objects
// ---------------------------------------------------------------------------

namespace {

struct JsonCursor {
    const std::string & s;
    size_t i = 0;

    bool eof() const { return i >= s.size(); }
    char peek() const { return eof() ? '\0' : s[i]; }
    char next() { return eof() ? '\0' : s[i++]; }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    [[noreturn]] void fail(const std::string & msg) const {
        throw InvalidArgument("bad JSON at offset " + std::to_string(i) + ": " + msg);
    }

    void expect(char c) {
        skip_ws();
        if (next() != c) fail(std::string("expected '") + c + "'");
    }

    std::string parse_string() {
        skip_ws();
        if (next() != '"') fail("expected string");
        std::string out;
        while (!eof()) {
            char c = next();
            if (c == '"') return out;
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"': out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/');  break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    default:  fail("unsupported escape");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    int parse_int() {
        skip_ws();
        std::string tok;
        if (peek() == '-' || peek() == '+') tok.push_back(next());
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) tok.push_back(next());
        if (tok.empty() || tok == "-" || tok == "+") fail("expected integer");
        try {
            return std::stoi(tok);
        } catch (...) {
            fail("integer out of range");
        }
    }
};

}  // namespace

std::map<std::string, int> parse_flat_int_object(const std::string & json) {
    JsonCursor c{json};
    c.expect('{');
    std::map<std::string, int> out;
    c.skip_ws();
    if (c.peek() == '}') { c.next(); return out; }

    while (true) {
        auto key = c.parse_string();
        c.expect(':');
        int value = c.parse_int();
        out.emplace(std::move(key), value);
        c.skip_ws();
        char nxt = c.next();
        if (nxt == ',') continue;
        if (nxt == '}') break;
        c.fail("expected ',' or '}'");
    }
    c.skip_ws();
    if (!c.eof()) c.fail("trailing content");
    return out;
}

// ---------------------------------------------------------------------------
// Config loader
// ---------------------------------------------------------------------------

namespace {

void fill_ebf_backbone(const GgufFile & f, const std::string & section, BackboneConfig & b, bool expect_latent) {
    const std::string p = "game." + section + ".";
    b.cls           = f.get_string_opt(p + "cls").value_or("");
    b.dim           = static_cast<int>(f.get_int(p + "dim"));
    b.num_layers    = static_cast<int>(f.get_int(p + "num_layers"));
    b.num_heads     = static_cast<int>(f.get_int(p + "num_heads"));
    b.head_dim      = static_cast<int>(f.get_int(p + "head_dim"));
    b.c_kernel_size = static_cast<int>(f.get_int(p + "c_kernel_size"));
    b.m_kernel_size = static_cast<int>(f.get_int(p + "m_kernel_size"));
    b.ffn_type      = f.get_string_opt(p + "ffn_type").value_or("glu");
    b.use_ls        = f.get_bool_opt(p + "use_ls").value_or(true);
    b.use_out_norm  = f.get_bool_opt(p + "use_out_norm").value_or(true);
    b.skip_first_ffn = f.get_bool_opt(p + "skip_first_ffn").value_or(false);
    b.skip_out_ffn   = f.get_bool_opt(p + "skip_out_ffn").value_or(false);

    if (expect_latent) {
        if (auto v = f.get_int_opt(p + "latent_layer_idx"); v.has_value()) {
            b.return_latent    = true;
            b.latent_layer_idx = static_cast<int>(*v);
            b.latent_out_dim   = static_cast<int>(f.get_int(p + "latent_out_dim"));
        }
    }
}

void fill_jebf_backbone(const GgufFile & f, const std::string & section, BackboneConfig & b) {
    const std::string p = "game." + section + ".";
    b.cls                = f.get_string_opt(p + "cls").value_or("");
    b.dim                = static_cast<int>(f.get_int(p + "dim"));
    b.num_layers         = static_cast<int>(f.get_int(p + "num_layers"));
    b.num_heads          = static_cast<int>(f.get_int(p + "num_heads"));
    b.head_dim           = static_cast<int>(f.get_int(p + "head_dim"));
    b.ffn_type           = f.get_string_opt(p + "ffn_type").value_or("glu");
    b.use_ls             = f.get_bool_opt(p + "use_ls").value_or(true);
    b.use_out_norm       = f.get_bool_opt(p + "use_out_norm").value_or(true);
    b.skip_first_ffn     = f.get_bool_opt(p + "skip_first_ffn").value_or(false);
    b.skip_out_ffn       = f.get_bool_opt(p + "skip_out_ffn").value_or(false);
    b.region_token_num   = static_cast<int>(f.get_int(p + "region_token_num"));
    b.pool_merge_mode    = f.get_string_opt(p + "pool_merge_mode").value_or("mean");
    b.attn_type          = f.get_string_opt(p + "attn_type").value_or("joint");
    b.rope_mode          = f.get_string_opt(p + "rope_mode").value_or("mixed");
    b.qk_norm            = f.get_bool_opt(p + "qk_norm").value_or(true);
    b.use_region_bias    = f.get_bool_opt(p + "use_region_bias").value_or(false);
    b.c_kernel_size_pool = static_cast<int>(f.get_int(p + "c_kernel_size_pool"));
    b.m_kernel_size_pool = static_cast<int>(f.get_int(p + "m_kernel_size_pool"));
    b.c_kernel_size_x    = static_cast<int>(f.get_int(p + "c_kernel_size_x"));
    b.m_kernel_size_x    = static_cast<int>(f.get_int(p + "m_kernel_size_x"));
    b.use_rope           = f.get_bool_opt(p + "use_rope").value_or(true);
    b.use_pool_offset    = f.get_bool_opt(p + "use_pool_offset").value_or(false);
    b.theta              = f.get_float_opt(p + "theta").value_or(10000.0f);
}

}  // namespace

GameModelConfig load_config(const GgufFile & f) {
    GameModelConfig c{};
    c.architecture = f.get_string("general.architecture");
    if (c.architecture != "game-me") {
        throw GgufError("unsupported GGUF architecture '" + c.architecture +
                        "' (expected 'game-me')");
    }
    c.name    = f.get_string_opt("general.name").value_or("");
    c.version = f.get_string_opt("general.version").value_or("");

    // Top-level model
    c.mode              = f.get_string("game.model.mode");
    c.embedding_dim     = static_cast<int>(f.get_int("game.model.embedding_dim"));
    c.in_dim            = static_cast<int>(f.get_int("game.model.in_dim"));
    c.estimator_out_dim = static_cast<int>(f.get_int("game.model.estimator_out_dim"));
    c.region_cycle_len  = static_cast<int>(f.get_int("game.model.region_cycle_len"));
    c.use_languages     = f.get_bool("game.model.use_languages");
    c.num_languages     = static_cast<int>(f.get_int("game.model.num_languages"));

    fill_ebf_backbone (f, "encoder",   c.encoder,   /*expect_latent=*/false);
    fill_ebf_backbone (f, "segmenter", c.segmenter, /*expect_latent=*/true);
    fill_jebf_backbone(f, "estimator", c.estimator);

    // Inference
    auto & inf = c.inference;
    inf.audio_sample_rate  = static_cast<int>(f.get_int("game.inference.audio_sample_rate"));
    inf.hop_size           = static_cast<int>(f.get_int("game.inference.hop_size"));
    inf.fft_size           = static_cast<int>(f.get_int("game.inference.fft_size"));
    inf.win_size           = static_cast<int>(f.get_int("game.inference.win_size"));
    inf.n_mels             = static_cast<int>(f.get_int("game.inference.spectrogram.num_bins"));
    inf.fmin               = f.get_float("game.inference.spectrogram.fmin");
    inf.fmax               = f.get_float("game.inference.spectrogram.fmax");
    inf.spectrogram_type   = f.get_string_opt("game.inference.spectrogram.type").value_or("mel");
    inf.midi_min           = f.get_float("game.inference.midi_min");
    inf.midi_max           = f.get_float("game.inference.midi_max");
    inf.midi_num_bins      = static_cast<int>(f.get_int("game.inference.midi_num_bins"));
    inf.midi_std           = f.get_float("game.inference.midi_std");

    if (c.use_languages) {
        auto raw = f.get_string_opt("game.inference.lang_map");
        if (raw.has_value() && !raw->empty()) {
            inf.lang_map = parse_flat_int_object(*raw);
        }
    }

    return c;
}

}  // namespace game_ggml::internal
