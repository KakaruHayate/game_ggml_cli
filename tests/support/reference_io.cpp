#include "reference_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace game_ggml::test {

namespace {
constexpr char   kMagic[4]    = {'G','R','E','F'};
constexpr int32_t kVersion    = 1;
constexpr size_t kHeaderBytes = 4 /*magic*/ + 4 /*version*/ + 4 /*dtype*/
                              + 4 /*ndim*/  + 8 * 8 /*dims*/;

size_t element_size(RefDType t) {
    switch (t) {
        case RefDType::Float32: return 4;
        case RefDType::Float16: return 2;
        case RefDType::Int32:   return 4;
        case RefDType::Int64:   return 8;
        case RefDType::Bool:    return 1;
    }
    return 0;
}

[[noreturn]] void bad(const std::string & msg, const std::string & path) {
    throw std::runtime_error("reference_io: " + msg + " (" + path + ")");
}
}  // namespace

size_t RefTensor::numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

RefTensor load_ref(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) bad("cannot open", path);

    char magic[4];
    int32_t version = 0;
    int32_t dtype_raw = 0;
    int32_t ndim = 0;
    int64_t dims[8] = {0};

    in.read(magic, 4);
    in.read(reinterpret_cast<char *>(&version),   sizeof(version));
    in.read(reinterpret_cast<char *>(&dtype_raw), sizeof(dtype_raw));
    in.read(reinterpret_cast<char *>(&ndim),      sizeof(ndim));
    in.read(reinterpret_cast<char *>(dims),       sizeof(dims));
    if (!in) bad("short read in header", path);
    if (std::memcmp(magic, kMagic, 4) != 0) bad("bad magic", path);
    if (version != kVersion) bad("unsupported version", path);
    if (ndim < 0 || ndim > 8)  bad("bad ndim", path);

    RefTensor out;
    out.dtype = static_cast<RefDType>(dtype_raw);
    out.shape.assign(dims, dims + ndim);

    size_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= static_cast<size_t>(dims[i]);

    size_t bytes = n * element_size(out.dtype);
    out.payload.resize(bytes);
    in.read(out.payload.data(), static_cast<std::streamsize>(bytes));
    if (!in) bad("short read in payload", path);

    // Trailing data is allowed (no-op); we only read what we need.
    return out;
}

float compare_f32(const float * a, const float * b, size_t n, float rtol, float atol) {
    float max_err = 0.0f;
    int fails = 0;
    for (size_t i = 0; i < n; ++i) {
        float da = a[i], db = b[i];
        if (!std::isfinite(da) || !std::isfinite(db)) {
            if (da != db && !(std::isnan(da) && std::isnan(db))) {
                if (fails < 5) {
                    std::fprintf(stderr, "compare_f32: non-finite mismatch @%zu: %f vs %f\n", i, da, db);
                }
                ++fails;
            }
            continue;
        }
        float err = std::fabs(da - db);
        float tol = atol + rtol * std::fabs(db);
        if (err > tol) {
            if (fails < 5) {
                std::fprintf(stderr, "compare_f32: mismatch @%zu: %.8g vs %.8g "
                             "(err=%.3g, tol=%.3g)\n", i, da, db, err, tol);
            }
            ++fails;
        }
        max_err = std::max(max_err, err);
    }
    if (fails > 0) {
        std::fprintf(stderr, "compare_f32: %d/%zu elements failed tolerance\n", fails, n);
        return std::nanf("");
    }
    return max_err;
}

float compare_f32(const RefTensor & expected, const std::vector<float> & got, float rtol, float atol) {
    if (expected.dtype != RefDType::Float32) {
        std::fprintf(stderr, "compare_f32: expected tensor is not f32\n");
        return std::nanf("");
    }
    if (got.size() != expected.numel()) {
        std::fprintf(stderr, "compare_f32: size mismatch: got=%zu expected=%zu\n",
                     got.size(), expected.numel());
        return std::nanf("");
    }
    return compare_f32(got.data(), expected.as_f32(), got.size(), rtol, atol);
}

std::string ref_data_path(const std::string & category, const std::string & name) {
    const char * env = std::getenv("GAME_GGML_TEST_DATA");
    fs::path root = env ? fs::path(env)
                        : fs::path(__FILE__).parent_path().parent_path() / "data";
    return (root / category / (name + ".bin")).string();
}

}  // namespace game_ggml::test
