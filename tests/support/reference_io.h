#pragma once

// Reference-dump I/O for numerical-parity tests against PyTorch.
//
// File layout (little-endian):
//   offset 0:  magic      char[4]   = "GREF"
//   offset 4:  version    int32     = 1
//   offset 8:  dtype      int32     (see RefDType)
//   offset 12: ndim       int32     in [0, 8]
//   offset 16: dims       int64[8]  only `ndim` entries are meaningful
//   offset 80: payload    raw bytes (size depends on dtype + product of dims)
//
// Dumps are produced by ggml_backend/scripts/dump_reference.py and stored
// under ggml_backend/tests/data/<category>/<name>.bin (gitignored).  They
// are loaded here and compared against ggml compute output.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace game_ggml::test {

enum class RefDType : int32_t {
    Float32 = 0,
    Float16 = 1,
    Int32   = 2,
    Int64   = 3,
    Bool    = 4,
};

struct RefTensor {
    RefDType             dtype = RefDType::Float32;
    std::vector<int64_t> shape;       // length = ndim
    std::vector<char>    payload;     // raw bytes

    // Convenience views — cast the payload for the appropriate dtype.
    const float *   as_f32()  const { return reinterpret_cast<const float   *>(payload.data()); }
    const int32_t * as_i32()  const { return reinterpret_cast<const int32_t *>(payload.data()); }
    const int64_t * as_i64()  const { return reinterpret_cast<const int64_t *>(payload.data()); }
    const uint8_t * as_bool() const { return reinterpret_cast<const uint8_t *>(payload.data()); }

    // Number of elements.
    size_t numel() const;
};

// Load a reference tensor from a file.  Throws on format / dtype mismatch.
RefTensor load_ref(const std::string & path);

// Compare two float tensors element-wise using mixed absolute+relative
// tolerance à la numpy.allclose.  Returns the max absolute error observed.
// If any pair exceeds the tolerance, the return value is NAN and the first
// few offending indices are logged to stderr.
float compare_f32(const float * a, const float * b, size_t n,
                  float rtol = 1e-5f, float atol = 1e-6f);

// Convenience: compare a RefTensor (f32) against a flat vector.
float compare_f32(const RefTensor & expected, const std::vector<float> & got,
                  float rtol = 1e-5f, float atol = 1e-6f);

// Locate a reference asset under tests/data/.  By default the loader expects
// `tests/data/` relative to the CMake source tree; the path can be overridden
// via the GAME_GGML_TEST_DATA environment variable.
std::string ref_data_path(const std::string & category, const std::string & name);

}  // namespace game_ggml::test
