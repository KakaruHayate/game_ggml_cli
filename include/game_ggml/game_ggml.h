#pragma once

// Umbrella header for the public C++ API of game_ggml.
//
// Third-party consumers should only need to `#include <game_ggml/game_ggml.h>`
// to reach the full public surface.  No ggml internals are exposed.

#include "game_ggml/errors.h"
#include "game_ggml/version.h"

// -----------------------------------------------------------------------------
// Forward declarations for types populated by later tasks.  Keeping the full
// definitions out of this umbrella header means consumers never need to see
// ggml's own headers (which are an implementation detail).
// -----------------------------------------------------------------------------
namespace game_ggml {

struct GameModelConfig;        // defined in game_ggml/config.h (task 2)
struct WaveformView;           // defined in game_ggml/types.h (task 9)
struct InferParams;            // defined in game_ggml/types.h (task 9)
struct InferResult;            // defined in game_ggml/types.h (task 9)

class Model;                   // defined in game_ggml/model.h  (task 9)
class MelExtractor;            // defined in game_ggml/mel.h    (task 6)

// Enumeration of ggml compute backends this build can attempt to initialize,
// in priority order.  The string form used by `available_backends()` is the
// canonical identifier ("cpu", "metal", "cuda", "vulkan").
enum class Backend {
    CPU,
    Metal,
    CUDA,
    Vulkan,
};

// Returns the list of backends this library was compiled with, ordered from
// highest to lowest preference.  Always non-empty (CPU is always compiled in).
const char * const * available_backends() noexcept;
int available_backends_count() noexcept;

}  // namespace game_ggml
