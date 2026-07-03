#pragma once

// Internal backend selection helpers.  NOT part of the public API —
// consumers should use the handles returned by Model instead.

#include "game_ggml/game_ggml.h"

struct ggml_backend;
typedef ggml_backend * ggml_backend_t;

namespace game_ggml::internal {

// Initialise the highest-priority backend this build supports, falling back
// to CPU.  Never returns null; throws `BackendError` if even CPU fails.
ggml_backend_t init_best_backend();

// Initialise a specific backend.  Returns null if the backend is compiled-in
// but cannot be started on this system (e.g. no Metal device).
ggml_backend_t init_backend(Backend which);

// Free a backend created by this module.  Safe to call with null.
void free_backend(ggml_backend_t backend);

// Human-readable name of `backend`'s type, for logging.
const char * backend_name(ggml_backend_t backend);

}  // namespace game_ggml::internal
