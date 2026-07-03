#pragma once

// Version constants surfaced as preprocessor macros by the build system.
// Prefer `game_ggml::version_string()` for runtime display.

#ifndef GAME_GGML_VERSION_MAJOR
#define GAME_GGML_VERSION_MAJOR 0
#endif
#ifndef GAME_GGML_VERSION_MINOR
#define GAME_GGML_VERSION_MINOR 1
#endif
#ifndef GAME_GGML_VERSION_PATCH
#define GAME_GGML_VERSION_PATCH 0
#endif

namespace game_ggml {

// Returns the project version as "major.minor.patch".
const char * version_string() noexcept;

// Returns the ggml runtime version (e.g. "0.11.0") that the library was
// compiled against.  Useful for logging.
const char * ggml_version_string() noexcept;

}  // namespace game_ggml
