#pragma once

#include <stdexcept>
#include <string>

namespace game_ggml {

// Base class for all library exceptions.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string & msg) : std::runtime_error(msg) {}
};

// Thrown when an argument violates API contracts (e.g. wrong sample rate).
class InvalidArgument : public Error {
public:
    explicit InvalidArgument(const std::string & msg) : Error(msg) {}
};

// Thrown when a GGUF file cannot be opened, parsed, or does not contain a
// required tensor/metadata key.
class GgufError : public Error {
public:
    explicit GgufError(const std::string & msg) : Error(msg) {}
};

// Thrown when a WAV file is malformed or has an unexpected sample rate /
// channel count.
class InvalidWav : public Error {
public:
    explicit InvalidWav(const std::string & msg) : Error(msg) {}
};

// Thrown when a requested backend (Metal/CUDA/Vulkan) is unavailable at
// runtime, or when none of the configured backends could be initialized.
class BackendError : public Error {
public:
    explicit BackendError(const std::string & msg) : Error(msg) {}
};

// Thrown when the loaded model requests a configuration branch that this
// implementation does not yet handle (e.g. split attention, learned pool
// merger).  Callers can catch this specifically to give users actionable
// guidance.
class NotImplemented : public Error {
public:
    explicit NotImplemented(const std::string & msg) : Error(msg) {}
};

}  // namespace game_ggml
