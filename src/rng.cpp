#include "rng.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace game_ggml::internal {

MT19937Rng::MT19937Rng(std::uint64_t seed) : gen_(seed) {}

void MT19937Rng::uniform(float * dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = dist_(gen_);
}

InjectedRng::InjectedRng(std::vector<float> values) : values_(std::move(values)) {}

void InjectedRng::uniform(float * dst, std::size_t n) {
    if (cursor_ + n > values_.size()) {
        throw std::out_of_range(
            "InjectedRng exhausted: needed " + std::to_string(n) +
            " more, have " + std::to_string(values_.size() - cursor_));
    }
    for (std::size_t i = 0; i < n; ++i) dst[i] = values_[cursor_ + i];
    cursor_ += n;
}

}  // namespace game_ggml::internal
