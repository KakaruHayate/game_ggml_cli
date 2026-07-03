#pragma once

// Abstract random-source interface so tests can inject PyTorch-dumped
// randomness for bit-exact D3PM parity while production inference uses
// the platform's `std::mt19937`.
//
// Internal use only.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace game_ggml::internal {

class IRandomSource {
public:
    virtual ~IRandomSource() = default;

    // Fill `n` uniformly-distributed floats in [0, 1) into `dst`.  Callers
    // must ensure `dst` is sized appropriately.
    virtual void uniform(float * dst, std::size_t n) = 0;
};

// Production RNG: seedable Mersenne Twister producing `uniform_real` in [0,1).
class MT19937Rng : public IRandomSource {
public:
    explicit MT19937Rng(std::uint64_t seed);
    void uniform(float * dst, std::size_t n) override;

private:
    std::mt19937_64 gen_;
    std::uniform_real_distribution<float> dist_{0.0f, 1.0f};
};

// Test RNG: reads from a pre-loaded buffer.  Aborts if the buffer runs out.
class InjectedRng : public IRandomSource {
public:
    explicit InjectedRng(std::vector<float> values);
    void uniform(float * dst, std::size_t n) override;
    std::size_t remaining() const { return values_.size() - cursor_; }

private:
    std::vector<float> values_;
    std::size_t cursor_ = 0;
};

}  // namespace game_ggml::internal
