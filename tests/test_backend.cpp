// Task 1 smoke test: verify backend selection + version reporting work.

#include <gtest/gtest.h>

#include "game_ggml/game_ggml.h"
#include "game_ggml/version.h"

#include "../src/backend.h"

namespace gg = game_ggml;

TEST(Version, NonEmpty) {
    EXPECT_STRNE(gg::version_string(), "");
    EXPECT_STRNE(gg::ggml_version_string(), "");
}

TEST(Backend, CpuAlwaysAvailable) {
    bool found_cpu = false;
    const int n = gg::available_backends_count();
    const char * const * names = gg::available_backends();
    ASSERT_GT(n, 0);
    for (int i = 0; i < n; ++i) {
        if (std::string(names[i]) == "cpu") found_cpu = true;
    }
    EXPECT_TRUE(found_cpu);
}

TEST(Backend, CpuInitSucceeds) {
    auto * b = gg::internal::init_backend(gg::Backend::CPU);
    ASSERT_NE(b, nullptr);
    EXPECT_STRNE(gg::internal::backend_name(b), "<null>");
    gg::internal::free_backend(b);
}

TEST(Backend, BestBackendInitSucceeds) {
    auto * b = gg::internal::init_best_backend();
    ASSERT_NE(b, nullptr);
    gg::internal::free_backend(b);
}
