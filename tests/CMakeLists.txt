# Test suite.  Only built when GAME_GGML_BUILD_TESTS=ON.
include(FetchContent)

# GoogleTest v1.14 — BSD-3-Clause, no extra deps.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK  OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Shared test-support library (fixtures, reference-dump loader).
# ---------------------------------------------------------------------------
add_library(game_ggml_test_support STATIC
    support/reference_io.cpp
    support/ggml_test_env.cpp
)
target_include_directories(game_ggml_test_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
target_link_libraries(game_ggml_test_support PUBLIC
    game_ggml::game_ggml
    GTest::gtest
)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
set(GAME_GGML_TEST_SOURCES
    test_backend.cpp
    test_gguf_io.cpp
    test_ops_basic.cpp
    test_ops_ffn.cpp
    test_ops_rope.cpp
    test_encoder.cpp
    test_mel.cpp
    test_segmenter.cpp
    test_estimator.cpp
    test_pipeline_e2e.cpp
    test_cli.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cli/wav_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cli/slicer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cli/midi_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cli/text_writer.cpp
)

add_executable(game_ggml_tests ${GAME_GGML_TEST_SOURCES})
target_link_libraries(game_ggml_tests PRIVATE
    game_ggml::game_ggml
    game_ggml_test_support
    GTest::gtest_main
    dr_wav
)
target_compile_options(game_ggml_tests PRIVATE ${GAME_GGML_WARN_FLAGS})

include(GoogleTest)
gtest_discover_tests(game_ggml_tests
    DISCOVERY_TIMEOUT 30
    PROPERTIES LABELS "unit"
)
