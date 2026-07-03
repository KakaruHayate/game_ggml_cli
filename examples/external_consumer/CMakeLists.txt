# Minimal example showing how to consume game_ggml as a third-party library.
#
# Usage:
#   cmake -S examples/external_consumer -B /tmp/consumer-build
#   cmake --build /tmp/consumer-build -j
#   /tmp/consumer-build/my_app <path/to/game_small.gguf> <path/to/audio.wav>
cmake_minimum_required(VERSION 3.18)
project(game_ggml_consumer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point at the parent ggml_backend directory.  Two levels up from this file.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../..
                  ${CMAKE_BINARY_DIR}/game_ggml)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE game_ggml::game_ggml)
