#pragma once

#include "game_ggml/types.h"

#include <string>
#include <vector>

namespace game_ggml::cli {

enum class TextFormat { Txt, Csv };

struct TextWriteOptions {
    bool round_pitch = false;      // round to nearest integer
    bool use_names   = false;      // emit "C4+10" instead of numeric pitch
};

// Format notes as a string — handy for tests.
std::string format_notes_text(const std::vector<Note> & notes,
                              TextFormat fmt,
                              const TextWriteOptions & opts = {});

// Write to file; throws on I/O failure.
void write_text_file(const std::string & path,
                     const std::vector<Note> & notes,
                     TextFormat fmt,
                     const TextWriteOptions & opts = {});

}  // namespace game_ggml::cli
