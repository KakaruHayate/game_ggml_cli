#include "text_writer.h"

#include "game_ggml/errors.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace game_ggml::cli {

namespace {

const char * kPitchNames[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

std::string format_pitch(float pitch, bool voiced, const TextWriteOptions & opts) {
    if (!voiced) return "rest";
    if (opts.use_names) {
        const int rounded = static_cast<int>(std::lround(pitch));
        const int pc   = ((rounded % 12) + 12) % 12;
        const int oct  = rounded / 12 - 1;
        const float cents = (pitch - rounded) * 100.0f;
        char buf[48];
        if (opts.round_pitch || std::fabs(cents) < 0.5f) {
            std::snprintf(buf, sizeof(buf), "%s%d", kPitchNames[pc], oct);
        } else {
            std::snprintf(buf, sizeof(buf), "%s%d%+d", kPitchNames[pc], oct,
                          static_cast<int>(std::lround(cents)));
        }
        return buf;
    }
    char buf[32];
    if (opts.round_pitch) std::snprintf(buf, sizeof(buf), "%d",   static_cast<int>(std::lround(pitch)));
    else                  std::snprintf(buf, sizeof(buf), "%.3f", pitch);
    return buf;
}

}  // namespace

std::string format_notes_text(const std::vector<Note> & notes, TextFormat fmt,
                              const TextWriteOptions & opts) {
    std::ostringstream os;
    if (fmt == TextFormat::Csv) os << "offset,duration,pitch\n";
    for (const auto & n : notes) {
        const char sep = (fmt == TextFormat::Csv) ? ',' : '\t';
        os << n.offset_seconds << sep << n.duration_seconds << sep
           << format_pitch(n.pitch_midi, n.voiced, opts) << '\n';
    }
    return os.str();
}

void write_text_file(const std::string & path, const std::vector<Note> & notes,
                     TextFormat fmt, const TextWriteOptions & opts) {
    std::ofstream f(path);
    if (!f) throw Error("failed to open text output: " + path);
    f << format_notes_text(notes, fmt, opts);
    if (!f) throw Error("failed to write text output: " + path);
}

}  // namespace game_ggml::cli
