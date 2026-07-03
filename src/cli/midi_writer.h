#pragma once

// Minimal Standard MIDI File (SMF) type-0 writer.  Emits a single track
// holding a tempo meta-event followed by one NoteOn/NoteOff pair per voiced
// note in the input list.  Unvoiced notes are skipped.

#include "game_ggml/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace game_ggml::cli {

struct MidiWriteOptions {
    int   tempo_bpm    = 120;
    int   velocity     = 96;
    int   ticks_per_qn = 480;
};

// Write the note list to `path`.  Throws on I/O failure.
void write_midi_file(const std::string & path,
                     const std::vector<Note> & notes,
                     const MidiWriteOptions & opts = {});

// Encode-in-memory variant used by tests.
std::vector<std::uint8_t> encode_midi(const std::vector<Note> & notes,
                                      const MidiWriteOptions & opts = {});

}  // namespace game_ggml::cli
