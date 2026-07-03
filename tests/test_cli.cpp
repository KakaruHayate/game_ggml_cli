// Slicer / MIDI / text writer unit tests (task 10).

#include <gtest/gtest.h>

#include "game_ggml/types.h"
#include "../src/cli/slicer.h"
#include "../src/cli/midi_writer.h"
#include "../src/cli/text_writer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace game_ggml;

TEST(Slicer, SingleChunkOnShortClip) {
    const int sr = 44100;
    const int n  = sr / 2;  // 0.5 s — below min_length
    std::vector<float> samples(n, 0.1f);
    cli::SlicerConfig cfg;
    cfg.sample_rate = sr;
    cfg.min_length_ms = 1000;
    auto chunks = cli::slice_waveform(samples.data(), samples.size(), cfg);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].offset_seconds, 0.0);
    EXPECT_EQ(chunks[0].waveform.size(), static_cast<std::size_t>(n));
}

TEST(Slicer, SplitsOnSilence) {
    // 1 s tone, 1 s silence, 1 s tone — expect 2 chunks.
    const int sr = 44100;
    const int seg = sr;
    std::vector<float> wav(seg * 3, 0.0f);
    for (int i = 0; i < seg; ++i)     wav[i]             = 0.5f * std::sin(2 * 3.14159f * 440.0f * i / sr);
    for (int i = 0; i < seg; ++i)     wav[2 * seg + i]   = 0.5f * std::sin(2 * 3.14159f * 440.0f * i / sr);
    cli::SlicerConfig cfg;
    cfg.sample_rate    = sr;
    cfg.min_length_ms  = 500;
    cfg.min_interval_ms = 300;
    cfg.max_sil_kept_ms = 200;
    auto chunks = cli::slice_waveform(wav.data(), wav.size(), cfg);
    EXPECT_GE(chunks.size(), 2u);
    // First chunk should start at or near 0.
    EXPECT_LE(chunks.front().offset_seconds, 0.1);
}

TEST(MidiWriter, EncodesNoteOnOff) {
    std::vector<Note> notes = {
        {0.0f, 0.5f, 60.0f, true},   // C4
        {0.5f, 0.5f, 64.0f, true},   // E4
        {1.0f, 0.5f, 67.0f, true},   // G4
    };
    auto bytes = cli::encode_midi(notes);
    // Sanity: starts with MThd, contains MTrk, ends with end-of-track meta.
    ASSERT_GE(bytes.size(), 22u);
    EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + 4), "MThd");
    bool has_mtrk = false;
    for (std::size_t i = 0; i + 4 < bytes.size(); ++i) {
        if (std::memcmp(bytes.data() + i, "MTrk", 4) == 0) { has_mtrk = true; break; }
    }
    EXPECT_TRUE(has_mtrk);
    // End-of-track meta FF 2F 00 appears somewhere near the end.
    bool has_eot = false;
    for (std::size_t i = 0; i + 2 < bytes.size(); ++i) {
        if (bytes[i] == 0xff && bytes[i + 1] == 0x2f && bytes[i + 2] == 0x00) {
            has_eot = true; break;
        }
    }
    EXPECT_TRUE(has_eot);
}

TEST(MidiWriter, SkipsUnvoiced) {
    std::vector<Note> notes = {
        {0.0f, 0.5f, 60.0f, true},
        {0.5f, 0.5f, 0.0f,  false},  // rest
        {1.0f, 0.5f, 64.0f, true},
    };
    auto a = cli::encode_midi(notes);
    auto b = cli::encode_midi({notes[0], notes[2]});
    EXPECT_EQ(a.size(), b.size()) << "rests must not produce MIDI events";
}

TEST(TextWriter, TxtAndCsv) {
    std::vector<Note> notes = {
        {0.0f, 0.5f, 60.0f, true},
        {0.5f, 0.5f, 0.0f,  false},
    };
    cli::TextWriteOptions opts;
    opts.use_names = false;
    opts.round_pitch = true;
    const std::string txt = cli::format_notes_text(notes, cli::TextFormat::Txt, opts);
    const std::string csv = cli::format_notes_text(notes, cli::TextFormat::Csv, opts);
    EXPECT_NE(txt.find("\t"), std::string::npos);
    EXPECT_NE(csv.find("offset,duration,pitch"), std::string::npos);
    EXPECT_NE(txt.find("rest"), std::string::npos);
    EXPECT_NE(csv.find("60"),   std::string::npos);
}

TEST(TextWriter, NoteNames) {
    std::vector<Note> notes = { {0.0f, 1.0f, 69.0f, true} };    // A4
    cli::TextWriteOptions opts;
    opts.use_names = true;
    const std::string s = cli::format_notes_text(notes, cli::TextFormat::Txt, opts);
    EXPECT_NE(s.find("A4"), std::string::npos);
}
