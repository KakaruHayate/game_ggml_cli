#include "midi_writer.h"

#include "game_ggml/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace game_ggml::cli {

namespace {

void write_u16_be(std::vector<std::uint8_t> & b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v >> 8));
    b.push_back(static_cast<std::uint8_t>(v & 0xff));
}
void write_u32_be(std::vector<std::uint8_t> & b, std::uint32_t v) {
    b.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
    b.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
    b.push_back(static_cast<std::uint8_t>((v >>  8) & 0xff));
    b.push_back(static_cast<std::uint8_t>( v        & 0xff));
}
void write_var_len(std::vector<std::uint8_t> & b, std::uint32_t v) {
    std::uint8_t buf[4];
    int n = 0;
    buf[n++] = static_cast<std::uint8_t>(v & 0x7f);
    v >>= 7;
    while (v) {
        buf[n++] = static_cast<std::uint8_t>((v & 0x7f) | 0x80);
        v >>= 7;
    }
    for (int i = n - 1; i >= 0; --i) b.push_back(buf[i]);
}

}  // namespace

std::vector<std::uint8_t> encode_midi(const std::vector<Note> & notes,
                                      const MidiWriteOptions & opts) {
    const int   ticks_per_qn = opts.ticks_per_qn;
    // At given BPM, 1 second = BPM/60 quarter notes = BPM/60 * ticks_per_qn ticks.
    const double ticks_per_sec =
        static_cast<double>(ticks_per_qn) * opts.tempo_bpm / 60.0;

    // Build (tick, event_bytes) list.
    struct Event { std::uint64_t tick; std::vector<std::uint8_t> data; };
    std::vector<Event> events;

    // Tempo meta: FF 51 03 tt tt tt  (microseconds per quarter note)
    {
        const std::uint32_t mpqn = static_cast<std::uint32_t>(60000000 / opts.tempo_bpm);
        Event e;
        e.tick = 0;
        e.data = {0xff, 0x51, 0x03,
                   static_cast<std::uint8_t>((mpqn >> 16) & 0xff),
                   static_cast<std::uint8_t>((mpqn >>  8) & 0xff),
                   static_cast<std::uint8_t>( mpqn        & 0xff)};
        events.push_back(std::move(e));
    }

    for (const auto & n : notes) {
        if (!n.voiced) continue;
        const int pitch = std::clamp<int>(
            static_cast<int>(std::lround(n.pitch_midi)), 0, 127);
        const auto start_tick = static_cast<std::uint64_t>(
            std::llround(n.offset_seconds * ticks_per_sec));
        const auto end_tick   = static_cast<std::uint64_t>(
            std::llround((n.offset_seconds + n.duration_seconds) * ticks_per_sec));
        const std::uint8_t vel = static_cast<std::uint8_t>(std::clamp(opts.velocity, 1, 127));

        Event on;
        on.tick = start_tick;
        on.data = {0x90, static_cast<std::uint8_t>(pitch), vel};
        events.push_back(std::move(on));

        Event off;
        off.tick = (end_tick > start_tick) ? end_tick : start_tick + 1;
        off.data = {0x80, static_cast<std::uint8_t>(pitch), 0x40};
        events.push_back(std::move(off));
    }

    // End of track meta: FF 2F 00.
    {
        std::uint64_t last = 0;
        for (const auto & e : events) last = std::max(last, e.tick);
        Event e;
        e.tick = last;
        e.data = {0xff, 0x2f, 0x00};
        events.push_back(std::move(e));
    }

    std::stable_sort(events.begin(), events.end(),
        [](const Event & a, const Event & b) { return a.tick < b.tick; });

    // Serialise track events with delta-time prefixes.
    std::vector<std::uint8_t> track_body;
    std::uint64_t prev_tick = 0;
    for (const auto & e : events) {
        const std::uint64_t dt = e.tick - prev_tick;
        prev_tick = e.tick;
        write_var_len(track_body, static_cast<std::uint32_t>(dt));
        track_body.insert(track_body.end(), e.data.begin(), e.data.end());
    }

    // MThd + MTrk.
    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'M','T','h','d'});
    write_u32_be(out, 6);
    write_u16_be(out, 0);                                      // format 0
    write_u16_be(out, 1);                                      // 1 track
    write_u16_be(out, static_cast<std::uint16_t>(ticks_per_qn));

    out.insert(out.end(), {'M','T','r','k'});
    write_u32_be(out, static_cast<std::uint32_t>(track_body.size()));
    out.insert(out.end(), track_body.begin(), track_body.end());
    return out;
}

void write_midi_file(const std::string & path,
                     const std::vector<Note> & notes,
                     const MidiWriteOptions & opts) {
    const auto bytes = encode_midi(notes, opts);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw Error("failed to open MIDI output: " + path);
    f.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!f) throw Error("failed to write MIDI: " + path);
}

}  // namespace game_ggml::cli
