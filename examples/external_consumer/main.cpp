// Minimal third-party integration example.
//
// Loads a GAME GGUF, reads a 44100 Hz mono WAV file (via any loader of your
// choice — here we use a very small hand-rolled WAV reader to keep this
// example free of extra deps) and prints the predicted notes.
//
// For production use you probably want to pull in a WAV library of your
// choice (dr_wav, libsndfile, miniaudio, ...).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <game_ggml/model.h>
#include <game_ggml/types.h>
#include <game_ggml/errors.h>

// ---- tiny PCM-16 WAV reader (just enough for the demo) -----------------
static std::vector<float> load_wav_mono_pcm16(const std::string & path, int expected_sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    char riff[4]; f.read(riff, 4); uint32_t _sz; f.read(reinterpret_cast<char*>(&_sz), 4);
    char wav[4];  f.read(wav, 4);
    if (std::memcmp(riff, "RIFF", 4) || std::memcmp(wav, "WAVE", 4))
        throw std::runtime_error("not a WAV: " + path);

    int16_t channels = 0, bps = 0;
    uint32_t sample_rate = 0, data_size = 0;
    std::vector<float> samples;

    while (f) {
        char id[4]; f.read(id, 4);
        uint32_t sz; f.read(reinterpret_cast<char*>(&sz), 4);
        if (!f) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            int16_t fmt; f.read(reinterpret_cast<char*>(&fmt), 2);
            f.read(reinterpret_cast<char*>(&channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            uint32_t _; int16_t __;
            f.read(reinterpret_cast<char*>(&_), 4);
            f.read(reinterpret_cast<char*>(&__), 2);
            f.read(reinterpret_cast<char*>(&bps), 2);
            if (sz > 16) f.ignore(sz - 16);
        } else if (!std::memcmp(id, "data", 4)) {
            data_size = sz;
            std::vector<int16_t> buf(sz / 2);
            f.read(reinterpret_cast<char*>(buf.data()), sz);
            samples.resize(buf.size() / channels);
            for (size_t i = 0; i < samples.size(); ++i) {
                int32_t acc = 0;
                for (int c = 0; c < channels; ++c) acc += buf[i * channels + c];
                samples[i] = static_cast<float>(acc) / (channels * 32768.0f);
            }
        } else {
            f.ignore(sz);
        }
    }
    if (bps != 16) throw std::runtime_error("expected PCM-16");
    if (static_cast<int>(sample_rate) != expected_sr)
        throw std::runtime_error("wrong sample rate: " + std::to_string(sample_rate));
    return samples;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <game_small.gguf> <input.wav>\n", argv[0]);
        return 1;
    }
    try {
        auto model = game_ggml::Model::load(argv[1]);
        const auto & cfg = model.config();

        auto samples = load_wav_mono_pcm16(argv[2], cfg.inference.audio_sample_rate);

        game_ggml::InferParams params;
        params.seed     = 42;
        params.language = 4;
        auto result = model.infer(samples.data(), samples.size(), params);

        std::fprintf(stdout, "# %zu note(s) across %d frames\n",
                     result.notes.size(), result.num_frames);
        for (const auto & n : result.notes) {
            std::fprintf(stdout, "  %7.3fs + %.3fs  pitch=%7.3f  %s\n",
                         n.offset_seconds, n.duration_seconds, n.pitch_midi,
                         n.voiced ? "voiced" : "rest");
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return 0;
}
