#include "wav_io.h"

#include "game_ggml/errors.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace game_ggml::cli {

WavFile load_wav_mono_f32(const std::string & path, int expected_sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw InvalidWav("failed to open WAV: " + path);
    }

    WavFile out;
    out.sample_rate = static_cast<int>(wav.sampleRate);
    out.channels    = static_cast<int>(wav.channels);

    if (expected_sample_rate > 0 && out.sample_rate != expected_sample_rate) {
        drwav_uninit(&wav);
        throw InvalidWav("WAV sample rate " + std::to_string(out.sample_rate) +
                         " != expected " + std::to_string(expected_sample_rate) +
                         " (" + path + ")");
    }

    const std::size_t total = static_cast<std::size_t>(wav.totalPCMFrameCount);
    std::vector<float> interleaved(total * wav.channels);
    const std::size_t got = drwav_read_pcm_frames_f32(&wav, total, interleaved.data());
    drwav_uninit(&wav);
    if (got != total) throw InvalidWav("short read while decoding WAV: " + path);

    if (wav.channels == 1) {
        out.samples = std::move(interleaved);
    } else {
        out.samples.resize(total);
        const float inv_c = 1.0f / static_cast<float>(wav.channels);
        for (std::size_t i = 0; i < total; ++i) {
            float acc = 0.0f;
            for (std::uint32_t c = 0; c < wav.channels; ++c) {
                acc += interleaved[i * wav.channels + c];
            }
            out.samples[i] = acc * inv_c;
        }
    }
    return out;
}

void write_wav_mono_i16(const std::string & path, const float * samples,
                        std::size_t n, int sample_rate) {
    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_PCM;
    fmt.channels      = 1;
    fmt.sampleRate    = static_cast<drwav_uint32>(sample_rate);
    fmt.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        throw InvalidWav("failed to open WAV for write: " + path);
    }

    std::vector<std::int16_t> pcm(n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = samples[i];
        if (v > 1.0f)  v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = static_cast<std::int16_t>(std::lround(v * 32767.0f));
    }
    drwav_write_pcm_frames(&wav, n, pcm.data());
    drwav_uninit(&wav);
}

}  // namespace game_ggml::cli
