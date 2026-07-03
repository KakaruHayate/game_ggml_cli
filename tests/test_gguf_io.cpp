// Tests for gguf_io.cpp (GgufFile + parse_flat_int_object + load_config).
//
// Strategy:
//   * Writes a tiny mock GGUF at runtime via ggml's own `gguf_context` / set_*
//     helpers (no Python roundtrip required so tests remain hermetic).
//   * Exercises the C++ readers against that file, then loads the real
//     `assets/game_small.gguf` if present for a smoke test (skipped otherwise).

#include <gtest/gtest.h>

#include "gguf_io.h"

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace gi = game_ggml::internal;

namespace {

// Write a minimal GAME-style GGUF to a temp file and return its path.
std::string write_mock_gguf(const fs::path & dir) {
    auto * ctx = gguf_init_empty();

    gguf_set_val_str (ctx, "general.architecture",                "game-me");
    gguf_set_val_str (ctx, "general.name",                        "mock");
    gguf_set_val_str (ctx, "general.version",                     "1");
    gguf_set_val_str (ctx, "game.model.mode",                     "d3pm");
    gguf_set_val_i32 (ctx, "game.model.embedding_dim",            128);
    gguf_set_val_i32 (ctx, "game.model.in_dim",                   80);
    gguf_set_val_i32 (ctx, "game.model.estimator_out_dim",        257);
    gguf_set_val_i32 (ctx, "game.model.region_cycle_len",         3);
    gguf_set_val_bool(ctx, "game.model.use_languages",            true);
    gguf_set_val_i32 (ctx, "game.model.num_languages",            127);

    // Encoder
    gguf_set_val_str (ctx, "game.encoder.cls",                    "modules.backbones.EBF.EBFBackbone");
    gguf_set_val_i32 (ctx, "game.encoder.dim",                    128);
    gguf_set_val_i32 (ctx, "game.encoder.num_layers",             4);
    gguf_set_val_i32 (ctx, "game.encoder.num_heads",              4);
    gguf_set_val_i32 (ctx, "game.encoder.head_dim",               64);
    gguf_set_val_i32 (ctx, "game.encoder.c_kernel_size",          31);
    gguf_set_val_i32 (ctx, "game.encoder.m_kernel_size",          31);
    gguf_set_val_str (ctx, "game.encoder.ffn_type",               "glu");

    // Segmenter
    gguf_set_val_str (ctx, "game.segmenter.cls",                  "modules.backbones.EBF.EBFBackbone");
    gguf_set_val_i32 (ctx, "game.segmenter.dim",                  128);
    gguf_set_val_i32 (ctx, "game.segmenter.num_layers",           8);
    gguf_set_val_i32 (ctx, "game.segmenter.num_heads",            4);
    gguf_set_val_i32 (ctx, "game.segmenter.head_dim",             64);
    gguf_set_val_i32 (ctx, "game.segmenter.c_kernel_size",        31);
    gguf_set_val_i32 (ctx, "game.segmenter.m_kernel_size",        31);
    gguf_set_val_str (ctx, "game.segmenter.ffn_type",             "glu");
    gguf_set_val_i32 (ctx, "game.segmenter.latent_layer_idx",     6);
    gguf_set_val_i32 (ctx, "game.segmenter.latent_out_dim",       16);

    // Estimator
    gguf_set_val_str (ctx, "game.estimator.cls",                  "modules.backbones.ebf_with_joint_attention.JEBFBackbone");
    gguf_set_val_i32 (ctx, "game.estimator.dim",                  128);
    gguf_set_val_i32 (ctx, "game.estimator.num_layers",           4);
    gguf_set_val_i32 (ctx, "game.estimator.num_heads",            4);
    gguf_set_val_i32 (ctx, "game.estimator.head_dim",             64);
    gguf_set_val_i32 (ctx, "game.estimator.region_token_num",     1);
    gguf_set_val_str (ctx, "game.estimator.pool_merge_mode",      "mean");
    gguf_set_val_str (ctx, "game.estimator.attn_type",            "joint");
    gguf_set_val_str (ctx, "game.estimator.rope_mode",            "mixed");
    gguf_set_val_bool(ctx, "game.estimator.qk_norm",              true);
    gguf_set_val_bool(ctx, "game.estimator.use_region_bias",      false);
    gguf_set_val_i32 (ctx, "game.estimator.c_kernel_size_pool",   7);
    gguf_set_val_i32 (ctx, "game.estimator.m_kernel_size_pool",   5);
    gguf_set_val_i32 (ctx, "game.estimator.c_kernel_size_x",      31);
    gguf_set_val_i32 (ctx, "game.estimator.m_kernel_size_x",      31);
    gguf_set_val_str (ctx, "game.estimator.ffn_type",             "glu");

    // Inference
    gguf_set_val_i32 (ctx, "game.inference.audio_sample_rate",    44100);
    gguf_set_val_i32 (ctx, "game.inference.hop_size",             441);
    gguf_set_val_i32 (ctx, "game.inference.fft_size",             2048);
    gguf_set_val_i32 (ctx, "game.inference.win_size",             2048);
    gguf_set_val_f32 (ctx, "game.inference.timestep",             0.01f);
    gguf_set_val_str (ctx, "game.inference.spectrogram.type",     "mel");
    gguf_set_val_i32 (ctx, "game.inference.spectrogram.num_bins", 80);
    gguf_set_val_f32 (ctx, "game.inference.spectrogram.fmin",     0.0f);
    gguf_set_val_f32 (ctx, "game.inference.spectrogram.fmax",     8000.0f);
    gguf_set_val_f32 (ctx, "game.inference.midi_min",             0.0f);
    gguf_set_val_f32 (ctx, "game.inference.midi_max",             128.0f);
    gguf_set_val_i32 (ctx, "game.inference.midi_num_bins",        257);
    gguf_set_val_f32 (ctx, "game.inference.midi_std",             0.5f);
    gguf_set_val_str (ctx, "game.inference.lang_map",             "{\"en\": 1, \"ja\": 2, \"yue\": 3, \"zh\": 4}");

    // Add a single tiny tensor so the file is well-formed.  We need a backing
    // ggml_context to hold it before gguf_write_to_file can emit the tensor
    // table.
    const int64_t ne[1] = {4};
    struct ggml_init_params ip{};
    ip.mem_size = 1 << 16;
    ip.no_alloc = false;
    struct ggml_context * gctx = ggml_init(ip);
    struct ggml_tensor * t = ggml_new_tensor(gctx, GGML_TYPE_F32, 1, ne);
    ggml_set_name(t, "mock.tensor");
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(t->data, data, sizeof(data));
    gguf_add_tensor(ctx, t);

    const std::string out = (dir / "mock.gguf").string();
    gguf_write_to_file(ctx, out.c_str(), /*only_meta=*/false);

    ggml_free(gctx);
    gguf_free(ctx);
    return out;
}

}  // namespace

// =============================================================================
// Unit tests
// =============================================================================

TEST(ParseFlatIntObject, Basics) {
    auto m = gi::parse_flat_int_object("{\"a\":1,\"b\":2}");
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m["a"], 1);
    EXPECT_EQ(m["b"], 2);
}

TEST(ParseFlatIntObject, Whitespace) {
    auto m = gi::parse_flat_int_object("  { \"zh\" : 4 , \"en\": 1 }  ");
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m["zh"], 4);
    EXPECT_EQ(m["en"], 1);
}

TEST(ParseFlatIntObject, Empty) {
    auto m = gi::parse_flat_int_object("{}");
    EXPECT_TRUE(m.empty());
}

TEST(ParseFlatIntObject, Errors) {
    EXPECT_THROW(gi::parse_flat_int_object("{\"a\":}"),       game_ggml::InvalidArgument);
    EXPECT_THROW(gi::parse_flat_int_object("{\"a\":1,}"),     game_ggml::InvalidArgument);
    EXPECT_THROW(gi::parse_flat_int_object("{\"a\":1"),       game_ggml::InvalidArgument);
    EXPECT_THROW(gi::parse_flat_int_object("[]"),             game_ggml::InvalidArgument);
}

TEST(GgufFile, RoundtripMockFile) {
    auto tmp = fs::temp_directory_path() / "game_ggml_gguf_test";
    fs::create_directories(tmp);
    std::string path = write_mock_gguf(tmp);

    auto file = gi::GgufFile::open(path);
    EXPECT_TRUE(file.has("general.architecture"));
    EXPECT_EQ(file.get_string("general.architecture"), "game-me");
    EXPECT_EQ(file.get_int("game.model.embedding_dim"), 128);
    EXPECT_TRUE(file.get_bool("game.model.use_languages"));

    auto tensors = file.list_tensors();
    ASSERT_EQ(tensors.size(), 1u);
    EXPECT_EQ(tensors[0].name, "mock.tensor");

    auto cfg = gi::load_config(file);
    EXPECT_EQ(cfg.architecture, "game-me");
    EXPECT_EQ(cfg.mode, "d3pm");
    EXPECT_EQ(cfg.embedding_dim, 128);
    EXPECT_EQ(cfg.encoder.num_layers, 4);
    EXPECT_EQ(cfg.segmenter.num_layers, 8);
    EXPECT_EQ(cfg.segmenter.return_latent, true);
    EXPECT_EQ(cfg.segmenter.latent_layer_idx, 6);
    EXPECT_EQ(cfg.estimator.num_layers, 4);
    EXPECT_EQ(cfg.estimator.attn_type, "joint");
    EXPECT_EQ(cfg.estimator.rope_mode, "mixed");
    EXPECT_EQ(cfg.estimator.region_token_num, 1);
    EXPECT_EQ(cfg.inference.audio_sample_rate, 44100);
    EXPECT_EQ(cfg.inference.n_mels, 80);
    EXPECT_FLOAT_EQ(cfg.inference.timestep(), 441.0f / 44100.0f);
    ASSERT_EQ(cfg.inference.lang_map.size(), 4u);
    EXPECT_EQ(cfg.inference.lang_map["zh"], 4);
}

// Smoke test: if the user ran the converter and produced assets/game_small.gguf,
// make sure we can read it.  Otherwise the test is skipped.
TEST(GgufFile, RealSmallModelSmoke) {
    const char * env = std::getenv("GAME_GGML_TEST_ASSET");
    fs::path path = env ? env :
        fs::current_path() / ".." / "assets" / "game_small.gguf";
    if (!fs::exists(path)) {
        GTEST_SKIP() << "no game_small.gguf found at " << path;
    }
    auto file = gi::GgufFile::open(path.string());
    auto cfg  = gi::load_config(file);
    EXPECT_EQ(cfg.architecture, "game-me");
    EXPECT_EQ(cfg.embedding_dim, 128);
    EXPECT_GT((int)file.list_tensors().size(), 100);
}
