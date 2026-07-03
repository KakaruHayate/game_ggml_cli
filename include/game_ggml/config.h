#pragma once

// Model configuration populated from a GGUF file's metadata section.
//
// This header is part of the public API so that third-party consumers can
// introspect a model's shape before/without running inference.

#include <map>
#include <string>
#include <vector>

namespace game_ggml {

// Configuration of an EBFBackbone / JEBFBackbone instance.  Fields that only
// apply to one of the two are optional; see the section-specific helpers on
// `GameModelConfig` for typed access.
struct BackboneConfig {
    // The fully-qualified PyTorch class, e.g. "modules.backbones.EBF.EBFBackbone".
    // Used purely for diagnostics.
    std::string cls;

    // ---- EBF shared ----
    int dim           = 0;
    int num_layers    = 0;
    int num_heads     = 0;
    int head_dim      = 0;
    int c_kernel_size = 0;
    int m_kernel_size = 0;

    std::string ffn_type = "glu";
    bool use_ls          = true;
    bool use_out_norm    = true;
    bool skip_first_ffn  = false;
    bool skip_out_ffn    = false;

    // ---- Segmenter-only (return_latent branch) ----
    bool return_latent     = false;
    int  latent_layer_idx  = 0;
    int  latent_out_dim    = 0;

    // ---- Estimator-only (JEBFBackbone) ----
    int  region_token_num   = 1;
    std::string pool_merge_mode = "mean";
    std::string attn_type       = "joint";
    std::string rope_mode       = "mixed";
    bool qk_norm                = true;
    bool use_region_bias        = false;
    int  c_kernel_size_pool     = 0;
    int  m_kernel_size_pool     = 0;
    int  c_kernel_size_x        = 0;
    int  m_kernel_size_x        = 0;
    bool use_rope               = true;
    bool use_pool_offset        = false;
    float theta                 = 10000.0f;
};

// Inference-time feature configuration (mel, bins, lang map).
struct InferenceConfig {
    int audio_sample_rate = 0;
    int hop_size          = 0;
    int fft_size          = 0;
    int win_size          = 0;
    int n_mels            = 0;
    float fmin            = 0.0f;
    float fmax            = 0.0f;
    std::string spectrogram_type = "mel";

    float midi_min     = 0.0f;
    float midi_max     = 0.0f;
    int   midi_num_bins = 0;
    float midi_std     = 0.0f;

    // If the model supports languages, this maps human-readable language
    // codes ("zh", "en", …) to the integer id expected by the segmenter's
    // language embedding.  Id 0 is reserved for "unknown / universal".
    std::map<std::string, int> lang_map;

    // Convenience derived field: frame duration in seconds.
    float timestep() const noexcept {
        return audio_sample_rate > 0
            ? static_cast<float>(hop_size) / static_cast<float>(audio_sample_rate)
            : 0.0f;
    }
};

// Top-level model configuration.
struct GameModelConfig {
    std::string architecture;       // must equal "game-me"
    std::string name;               // human name from converter
    std::string version;            // schema version ("1", …)

    // Top-level model hyperparameters.
    std::string mode;               // "d3pm" | "completion"
    int embedding_dim       = 0;
    int in_dim              = 0;    // number of mel bins
    int estimator_out_dim   = 0;    // pitch bin count
    int region_cycle_len    = 3;
    bool use_languages      = true;
    int  num_languages      = 0;    // size of learnable bank, pad id 0

    BackboneConfig   encoder;
    BackboneConfig   segmenter;
    BackboneConfig   estimator;

    InferenceConfig  inference;
};

}  // namespace game_ggml
