#!/usr/bin/env python3
"""Dump PyTorch reference tensors for C++ numerical-parity tests.

Each invocation produces a set of `.bin` files under the given output
directory (default: ggml_backend/tests/data/).  Files are read by
`ggml_backend/tests/support/reference_io.cpp`.

Binary layout (little endian):
    offset 0:  magic      char[4]  = 'G','R','E','F'
    offset 4:  version    int32    = 1
    offset 8:  dtype      int32    (0=f32, 1=f16, 2=i32, 3=i64, 4=bool)
    offset 12: ndim       int32    in [0, 8]
    offset 16: dims       int64[8] — only the first `ndim` are meaningful
    offset 80: payload    raw bytes

Usage:
    python dump_reference.py --category basic --out tests/data/

Multiple categories can be dumped separately:
    --category basic        RMSNorm, Linear, LayerScale, Embedding (task 3)
    --category ffn          GLUFFN, CgMLP              (task 4)
    --category rope         SingleRoPE, RegionRoPE     (task 4)
    ... (later tasks extend this list)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import struct
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger("dump_reference")

# -----------------------------------------------------------------------------
# Binary writer
# -----------------------------------------------------------------------------
MAGIC = b"GREF"
VERSION = 1


DTYPE_CODE = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int32"):   2,
    np.dtype("int64"):   3,
    np.dtype("uint8"):   4,   # bool stored as 0/1
}


def write_ref(path: pathlib.Path, arr) -> None:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().contiguous().numpy()
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.dtype("bool"):
        arr = arr.astype("uint8")
    if arr.dtype not in DTYPE_CODE:
        raise ValueError(f"unsupported dtype: {arr.dtype}")
    ndim = arr.ndim
    if ndim > 8:
        raise ValueError(f"too many dimensions: {ndim}")

    dims = list(arr.shape) + [0] * (8 - ndim)
    header = struct.pack("<4siii", MAGIC, VERSION, DTYPE_CODE[arr.dtype], ndim) + struct.pack("<8q", *dims)
    assert len(header) == 80
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


# -----------------------------------------------------------------------------
# Category registry
# -----------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    fn: Callable[[pathlib.Path], None]


CATEGORIES: dict[str, list[Case]] = {}


def register(category: str):
    CATEGORIES.setdefault(category, [])
    def deco(fn: Callable[[pathlib.Path], None]) -> Callable[[pathlib.Path], None]:
        CATEGORIES[category].append(Case(name=fn.__name__.removeprefix("case_"), fn=fn))
        return fn
    return deco


def dump_category(category: str, out: pathlib.Path) -> int:
    cases = CATEGORIES.get(category, [])
    if not cases:
        log.warning("no cases registered for category '%s'", category)
        return 0
    out_dir = out / category
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        log.info("dumping %s/%s", category, c.name)
        c.fn(out_dir)
    return len(cases)


# -----------------------------------------------------------------------------
# PyTorch reference modules (must match the ggml implementations exactly)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x):
        return x * self.scale


# -----------------------------------------------------------------------------
# Basic category — task 3
# -----------------------------------------------------------------------------

@register("basic")
def case_rms_norm(out: pathlib.Path):
    torch.manual_seed(0)
    B, T, D = 2, 16, 128
    m = RMSNorm(D).eval()
    # Randomize weight to a non-trivial value.
    with torch.no_grad():
        m.weight.normal_(mean=1.0, std=0.1)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y = m(x)
    write_ref(out / "rms_norm_input.bin",  x)
    write_ref(out / "rms_norm_weight.bin", m.weight)
    write_ref(out / "rms_norm_output.bin", y)


@register("basic")
def case_linear(out: pathlib.Path):
    torch.manual_seed(1)
    B, T, I, O = 2, 16, 64, 128
    lin = nn.Linear(I, O, bias=True).eval()
    x = torch.randn(B, T, I)
    with torch.no_grad():
        y = lin(x)
    write_ref(out / "linear_input.bin",   x)
    write_ref(out / "linear_weight.bin",  lin.weight)
    write_ref(out / "linear_bias.bin",    lin.bias)
    write_ref(out / "linear_output.bin",  y)


@register("basic")
def case_linear_no_bias(out: pathlib.Path):
    torch.manual_seed(2)
    B, T, I, O = 1, 8, 128, 64
    lin = nn.Linear(I, O, bias=False).eval()
    x = torch.randn(B, T, I)
    with torch.no_grad():
        y = lin(x)
    write_ref(out / "linear_nb_input.bin",  x)
    write_ref(out / "linear_nb_weight.bin", lin.weight)
    write_ref(out / "linear_nb_output.bin", y)


@register("basic")
def case_layer_scale(out: pathlib.Path):
    torch.manual_seed(3)
    B, T, D = 2, 16, 128
    ls = LayerScale(D, init=0.5).eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y = ls(x)
    write_ref(out / "layer_scale_input.bin",  x)
    write_ref(out / "layer_scale_scale.bin",  ls.scale)
    write_ref(out / "layer_scale_output.bin", y)


@register("basic")
def case_embedding(out: pathlib.Path):
    torch.manual_seed(4)
    V, D = 128, 128
    emb = nn.Embedding(V, D, padding_idx=0).eval()
    idx = torch.randint(0, V, (2, 16), dtype=torch.int64)
    with torch.no_grad():
        y = emb(idx)
    write_ref(out / "embedding_weight.bin",  emb.weight)
    write_ref(out / "embedding_indices.bin", idx.to(torch.int32))
    write_ref(out / "embedding_output.bin",  y)


@register("basic")
def case_cyclic_region_embedding(out: pathlib.Path):
    """CyclicRegionEmbedding in eval mode is plain embedding(idx % cycle).

    We dump the raw indices and the pre-modulo weight so the test can use
    the `embedding()` builder with modded indices computed in C++.
    """
    torch.manual_seed(5)
    B, T = 2, 16
    cycle = 3
    dim = 128
    emb = nn.Embedding(cycle, dim).eval()
    # Region ids in [1, some_max] with occasional 0 for padding.
    raw_idx = torch.randint(0, 8, (B, T), dtype=torch.int64)
    with torch.no_grad():
        y = emb(raw_idx % cycle)
    write_ref(out / "cyclic_weight.bin",     emb.weight)
    write_ref(out / "cyclic_raw_indices.bin", raw_idx.to(torch.int32))
    write_ref(out / "cyclic_mod_indices.bin", (raw_idx % cycle).to(torch.int32))
    write_ref(out / "cyclic_output.bin",      y)


# -----------------------------------------------------------------------------
# FFN category — task 4  (GLU-FFN, CgMLP)
# -----------------------------------------------------------------------------

class GLUFFN(nn.Module):
    """Verbatim match of modules.backbones.layers.GLUFFN in inference mode."""
    def __init__(self, dim: int, latent_dim: int | None = None):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim * 4
        self.ln1 = nn.Linear(dim, latent_dim * 2)
        self.ln2 = nn.Linear(latent_dim, dim)

    def forward(self, x):
        x1, x2 = self.ln1(x).chunk(2, dim=-1)
        return self.ln2(F.gelu(x1) * x2)


class CgMLP(nn.Module):
    """Verbatim match of modules.backbones.layers.CgMLP."""
    def __init__(self, dim: int, kernel_size: int = 31, use_dw_act: bool = True):
        super().__init__()
        latent_dim = dim
        self.pw1 = nn.Conv1d(dim, latent_dim * 2, kernel_size=1)
        self.use_dw_act = use_dw_act
        self.norm = RMSNorm(latent_dim)
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(latent_dim, latent_dim, kernel_size,
                            stride=1, padding=padding, groups=latent_dim)
        self.pw2 = nn.Conv1d(latent_dim, dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = F.gelu(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.norm(x2.transpose(1, 2)).transpose(1, 2)
        x2 = self.dw(x2)
        if self.use_dw_act:
            x2 = F.gelu(x2)
        x = x1 * x2
        x = self.pw2(x)
        return x.transpose(1, 2)


@register("ffn")
def case_glu_ffn(out: pathlib.Path):
    torch.manual_seed(10)
    B, T, D = 2, 16, 128
    m = GLUFFN(D).eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y = m(x)
    write_ref(out / "glu_input.bin",     x)
    write_ref(out / "glu_w_ln1.bin",     m.ln1.weight)
    write_ref(out / "glu_b_ln1.bin",     m.ln1.bias)
    write_ref(out / "glu_w_ln2.bin",     m.ln2.weight)
    write_ref(out / "glu_b_ln2.bin",     m.ln2.bias)
    write_ref(out / "glu_output.bin",    y)


@register("ffn")
def case_cgmlp(out: pathlib.Path):
    # Note: ggml_conv_1d_dw (v0.11.0) asserts batch dim == 1 inside im2col.
    # Since GAME's inference pipeline always runs B=1, our port matches that
    # contract.  The dump uses B=1 so the unit test stays meaningful.
    torch.manual_seed(11)
    B, T, D, K = 1, 16, 128, 31
    m = CgMLP(D, kernel_size=K).eval()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y = m(x)
    write_ref(out / "cg_input.bin",     x)
    write_ref(out / "cg_w_pw1.bin",     m.pw1.weight)
    write_ref(out / "cg_b_pw1.bin",     m.pw1.bias)
    write_ref(out / "cg_w_norm.bin",    m.norm.weight)
    write_ref(out / "cg_w_dw.bin",      m.dw.weight)
    write_ref(out / "cg_b_dw.bin",      m.dw.bias)
    write_ref(out / "cg_w_pw2.bin",     m.pw2.weight)
    write_ref(out / "cg_b_pw2.bin",     m.pw2.bias)
    write_ref(out / "cg_output.bin",    y)


# -----------------------------------------------------------------------------
# RoPE category — task 4  (SingleRoPE + RegionRoPE local/global/mixed)
# -----------------------------------------------------------------------------

def _compute_inv_freq(dim: int, theta: float = 10000.0):
    return 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))


def _apply_rotary_by_positions(x, positions, inv_freq):
    """Matches modules.backbones.rope.apply_rotary_by_positions."""
    pos = positions.unsqueeze(-1).float()                     # [..., 1]
    inv = inv_freq.view(*((1,) * (pos.ndim - 1)), -1)         # [..., D/2]
    freqs = pos * inv                                          # [..., T, D/2]
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    # Broadcast across the attention layout [B, H, T, D].
    # x.ndim - positions.ndim - 1 extra singleton dims between batch dims and T.
    n_extra = x.ndim - positions.ndim - 1
    shape = freqs_cos.shape[:1] + (1,) * n_extra + freqs_cos.shape[1:]
    freqs_cos = freqs_cos.view(shape)
    freqs_sin = freqs_sin.view(shape)

    x_ = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    x_r, x_i = x_[..., 0], x_[..., 1]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    return torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2).to(x.dtype)


@register("rope")
def case_single_rope(out: pathlib.Path):
    torch.manual_seed(20)
    B, H, T, D = 1, 4, 32, 64
    theta = 10000.0
    x = torch.randn(B, H, T, D)
    positions = torch.arange(T).unsqueeze(0).float()          # [1, T]
    inv_freq = _compute_inv_freq(D, theta)
    with torch.no_grad():
        y = _apply_rotary_by_positions(x, positions, inv_freq)
    # Dump in [B, T, H, D] layout so that the flat bytes correspond to the
    # ggml shape ne=(D, H, T, B) — the order ggml_rope_ext expects.
    write_ref(out / "single_x.bin",         x.transpose(1, 2).contiguous())
    write_ref(out / "single_positions.bin", positions.to(torch.int32).squeeze(0))
    write_ref(out / "single_output.bin",    y.transpose(1, 2).contiguous())


@register("rope")
def case_region_rope_local(out: pathlib.Path):
    torch.manual_seed(21)
    B, H, T, D = 1, 4, 32, 64
    theta = 10000.0
    x = torch.randn(B, H, T, D)
    local_pos = torch.arange(T).unsqueeze(0) % 8
    inv_freq = _compute_inv_freq(D, theta)
    with torch.no_grad():
        y = _apply_rotary_by_positions(x, local_pos.float(), inv_freq)
    write_ref(out / "local_x.bin",          x.transpose(1, 2).contiguous())
    write_ref(out / "local_positions.bin",  local_pos.to(torch.int32).squeeze(0))
    write_ref(out / "local_output.bin",     y.transpose(1, 2).contiguous())


@register("rope")
def case_region_rope_mixed(out: pathlib.Path):
    torch.manual_seed(22)
    B, H, T, D = 1, 4, 32, 64
    theta = 10000.0
    x = torch.randn(B, H, T, D)
    global_pos = torch.arange(T).unsqueeze(0).float()
    region_idx = (torch.arange(T).unsqueeze(0) // 4 + 1).float()
    half = D // 2
    inv_freq_g = _compute_inv_freq(half, theta)
    inv_freq_r = _compute_inv_freq(half, theta)
    with torch.no_grad():
        x_lo = _apply_rotary_by_positions(x[..., :half], global_pos, inv_freq_g)
        x_hi = _apply_rotary_by_positions(x[..., half:], region_idx, inv_freq_r)
        y = torch.cat([x_lo, x_hi], dim=-1)
    write_ref(out / "mixed_x.bin",             x.transpose(1, 2).contiguous())
    write_ref(out / "mixed_global_pos.bin",    global_pos.to(torch.int32).squeeze(0))
    write_ref(out / "mixed_region_idx.bin",    region_idx.to(torch.int32).squeeze(0))
    write_ref(out / "mixed_output.bin",        y.transpose(1, 2).contiguous())


# =============================================================================
# Encoder category — task 5  (full EBFBackbone encoder E2E)
# =============================================================================

@register("encoder")
def case_encoder_full(out: pathlib.Path):
    """Dump a full encoder pass using the actual GAME checkpoint.

    This case relies on importing the GAME project itself (must be run from
    the repo root so `modules.*` and `lib.*` are importable).  It skips
    gracefully if the checkpoint or config cannot be loaded.
    """
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ckpt_dir = root / "GAME-pt-1.0-small"
    if not (ckpt_dir / "model.pt").exists():
        log.warning("encoder dump skipped: %s missing", ckpt_dir)
        return
    try:
        from lib.config.io import load_raw_config  # type: ignore
        from lib.config.schema import ModelConfig  # type: ignore
        from modules.midi_extraction import SegmentationEstimationModel  # type: ignore
    except ImportError as e:
        log.warning("encoder dump skipped: cannot import GAME modules (%s)", e)
        return

    cfg_raw = load_raw_config(ckpt_dir / "config.yaml", inherit=False, overrides=None)
    model_cfg = ModelConfig.model_validate(cfg_raw["model"], scope=0)
    model_cfg.check(scope_mask=0)

    model = SegmentationEstimationModel(model_cfg).eval()
    ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)

    torch.manual_seed(30)
    B, T = 1, 100
    mel = torch.randn(B, T, model_cfg.in_dim)
    enc = model.encoder
    with torch.no_grad():
        x_projected = model.spectrogram_projection(mel)       # [B, T, D_embed]
        # Walk the encoder manually so we can dump intermediates.
        x = enc.input_proj(x_projected)
        write_ref(out / "enc_after_input_proj.bin", x)
        for i, layer in enumerate(enc.layers):
            x = layer(x, mask=None)
            write_ref(out / f"enc_after_layer_{i}.bin", x)
        if enc.use_out_norm:
            x = enc.output_norm(x)
            write_ref(out / "enc_after_output_norm.bin", x)
        x_full = enc.output_proj(x)
        write_ref(out / "enc_after_output_proj.bin", x_full)
        x_seg, x_est = torch.split(x_full,
            [model_cfg.embedding_dim, model_cfg.embedding_dim], dim=-1)
    write_ref(out / "enc_x_projected.bin", x_projected)
    write_ref(out / "enc_x_seg.bin",       x_seg)
    write_ref(out / "enc_x_est.bin",       x_est)


# =============================================================================
# Mel category — task 6 (StretchableMelSpectrogram parity)
# =============================================================================

@register("mel")
def case_mel_noise(out: pathlib.Path):
    """Reproduce the training front-end's mel output on seeded noise.

    Imports lib.feature.mel.StretchableMelSpectrogram from GAME itself so
    there is no drift between the reference and the C++ port.
    """
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from lib.feature.mel import StretchableMelSpectrogram  # type: ignore
    except ImportError as e:
        log.warning("mel dump skipped: cannot import GAME mel (%s)", e)
        return

    torch.manual_seed(40)
    sr  = 44100
    wav = torch.randn(1, sr)                                # 1 second
    mel = StretchableMelSpectrogram(
        sample_rate=sr, n_mels=80, n_fft=2048, win_length=2048,
        hop_length=441, fmin=0.0, fmax=8000.0, clip_val=1e-5,
    ).eval()
    with torch.no_grad():
        spec = mel(wav).transpose(-2, -1)                   # [1, T, 80]
    write_ref(out / "mel_wav.bin",    wav.squeeze(0))
    write_ref(out / "mel_output.bin", spec.squeeze(0))


# =============================================================================
# Segmenter category — task 7  (single-step + D3PM loop with RNG replay)
# =============================================================================

@register("segmenter")
def case_segmenter_single_step(out: pathlib.Path):
    """One-shot forward_segmentation: deterministic, no RNG."""
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from lib.config.io import load_raw_config  # type: ignore
        from lib.config.schema import ModelConfig  # type: ignore
        from modules.midi_extraction import SegmentationEstimationModel  # type: ignore
    except ImportError as e:
        log.warning("segmenter dump skipped: %s", e)
        return

    ckpt_dir = root / "GAME-pt-1.0-small"
    if not (ckpt_dir / "model.pt").exists():
        log.warning("segmenter dump skipped: checkpoint missing")
        return

    cfg_raw = load_raw_config(ckpt_dir / "config.yaml", inherit=False, overrides=None)
    model_cfg = ModelConfig.model_validate(cfg_raw["model"], scope=0)
    model_cfg.check(scope_mask=0)

    model = SegmentationEstimationModel(model_cfg).eval()
    ckpt  = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()},
                          strict=True)

    torch.manual_seed(50)
    B, T = 1, 100
    D = model_cfg.embedding_dim
    x_seg = torch.randn(B, T, D)
    # Construct noise (region ids 1..N cycling) and a valid mask.
    region_len = 10
    noise = ((torch.arange(T) // region_len) + 1).unsqueeze(0).long()  # [B, T]
    mask  = torch.ones(B, T, dtype=torch.bool)
    t_val = torch.tensor([0.5])
    lang  = torch.tensor([4])  # zh

    with torch.no_grad():
        logits, latent = model.forward_segmentation(
            x_seg, noise=noise, t=t_val, language=lang, mask=mask)

    write_ref(out / "seg_x_seg.bin",    x_seg.squeeze(0))
    write_ref(out / "seg_noise.bin",    noise.squeeze(0).to(torch.int32))
    write_ref(out / "seg_t.bin",        t_val)
    write_ref(out / "seg_lang.bin",     lang.to(torch.int32))
    write_ref(out / "seg_logits.bin",   logits.squeeze(0))
    write_ref(out / "seg_latent.bin",   latent.squeeze(0))


@register("segmenter")
def case_d3pm_loop(out: pathlib.Path):
    """Full 8-step D3PM loop with random-number capture for C++ replay.

    We monkey-patch torch.rand_like to append the drawn tensor so the C++
    side can feed the exact same uniform samples into its InjectedRng.
    """
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from lib.config.io import load_raw_config  # type: ignore
        from lib.config.schema import ModelConfig, InferenceConfig  # type: ignore
        from inference.me_infer import SegmentationEstimationInferenceModel  # type: ignore
    except ImportError as e:
        log.warning("d3pm dump skipped: %s", e)
        return

    ckpt_dir = root / "GAME-pt-1.0-small"
    if not (ckpt_dir / "model.pt").exists():
        log.warning("d3pm dump skipped: checkpoint missing")
        return

    cfg_raw = load_raw_config(ckpt_dir / "config.yaml", inherit=False, overrides=None)
    model_cfg = ModelConfig.model_validate(cfg_raw["model"], scope=0)
    infer_cfg = InferenceConfig.model_validate(cfg_raw["inference"], scope=0)
    for c in (model_cfg, infer_cfg):
        c.check(scope_mask=0)

    model = SegmentationEstimationInferenceModel(model_cfg, infer_cfg).eval()
    ckpt  = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    torch.manual_seed(60)
    B, T = 1, 100
    x_seg = torch.randn(B, T, model_cfg.embedding_dim)
    mask  = torch.ones(B, T, dtype=torch.bool)
    known = torch.zeros(B, T, dtype=torch.bool)
    lang  = torch.tensor([4])
    num_steps = 8
    ts = torch.tensor([i / num_steps for i in range(num_steps)])

    # Monkey-patch torch.rand_like to capture generated tensors.
    captured = []
    orig_rand_like = torch.rand_like
    def spy(x, *a, **kw):
        r = orig_rand_like(x, *a, **kw)
        captured.append(r.clone().flatten())
        return r
    torch.rand_like = spy
    try:
        with torch.no_grad():
            boundaries, _regions, _max_n = model.forward_segmenter_main(
                x_seg, known_boundaries=known, mask=mask, language=lang, t=ts,
                threshold=torch.tensor(0.2),
                radius=torch.tensor(2),
            )
    finally:
        torch.rand_like = orig_rand_like

    rnd_flat = torch.cat(captured) if captured else torch.zeros(0)
    write_ref(out / "d3pm_x_seg.bin",       x_seg.squeeze(0))
    write_ref(out / "d3pm_mask.bin",        mask.squeeze(0).to(torch.uint8))
    write_ref(out / "d3pm_known.bin",       known.squeeze(0).to(torch.uint8))
    write_ref(out / "d3pm_lang.bin",        lang.to(torch.int32))
    write_ref(out / "d3pm_ts.bin",          ts)
    write_ref(out / "d3pm_rng.bin",         rnd_flat)
    write_ref(out / "d3pm_boundaries.bin",  boundaries.squeeze(0).to(torch.uint8))


# =============================================================================
# Estimator category — task 8  (JEBFBackbone + joint attention)
# =============================================================================

@register("estimator")
def case_estimator_full(out: pathlib.Path):
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from lib.config.io import load_raw_config  # type: ignore
        from lib.config.schema import ModelConfig  # type: ignore
        from modules.midi_extraction import SegmentationEstimationModel  # type: ignore
    except ImportError as e:
        log.warning("estimator dump skipped: %s", e)
        return

    ckpt_dir = root / "GAME-pt-1.0-small"
    if not (ckpt_dir / "model.pt").exists():
        log.warning("estimator dump skipped: checkpoint missing")
        return

    cfg_raw = load_raw_config(ckpt_dir / "config.yaml", inherit=False, overrides=None)
    model_cfg = ModelConfig.model_validate(cfg_raw["model"], scope=0)
    model_cfg.check(scope_mask=0)
    model = SegmentationEstimationModel(model_cfg).eval()
    ckpt  = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()},
                          strict=True)

    torch.manual_seed(70)
    B, T, N = 1, 100, 10
    D = model_cfg.embedding_dim
    x_est = torch.randn(B, T, D)
    # Construct synthetic regions: 10 evenly-spaced regions of length 10.
    region_len = T // N
    regions = torch.zeros(B, T, dtype=torch.long)
    for r in range(N):
        regions[:, r*region_len:(r+1)*region_len] = r + 1
    t_mask = torch.ones(B, T, dtype=torch.bool)
    n_mask = torch.ones(B, N, dtype=torch.bool)

    with torch.no_grad():
        logits = model.forward_estimation(x_est, regions, t_mask=t_mask, n_mask=n_mask)

    write_ref(out / "est_x_est.bin",   x_est.squeeze(0))
    write_ref(out / "est_regions.bin", regions.squeeze(0).to(torch.int32))
    write_ref(out / "est_logits.bin",  logits.squeeze(0))


# =============================================================================
# E2E pipeline — task 9 (Model::infer parity)
# =============================================================================

@register("pipeline")
def case_pipeline_e2e(out: pathlib.Path):
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from lib.config.io import load_raw_config                                 # type: ignore
        from lib.config.schema import ModelConfig, InferenceConfig                # type: ignore
        from inference.me_infer import SegmentationEstimationInferenceModel       # type: ignore
    except ImportError as e:
        log.warning("pipeline dump skipped: %s", e)
        return

    ckpt_dir = root / "GAME-pt-1.0-small"
    if not (ckpt_dir / "model.pt").exists():
        log.warning("pipeline dump skipped: checkpoint missing")
        return

    cfg_raw = load_raw_config(ckpt_dir / "config.yaml", inherit=False, overrides=None)
    model_cfg = ModelConfig.model_validate(cfg_raw["model"], scope=0)
    infer_cfg = InferenceConfig.model_validate(cfg_raw["inference"], scope=0)
    for c in (model_cfg, infer_cfg):
        c.check(scope_mask=0)

    model = SegmentationEstimationInferenceModel(model_cfg, infer_cfg).eval()
    ckpt  = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # ~2-second synthetic waveform.
    torch.manual_seed(80)
    sr = infer_cfg.features.audio_sample_rate
    n  = sr * 2
    wav = 0.1 * torch.randn(1, n)

    # Capture all torch.rand_like calls flattened.
    captured = []
    orig_rand_like = torch.rand_like
    def spy(x, *a, **kw):
        r = orig_rand_like(x, *a, **kw)
        captured.append(r.clone().flatten())
        return r
    torch.rand_like = spy

    num_steps = 8
    t0 = 0.0
    ts = torch.tensor([t0 + i * (1 - t0) / num_steps for i in range(num_steps)])
    try:
        with torch.no_grad():
            durations, presence, scores = model(
                waveform=wav,
                known_durations=torch.tensor([[n / sr]]),   # single full-clip word
                boundary_threshold=torch.tensor(0.2),
                boundary_radius=torch.tensor(2),
                score_threshold=torch.tensor(0.2),
                language=torch.tensor([4]),
                t=ts,
            )
    finally:
        torch.rand_like = orig_rand_like

    rng_flat = torch.cat(captured) if captured else torch.zeros(0)
    write_ref(out / "pipe_wav.bin",        wav.squeeze(0))
    write_ref(out / "pipe_ts.bin",         ts)
    write_ref(out / "pipe_rng.bin",        rng_flat)
    write_ref(out / "pipe_durations.bin",  durations.squeeze(0))
    write_ref(out / "pipe_presence.bin",   presence.squeeze(0).to(torch.uint8))
    write_ref(out / "pipe_scores.bin",     scores.squeeze(0))


# =============================================================================
# Entry point
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", type=str, required=True,
                        choices=sorted(CATEGORIES.keys()) + ["all"])
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Output directory (cases are grouped in subdirs by category).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    total = 0
    if args.category == "all":
        for cat in CATEGORIES:
            total += dump_category(cat, args.out)
    else:
        total = dump_category(args.category, args.out)

    log.info("wrote %d cases to %s", total, args.out)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
