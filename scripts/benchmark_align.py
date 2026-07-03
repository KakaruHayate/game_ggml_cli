#!/usr/bin/env python3
"""Benchmark PyTorch vs ggml on the same waveform with bit-exact RNG replay.

Runs each backend in a fresh subprocess (so peak RSS is isolated from the
driver's memory footprint) multiple times, using the same align_rng.bin so
both sides produce identical MIDI.  Reports wall-time stats + peak RSS +
output note count.

Usage:
    python benchmark_align.py <wav> \
        -m GAME-pt-1.0-small/model.pt \
        -g ggml_backend/assets/game_small.gguf \
        --cli ggml_backend/build/bin/game_ggml_cli \
        --rng /tmp/align_out/align_rng.bin \
        -l zh -o /tmp/bench_out \
        --runs 3
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


def run_with_time(cmd: list[str], label: str) -> dict:
    """Run `cmd` under `/usr/bin/time -l`, parse wall time + peak RSS."""
    proc = subprocess.run(
        ["/usr/bin/time", "-l"] + cmd,
        capture_output=True, text=True,
    )
    stderr = proc.stderr
    out = {"label": label, "rc": proc.returncode}

    # Wall time:  "  9.11 real ..."
    m_real = re.search(r"(\d+(?:\.\d+)?)\s+real", stderr)
    if m_real:
        out["wall_s"] = float(m_real.group(1))

    # Peak memory footprint:  "     5291181712  peak memory footprint"
    m_rss = re.search(r"(\d+)\s+peak memory footprint", stderr)
    if m_rss:
        out["peak_rss_bytes"] = int(m_rss.group(1))

    # User / sys
    m_user = re.search(r"(\d+(?:\.\d+)?)\s+user", stderr)
    m_sys  = re.search(r"(\d+(?:\.\d+)?)\s+sys", stderr)
    if m_user: out["user_s"] = float(m_user.group(1))
    if m_sys:  out["sys_s"]  = float(m_sys.group(1))

    return out


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def run_torch_direct(wav, pt, lang, rng_bin, out_midi, t0, nsteps,
                      seg_thr, seg_rad, est_thr) -> list[str]:
    """Construct a one-shot `python -c "..."` command that runs the exact
    same sequential-chunk path as align_demo.py but with torch.rand_like
    REPLACED with a generator reading floats from rng_bin.  That keeps the
    comparison apples-to-apples."""
    script = f'''
import sys, struct, pathlib, math, numpy as np, torch
root = pathlib.Path({str(pt)!r}).resolve().parents[1]
sys.path.insert(0, str(root))

import librosa
from inference.api import load_inference_model
from inference.slicer2 import Slicer
import mido

rng_bytes = open({str(rng_bin)!r}, "rb").read()
rng_arr   = np.frombuffer(rng_bytes, dtype=np.float32)
cursor    = [0]

def rand_like_replay(x, *a, **kw):
    n = x.numel()
    c = cursor[0]
    if c + n > len(rng_arr):
        raise RuntimeError("rng bin exhausted")
    cursor[0] = c + n
    r = torch.from_numpy(rng_arr[c:c+n].copy()).reshape(x.shape)
    return r.to(x.device).float()

torch.rand_like = rand_like_replay

model, lang_map = load_inference_model(pathlib.Path({str(pt)!r}))
model.eval()
sr = model.inference_config.features.audio_sample_rate
language_id = lang_map.get({lang!r}, 0)

wav_np, _ = librosa.load({str(wav)!r}, sr=sr, mono=True)
slicer = Slicer(sr=sr, threshold=-40.0, min_length=5000,
                min_interval=300, max_sil_kept=5000)
chunks = slicer.slice(wav_np)

ts = torch.tensor([{t0} + i * (1 - {t0}) / {nsteps} for i in range({nsteps})])
py_notes = []
with torch.no_grad():
    for ch in chunks:
        ch_wav = torch.from_numpy(ch["waveform"]).float().unsqueeze(0)
        duration = ch_wav.shape[1] / sr
        durations, presence, scores = model(
            waveform=ch_wav,
            known_durations=torch.tensor([[duration]]),
            language=torch.tensor([language_id]),
            t=ts,
            boundary_threshold=torch.tensor({seg_thr}),
            boundary_radius=torch.tensor({seg_rad}),
            score_threshold=torch.tensor({est_thr}),
        )
        durs = durations[0].tolist(); pres = presence[0].tolist(); scs = scores[0].tolist()
        offset = float(ch["offset"])
        for d, p, s in zip(durs, pres, scs):
            if d <= 0: continue
            py_notes.append({{"offset": offset, "duration": d, "pitch": s, "voiced": bool(p)}})
            offset += d

# Write MIDI
mid = mido.MidiFile(type=0, ticks_per_beat=480)
track = mido.MidiTrack(); mid.tracks.append(track)
track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
tps = mid.ticks_per_beat * 120 / 60.0
events = []
for n in py_notes:
    if not n["voiced"]: continue
    p = max(0, min(127, int(round(n["pitch"]))))
    on_t  = int(round(n["offset"] * tps))
    off_t = int(round((n["offset"] + n["duration"]) * tps))
    if off_t <= on_t: off_t = on_t + 1
    events.append((on_t,  "note_on",  p, 96))
    events.append((off_t, "note_off", p, 64))
events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
last = 0
for t, kind, pitch, vel in events:
    track.append(mido.Message(kind, note=pitch, velocity=vel, time=t - last))
    last = t
track.append(mido.MetaMessage("end_of_track", time=0))
mid.save({str(out_midi)!r})
print(f"[pytorch] {{len(py_notes)}} notes written")
'''
    return [sys.executable, "-c", script]


def count_midi_notes(path: pathlib.Path) -> int:
    import mido
    m = mido.MidiFile(path)
    return sum(1 for t in m.tracks for msg in t
               if msg.type == "note_on" and msg.velocity > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav",         type=pathlib.Path)
    ap.add_argument("-m", "--pt",  type=pathlib.Path, required=True)
    ap.add_argument("-g", "--gguf", type=pathlib.Path, required=True)
    ap.add_argument("--cli",       type=pathlib.Path, required=True)
    ap.add_argument("--rng",       type=pathlib.Path, required=True,
                    help="align_rng.bin produced by align_demo.py")
    ap.add_argument("-l", "--language", type=str, default="zh")
    ap.add_argument("-o", "--out-dir",  type=pathlib.Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--t0",   type=float, default=0.0)
    ap.add_argument("--nsteps", type=int, default=1)
    ap.add_argument("--seg-threshold", type=float, default=0.2)
    ap.add_argument("--seg-radius",    type=int,   default=2)
    ap.add_argument("--est-threshold", type=float, default=0.2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lang_map = {"en": 1, "ja": 2, "yue": 3, "zh": 4}
    language_id = lang_map.get(args.language, 0)

    # ---- PyTorch (direct-chunk, no Lightning) ----
    py_out = args.out_dir / (args.wav.stem + ".pytorch.mid")
    py_cmd = run_torch_direct(
        wav=args.wav, pt=args.pt, lang=args.language, rng_bin=args.rng,
        out_midi=py_out, t0=args.t0, nsteps=args.nsteps,
        seg_thr=args.seg_threshold, seg_rad=args.seg_radius, est_thr=args.est_threshold,
    )

    # ---- ggml CLI ----
    gg_out = args.out_dir / (args.wav.stem + ".mid")
    gg_cmd = [
        str(args.cli), "extract", str(args.wav),
        "-m", str(args.gguf),
        "--output-formats", "mid",
        "--output-dir", str(args.out_dir),
        "--language", str(language_id),
        "--rng-replay", str(args.rng),
        "--seg-threshold", str(args.seg_threshold),
        "--seg-radius",    str(args.seg_radius),
        "--est-threshold", str(args.est_threshold),
        "--t0",            str(args.t0),
        "--nsteps",        str(args.nsteps),
    ]

    runs = {"pytorch": [], "ggml": []}
    for i in range(args.runs):
        print(f"\n=== run {i + 1}/{args.runs} ===", flush=True)
        print("  pytorch ...", flush=True)
        runs["pytorch"].append(run_with_time(py_cmd, f"pytorch-run{i}"))
        print(f"    wall={runs['pytorch'][-1].get('wall_s'):.2f}s "
              f"rss={human_bytes(runs['pytorch'][-1].get('peak_rss_bytes', 0))}")
        print("  ggml    ...", flush=True)
        runs["ggml"].append(run_with_time(gg_cmd, f"ggml-run{i}"))
        print(f"    wall={runs['ggml'][-1].get('wall_s'):.2f}s "
              f"rss={human_bytes(runs['ggml'][-1].get('peak_rss_bytes', 0))}")

    # ---- Summarise ----
    def agg(rows, key):
        vs = [r[key] for r in rows if key in r]
        return (min(vs), sum(vs) / len(vs), max(vs)) if vs else (None, None, None)

    py_wall = agg(runs["pytorch"], "wall_s")
    gg_wall = agg(runs["ggml"],    "wall_s")
    py_rss  = agg(runs["pytorch"], "peak_rss_bytes")
    gg_rss  = agg(runs["ggml"],    "peak_rss_bytes")

    # Verify outputs match.
    py_n = count_midi_notes(py_out)
    gg_n = count_midi_notes(gg_out)

    print("\n============================== Summary ==============================")
    print(f"Audio: {args.wav}")
    print(f"Runs per side: {args.runs}")
    print()
    print(f"{'':<20}{'min':>10} {'mean':>10} {'max':>10}")
    print(f"{'PyTorch wall (s)':<20}{py_wall[0]:>10.2f} {py_wall[1]:>10.2f} {py_wall[2]:>10.2f}")
    print(f"{'ggml wall (s)':<20}{gg_wall[0]:>10.2f} {gg_wall[1]:>10.2f} {gg_wall[2]:>10.2f}")
    print(f"{'Speedup (min/mean)':<20}{py_wall[0]/gg_wall[0]:>10.2f}×{py_wall[1]/gg_wall[1]:>9.2f}×")
    print()
    print(f"{'PyTorch peak RSS':<20}{human_bytes(py_rss[0]):>10} {human_bytes(py_rss[1]):>10} {human_bytes(py_rss[2]):>10}")
    print(f"{'ggml peak RSS':<20}{human_bytes(gg_rss[0]):>10} {human_bytes(gg_rss[1]):>10} {human_bytes(gg_rss[2]):>10}")
    print(f"{'Memory ratio':<20}{py_rss[1]/gg_rss[1]:>10.2f}× (PyTorch / ggml)")
    print()
    print(f"PyTorch notes: {py_n}")
    print(f"ggml    notes: {gg_n}")
    print(f"Notes match:   {'YES' if py_n == gg_n else 'NO'}")
    print("=====================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
