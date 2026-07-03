#!/usr/bin/env python3
"""Compare PyTorch and ggml extract outputs on the same waveform, with
bit-exact alignment of the D3PM random sequence.

Flow:
  1. Re-run the slicer + PyTorch inference chunk-by-chunk in an explicit
     loop (no Lightning Trainer) so we control RNG consumption order.
  2. Monkey-patch `torch.rand_like` to capture every draw; flush the
     captured floats to `align_rng.bin`.
  3. Invoke `game_ggml_cli extract --rng-replay align_rng.bin`.
  4. Compare the two resulting MIDI files.

Usage:
    python align_demo.py <wav> \
        -m GAME-pt-1.0-small/model.pt \
        -g ggml_backend/assets/game_small.gguf \
        --cli ggml_backend/build/bin/game_ggml_cli \
        -l zh -o /tmp/align_out
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav",         type=pathlib.Path)
    ap.add_argument("-m", "--pt",  type=pathlib.Path, required=True, help="PyTorch .pt checkpoint")
    ap.add_argument("-g", "--gguf", type=pathlib.Path, required=True, help="Converted GGUF")
    ap.add_argument("--cli",       type=pathlib.Path, required=True, help="game_ggml_cli binary")
    ap.add_argument("-l", "--language", type=str, default="zh")
    ap.add_argument("-o", "--out-dir",  type=pathlib.Path, required=True)
    ap.add_argument("--t0",     type=float, default=0.0)
    ap.add_argument("--nsteps", type=int,   default=1)
    ap.add_argument("--seg-threshold", type=float, default=0.2)
    ap.add_argument("--seg-radius",    type=int,   default=2)
    ap.add_argument("--est-threshold", type=float, default=0.2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Make GAME's Python modules importable.
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from inference.api import load_inference_model
    from inference.slicer2 import Slicer
    import librosa

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"loading PyTorch model: {args.pt}")
    model, lang_map = load_inference_model(args.pt)
    model.eval()
    sr = model.inference_config.features.audio_sample_rate
    language_id = lang_map[args.language] if lang_map and args.language in lang_map else 0

    # ------------------------------------------------------------------
    # Load + slice the waveform using GAME's own Slicer (same as ggml CLI).
    # ------------------------------------------------------------------
    print(f"loading wav: {args.wav}")
    wav_np, _ = librosa.load(args.wav, sr=sr, mono=True)
    print(f"  duration: {len(wav_np) / sr:.2f}s")

    slicer = Slicer(sr=sr, threshold=-40.0, min_length=5000,
                    min_interval=300, max_sil_kept=5000)
    chunks = slicer.slice(wav_np)
    print(f"sliced into {len(chunks)} chunk(s)")

    # ------------------------------------------------------------------
    # Monkey-patch torch.rand_like so we keep every uniform draw in order.
    # ------------------------------------------------------------------
    captured: list[torch.Tensor] = []
    orig_rand_like = torch.rand_like

    def spy(x, *a, **kw):
        r = orig_rand_like(x, *a, **kw)
        captured.append(r.detach().cpu().float().flatten().clone())
        return r

    torch.rand_like = spy

    # ------------------------------------------------------------------
    # Run PyTorch inference chunk-by-chunk (no Lightning, no batching).
    # ------------------------------------------------------------------
    ts = torch.tensor([args.t0 + i * (1 - args.t0) / args.nsteps
                       for i in range(args.nsteps)])
    py_notes: list[dict] = []

    t_py_start = time.perf_counter()
    with torch.no_grad():
        for i, ch in enumerate(chunks):
            ch_wav = torch.from_numpy(ch["waveform"]).float().unsqueeze(0)   # [1, L]
            duration = ch_wav.shape[1] / sr
            known_durations = torch.tensor([[duration]])
            language = torch.tensor([language_id])

            durations, presence, scores = model(
                waveform=ch_wav,
                known_durations=known_durations,
                language=language,
                t=ts,
                boundary_threshold=torch.tensor(args.seg_threshold),
                boundary_radius=torch.tensor(args.seg_radius),
                score_threshold=torch.tensor(args.est_threshold),
            )
            durations = durations[0].tolist()
            presence  = presence[0].tolist()
            scores    = scores[0].tolist()

            offset = float(ch["offset"])
            for d, p, s in zip(durations, presence, scores):
                # Python side pads with -timestep; only keep real entries.
                if d <= 0:
                    continue
                py_notes.append({"offset": offset, "duration": d,
                                 "pitch": s, "voiced": bool(p)})
                offset += d
    t_py = time.perf_counter() - t_py_start

    torch.rand_like = orig_rand_like

    # Dump captured rands.
    rng_flat = torch.cat(captured) if captured else torch.zeros(0)
    rng_path = args.out_dir / "align_rng.bin"
    rng_path.write_bytes(rng_flat.numpy().astype(np.float32).tobytes())
    print(f"captured {rng_flat.numel()} uniform draws → {rng_path}")
    print(f"PyTorch: {t_py:.2f}s wall, {len(py_notes)} notes")

    # ------------------------------------------------------------------
    # Write PyTorch output as a MIDI (for side-by-side comparison).
    # ------------------------------------------------------------------
    import mido
    def notes_to_midi(notes, tempo_bpm=120, out=None):
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo",
                      tempo=mido.bpm2tempo(tempo_bpm), time=0))
        tps = mid.ticks_per_beat * tempo_bpm / 60.0
        events = []
        for n in notes:
            if not n["voiced"]: continue
            pitch = int(round(n["pitch"]))
            if pitch < 0 or pitch > 127: continue
            on_t  = int(round(n["offset"] * tps))
            off_t = int(round((n["offset"] + n["duration"]) * tps))
            if off_t <= on_t: off_t = on_t + 1
            events.append((on_t,  "note_on",  pitch, 96))
            events.append((off_t, "note_off", pitch, 64))
        events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
        last = 0
        for t, kind, pitch, vel in events:
            track.append(mido.Message(kind, note=pitch, velocity=vel, time=t - last))
            last = t
        track.append(mido.MetaMessage("end_of_track", time=0))
        if out: mid.save(out)
        return mid

    py_mid_path = args.out_dir / (args.wav.stem + ".pytorch.mid")
    notes_to_midi(py_notes, out=py_mid_path)

    # ------------------------------------------------------------------
    # Run ggml CLI with --rng-replay pointing to the captured file.
    # ------------------------------------------------------------------
    print(f"running ggml CLI with --rng-replay {rng_path}")
    t_gg_start = time.perf_counter()
    cmd = [
        str(args.cli), "extract", str(args.wav),
        "-m", str(args.gguf),
        "--output-formats", "mid",
        "--output-dir", str(args.out_dir),
        "--language", str(language_id),
        "--rng-replay", str(rng_path),
        "--seg-threshold", str(args.seg_threshold),
        "--seg-radius",    str(args.seg_radius),
        "--est-threshold", str(args.est_threshold),
        "--t0",            str(args.t0),
        "--nsteps",        str(args.nsteps),
    ]
    subprocess.run(cmd, check=True)
    t_gg = time.perf_counter() - t_gg_start
    gg_mid_path = args.out_dir / (args.wav.stem + ".mid")
    print(f"ggml: {t_gg:.2f}s wall")

    # ------------------------------------------------------------------
    # Compare.
    # ------------------------------------------------------------------
    def extract_notes(mid_path):
        m = mido.MidiFile(mid_path)
        out, cur = [], 0.0
        tps = None
        for track in m.tracks:
            t_abs = 0
            active = {}   # pitch -> on_tick
            for msg in track:
                t_abs += msg.time
                if msg.type == "set_tempo":
                    tps = m.ticks_per_beat * 1_000_000 / msg.tempo
                elif msg.type == "note_on" and msg.velocity > 0:
                    active[msg.note] = t_abs
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in active and tps:
                        on = active.pop(msg.note)
                        out.append({
                            "offset": on / tps,
                            "duration": (t_abs - on) / tps,
                            "pitch": msg.note,
                        })
        return sorted(out, key=lambda n: n["offset"])

    py_events = extract_notes(py_mid_path)
    gg_events = extract_notes(gg_mid_path)

    def time_match(a, b, tol=0.015):
        return abs(a["offset"] - b["offset"]) <= tol and abs(a["duration"] - b["duration"]) <= tol

    # Greedy 1-to-1 alignment.
    matched = 0
    used = [False] * len(gg_events)
    for pe in py_events:
        for i, ge in enumerate(gg_events):
            if used[i]: continue
            if time_match(pe, ge):
                matched += 1
                used[i] = True
                break

    print()
    print("========================== Alignment report ==========================")
    print(f"PyTorch notes: {len(py_events):>4d}   ({t_py:6.2f}s)")
    print(f"ggml notes:    {len(gg_events):>4d}   ({t_gg:6.2f}s)")
    print(f"Speedup:       {t_py / t_gg:5.2f}×")
    print(f"Matched 1-1:   {matched:>4d}   ({100 * matched / max(1, len(py_events)):.1f}% of PyTorch, "
          f"{100 * matched / max(1, len(gg_events)):.1f}% of ggml)")
    # Pitch disagreement on matched pairs.
    pitch_diffs = []
    for pe in py_events:
        for ge in gg_events:
            if time_match(pe, ge):
                pitch_diffs.append(ge["pitch"] - pe["pitch"])
                break
    if pitch_diffs:
        arr = np.array(pitch_diffs)
        print(f"Pitch Δ (semitone): mean={arr.mean():+.3f}  "
              f"std={arr.std():.3f}  |max|={np.abs(arr).max():.3f}")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
