"""
experiments/exp6_codec.py

Experiment 6 — Lossy codec compression.

Tests how MP3, AAC, and Opus compression at various bitrates affects
speaker verification performance.  Each trial compresses the original
audio file with ffmpeg, decodes back to 16 kHz PCM, then runs the
normal embedding + distance pipeline.

Codecs and bitrates:
  MP3  : 32, 64, 128 kbps   (libmp3lame)
  AAC  : 32, 64, 128 kbps   (built-in aac encoder)
  Opus :  8, 16,  32 kbps   (libopus — Opus is very efficient at low bitrates)

Requires: ffmpeg with libmp3lame and libopus support on PATH.

Outputs (results/exp6/):
  scores.csv        distance, codec, bitrate_kbps, genuine per trial
  metrics.txt       EER per codec/bitrate + delta vs exp1 baseline
  roc.png / det.png one subplot per codec, one line per bitrate

Usage:
  python experiments/exp6_codec.py
  python experiments/exp6_codec.py --n_genuine 100 --n_impostor 100
  python experiments/exp6_codec.py --rerun
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import csv
import time
import random
import argparse
import tempfile
import subprocess
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from enroll import load_model, load_db, preprocess_wav, get_embedding, SAMPLE_RATE
from auth import score_against_speaker
from experiments.exp1_baseline import (
    compute_roc, compute_eer, far_at_frr, frr_at_far,
    load_genuine_scores, load_impostor_scores,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

GENUINE_CSV = "results/exp1_baseline.csv"
IMP_CSV     = "results/exp1/impostor_scores.csv"
OUT_DIR     = Path("results/exp6")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"
DET_PNG     = OUT_DIR / "det.png"

# codec → (ffmpeg encoder name, file extension, bitrates_kbps)
CODECS = {
    "mp3":  ("libmp3lame", ".mp3",  [32, 64, 128]),
    "aac":  ("aac",        ".m4a",  [32, 64, 128]),
    "opus": ("libopus",    ".opus", [8,  16,  32]),
}

# color per codec; linestyle per bitrate position
CODEC_COLORS = {"mp3": "#3498db", "aac": "#2ecc71", "opus": "#e74c3c"}
BITRATE_STYLES = ["-", "--", ":"]


# ── Codec compression ──────────────────────────────────────────────────────────

def compress_decompress(wav_path: str, codec: str, bitrate_kbps: int,
                        ffmpeg_bin: str = "ffmpeg") -> np.ndarray:
    """
    Encodes wav_path with the given codec/bitrate via ffmpeg, then decodes
    back to 16 kHz mono PCM.  Returns a float32 numpy array (samples,).
    Raises RuntimeError if ffmpeg fails.
    """
    encoder, ext, _ = CODECS[codec]

    fd_c, path_c = tempfile.mkstemp(suffix=ext)
    fd_d, path_d = tempfile.mkstemp(suffix=".wav")
    os.close(fd_c)
    os.close(fd_d)

    try:
        # Encode to compressed format
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-i", wav_path,
             "-c:a", encoder, "-b:a", f"{bitrate_kbps}k",
             path_c],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {r.stderr.decode()[-300:]}")

        # Decode back to 16 kHz mono PCM wav
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-i", path_c,
             "-ar", str(SAMPLE_RATE), "-ac", "1",
             path_d],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode()[-300:]}")

        data, _ = sf.read(path_d, dtype="float32")
        return data
    finally:
        for p in (path_c, path_d):
            try:
                os.unlink(p)
            except OSError:
                pass


def get_embedding_from_array(model, audio: np.ndarray,
                              chunk_sec: float = 3.0) -> np.ndarray:
    """
    Converts a float32 numpy array (already at 16 kHz) to a normalised
    embedding, applying the same centre-crop / repeat-pad as preprocess_wav.
    """
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    chunk_samples = int(chunk_sec * SAMPLE_RATE)

    if wav.shape[1] >= chunk_samples:
        start = (wav.shape[1] - chunk_samples) // 2
        wav   = wav[:, start: start + chunk_samples]
    else:
        repeats = (chunk_samples // wav.shape[1]) + 1
        wav     = wav.repeat(1, repeats)[:, :chunk_samples]

    return get_embedding(model, wav)


# ── Trial running ──────────────────────────────────────────────────────────────

def run_trials(model, db, trials: list[dict], ffmpeg_bin: str = "ffmpeg") -> list[dict]:
    results = []
    for trial in tqdm(trials):
        codec   = trial["codec"]
        bitrate = trial["bitrate_kbps"]
        try:
            t0    = time.perf_counter()
            audio = compress_decompress(trial["wav_path"], codec, bitrate, ffmpeg_bin)
            emb   = get_embedding_from_array(model, audio)
            dist  = score_against_speaker(emb, db[trial["claimed_id"]]["embeddings"])
            lat   = (time.perf_counter() - t0) * 1000

            results.append({
                "wav_path":    trial["wav_path"],
                "claimed_id":  trial["claimed_id"],
                "genuine":     trial["genuine"],
                "codec":       codec,
                "bitrate_kbps": bitrate,
                "distance":    dist,
                "latency_ms":  lat,
            })
        except Exception as e:
            print(f"  [warn] Skipping {trial['wav_path']} "
                  f"({codec}@{bitrate}kbps): {e}")
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_curves(results, eer_baseline, out_path, kind="roc"):
    eps  = 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=(kind == "det"))

    for ax, codec in zip(axes, CODECS):
        color    = CODEC_COLORS[codec]
        _, _, bitrates = CODECS[codec]

        for i, br in enumerate(bitrates):
            sub = [r for r in results
                   if r["codec"] == codec and r["bitrate_kbps"] == br]
            gen = np.array([r["distance"] for r in sub if     r["genuine"]])
            imp = np.array([r["distance"] for r in sub if not r["genuine"]])
            if len(gen) == 0 or len(imp) == 0:
                continue

            thresholds, far, frr = compute_roc(gen, imp)
            eer, _ = compute_eer(far, frr, thresholds)
            ls     = BITRATE_STYLES[i]
            label  = f"{br} kbps  EER={eer:.2%}"

            if kind == "roc":
                ax.plot(far, 1 - frr, lw=2, color=color, ls=ls, label=label)
            else:
                far_nd = stats.norm.ppf(np.clip(far, eps, 1 - eps))
                frr_nd = stats.norm.ppf(np.clip(frr, eps, 1 - eps))
                ax.plot(far_nd, frr_nd, lw=2, color=color, ls=ls, label=label)

        if kind == "roc":
            ax.axhline(1 - eer_baseline, color="grey", lw=1, ls="--",
                       label=f"Baseline EER={eer_baseline:.2%}")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("FAR")
            ax.set_ylabel("TAR (1 − FRR)")
        else:
            ticks     = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
            tick_vals = stats.norm.ppf(ticks)
            tick_lbls = [f"{t*100:.1f}" for t in ticks]
            eer_nd    = stats.norm.ppf(max(eer_baseline, eps))
            ax.scatter([eer_nd], [eer_nd], color="grey", zorder=5,
                       label=f"Baseline EER={eer_baseline:.2%}")
            ax.plot([tick_vals[0], tick_vals[-1]],
                    [tick_vals[0], tick_vals[-1]], "k--", lw=0.8)
            ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls)
            ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls)
            ax.set_xlabel("FAR (%)")
            ax.set_ylabel("FRR (%)")

        ax.set_title(f"{codec.upper()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    curve = "ROC" if kind == "roc" else "DET"
    fig.suptitle(f"{curve} — Experiment 6: Codec Compression", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp6] {curve} saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine_csv", default=GENUINE_CSV)
    parser.add_argument("--imp_csv",     default=IMP_CSV)
    parser.add_argument("--db_path",     default="speaker_db.json")
    parser.add_argument("--n_genuine",   type=int, default=None,
                        help="Genuine trials per (codec, bitrate) — default: 100")
    parser.add_argument("--n_impostor",  type=int, default=None,
                        help="Impostor trials per (codec, bitrate) — default: 100")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    # Locate ffmpeg: check PATH first, then known winget install path
    import shutil, glob as _glob
    _FFMPEG_BIN = shutil.which("ffmpeg")
    if _FFMPEG_BIN is None:
        _localappdata = os.environ.get("LOCALAPPDATA", "")
        _candidates = _glob.glob(
            os.path.join(_localappdata, "Microsoft", "WinGet", "Packages",
                         "Gyan.FFmpeg*", "ffmpeg-*", "bin", "ffmpeg.exe")
        ) + _glob.glob(
            os.path.join(_localappdata, "Microsoft", "WinGet", "Links", "ffmpeg.exe")
        )
        if _candidates:
            _FFMPEG_BIN = _candidates[0]
    if _FFMPEG_BIN is None:
        print("[exp6] ERROR: ffmpeg not found. Install ffmpeg and retry.")
        sys.exit(1)
    print(f"[exp6] Using ffmpeg: {_FFMPEG_BIN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline EER ─────────────────────────────────────────────────────────
    exp1_gen = np.array([s[0] for s in load_genuine_scores(args.genuine_csv)])
    exp1_imp = np.array([s[0] for s in load_impostor_scores(Path(args.imp_csv))])
    _, e1_far, e1_frr = compute_roc(exp1_gen, exp1_imp)
    eer_baseline, _   = compute_eer(e1_far, e1_frr, _)

    # ── Build trial list ──────────────────────────────────────────────────────
    rng = random.Random(args.seed)

    all_genuine = []
    with open(args.genuine_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["genuine"].lower() == "true":
                all_genuine.append({"wav_path":   row["wav_path"],
                                    "claimed_id": row["claimed_id"],
                                    "genuine":    True})

    all_impostor = []
    with open(args.imp_csv, newline="") as f:
        for row in csv.DictReader(f):
            all_impostor.append({"wav_path":   row["wav_path"],
                                 "claimed_id": row["claimed_id"],
                                 "genuine":    False})

    n_gen = args.n_genuine  or 100
    n_imp = args.n_impostor or 100

    trials = []
    for codec, (_, _, bitrates) in CODECS.items():
        for br in bitrates:
            for row in rng.sample(all_genuine,  min(n_gen, len(all_genuine))):
                trials.append({**row, "codec": codec, "bitrate_kbps": br})
            for row in rng.sample(all_impostor, min(n_imp, len(all_impostor))):
                trials.append({**row, "codec": codec, "bitrate_kbps": br})

    n_conditions = sum(len(brs) for _, _, brs in CODECS.values())
    print(f"[exp6] {len(trials)} total trials  "
          f"({n_gen} genuine + {n_imp} impostor) × {n_conditions} codec/bitrate combos")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"[exp6] Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "wav_path":    row["wav_path"],
                    "claimed_id":  row["claimed_id"],
                    "genuine":     row["genuine"].lower() == "true",
                    "codec":       row["codec"],
                    "bitrate_kbps": int(row["bitrate_kbps"]),
                    "distance":    float(row["distance"]),
                    "latency_ms":  float(row["latency_ms"]),
                })
    else:
        model   = load_model()
        db      = load_db(args.db_path)
        results = run_trials(model, db, trials, _FFMPEG_BIN)

        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "claimed_id", "genuine",
                                                    "codec", "bitrate_kbps",
                                                    "distance", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[exp6] Scores saved to {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = ["Experiment 6 - Lossy Codec Compression",
             "=" * 50,
             f"Baseline EER (Exp 1): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
             ""]

    for codec, (_, _, bitrates) in CODECS.items():
        lines.append(f"Codec: {codec.upper()}")
        for br in bitrates:
            sub = [r for r in results
                   if r["codec"] == codec and r["bitrate_kbps"] == br]
            gen = np.array([r["distance"] for r in sub if     r["genuine"]])
            imp = np.array([r["distance"] for r in sub if not r["genuine"]])
            if len(gen) == 0 or len(imp) == 0:
                lines.append(f"  {br:3d} kbps : no data")
                continue

            thresholds, far, frr = compute_roc(gen, imp)
            eer, eer_t  = compute_eer(far, frr, thresholds)
            f1pct       = far_at_frr(far, frr, 0.01)
            r1pct       = frr_at_far(far, frr, 0.01)
            avg_lat     = np.mean([r["latency_ms"] for r in sub])
            delta       = eer - eer_baseline

            lines += [
                f"  {br:3d} kbps",
                f"    Trials       : {len(gen)} genuine, {len(imp)} impostor",
                f"    EER          : {eer:.4f} ({eer*100:.2f}%)  delta vs baseline: {delta*100:+.2f}pp",
                f"    EER threshold: {eer_t:.4f}",
                f"    FAR@FRR=1%   : {f1pct:.4f} ({f1pct*100:.2f}%)",
                f"    FRR@FAR=1%   : {r1pct:.4f} ({r1pct*100:.2f}%)",
                f"    Avg latency  : {avg_lat:.1f} ms",
            ]
        lines.append("")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[exp6] Metrics saved to {METRICS_TXT}")

    plot_curves(results, eer_baseline, ROC_PNG, kind="roc")
    plot_curves(results, eer_baseline, DET_PNG, kind="det")


if __name__ == "__main__":
    main()
