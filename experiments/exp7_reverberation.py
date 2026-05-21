"""
experiments/exp7_reverberation.py

Experiment 7 — Room reverberation (RIR convolution).

Each speech clip is convolved with a room impulse response (RIR) from a
provided directory (e.g. OpenSLR SLR28).  RIRs are automatically binned
into three reverb conditions by their estimated RT60:
  short  : RT60 < 0.3 s
  medium : 0.3 s ≤ RT60 < 0.7 s
  long   : RT60 ≥ 0.7 s

RT60 is estimated via Schroeder backward integration of the RIR energy
decay curve (EDC).  Only conditions with at least one matching RIR file
are tested.

RIR datasets compatible with this script:
  - OpenSLR SLR28: https://openslr.org/28/
  - MIT AIR database, RVB2014, REVERB Challenge RIRs

Convolution uses scipy.signal.fftconvolve.  The output is truncated to
the original speech length and peak-normalised to prevent clipping.

Outputs (results/exp7/):
  scores.csv        distance, rt60_bin, rt60_s, rir_file, genuine per trial
  metrics.txt       EER per RT60 bin + delta vs exp1 baseline
  roc.png / det.png curves per RT60 bin

Usage:
  python experiments/exp7_reverberation.py --rir_dir path/to/SLR28
  python experiments/exp7_reverberation.py --rir_dir path/to/SLR28 --n_genuine 100 --n_impostor 100
  python experiments/exp7_reverberation.py --rir_dir path/to/SLR28 --rerun
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import csv
import time
import random
import argparse
import numpy as np
import torch
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.signal import fftconvolve

from enroll import load_model, load_db, preprocess_wav, get_embedding, SAMPLE_RATE
from auth import score_against_speaker
from experiments.exp1_baseline import (
    compute_roc, compute_eer, far_at_frr, frr_at_far,
    load_genuine_scores, load_impostor_scores,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

GENUINE_CSV = "results/exp1_baseline.csv"
IMP_CSV     = "results/exp1/impostor_scores.csv"
OUT_DIR     = Path("results/exp7")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"
DET_PNG     = OUT_DIR / "det.png"

RT60_BINS   = ["short", "medium", "long"]
RT60_BOUNDS = {"short": (0.0, 0.3), "medium": (0.3, 0.7), "long": (0.7, float("inf"))}
COLORS      = {"short": "#2ecc71", "medium": "#e67e22", "long": "#e74c3c"}


# ── RIR loading and RT60 estimation ───────────────────────────────────────────

def load_rir_files(rir_dir: str) -> list[Path]:
    root  = Path(rir_dir)
    files = list(root.rglob("*.wav")) + list(root.rglob("*.WAV"))
    if not files:
        raise FileNotFoundError(
            f"No .wav files found in '{rir_dir}'.\n"
            "Download OpenSLR SLR28 from https://openslr.org/28/ and pass --rir_dir."
        )
    print(f"[exp7] Found {len(files)} RIR files in {rir_dir}")
    return files


def estimate_rt60(rir_np: np.ndarray, sr: int) -> float:
    """
    Estimate RT60 via Schroeder backward integration.
    Uses the -60 dB crossing of the energy decay curve.
    Falls back to signal duration if the curve doesn't reach -60 dB.
    """
    h2  = rir_np.astype(np.float64) ** 2
    edc = np.cumsum(h2[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        edc_db = 10.0 * np.log10(edc / (edc[0] + 1e-12) + 1e-12)
    idx = np.where(edc_db <= -60.0)[0]
    return float(idx[0] / sr) if len(idx) > 0 else float(len(rir_np) / sr)


def load_rir(rir_path: Path) -> tuple[np.ndarray, float]:
    """
    Load a RIR file, resample to 16 kHz, convert to mono.
    Returns (rir_np, rt60_seconds).
    """
    try:
        data, sr = sf.read(str(rir_path), dtype="float32", always_2d=True)
    except Exception as e:
        raise RuntimeError(f"Cannot read {rir_path}: {e}")

    rir = torch.from_numpy(data.T)  # (C, T)
    if sr != SAMPLE_RATE:
        rir = torchaudio.functional.resample(rir, sr, SAMPLE_RATE)
    if rir.shape[0] > 1:
        rir = rir.mean(dim=0, keepdim=True)

    rir_np = rir.squeeze().numpy()
    rt60   = estimate_rt60(rir_np, SAMPLE_RATE)
    return rir_np, rt60


def bin_rirs(rir_files: list[Path]) -> dict[str, list[tuple[Path, np.ndarray, float]]]:
    """
    Load all RIRs, estimate their RT60, and bin into short/medium/long.
    Returns {bin_name: [(path, rir_np, rt60), ...]}.
    """
    print("[exp7] Loading and analysing RIRs...")
    bins: dict[str, list] = {b: [] for b in RT60_BINS}
    skipped = 0

    for path in tqdm(rir_files):
        try:
            rir_np, rt60 = load_rir(path)
            for b, (lo, hi) in RT60_BOUNDS.items():
                if lo <= rt60 < hi:
                    bins[b].append((path, rir_np, rt60))
                    break
        except Exception as e:
            skipped += 1

    if skipped:
        print(f"[exp7] Skipped {skipped} unreadable RIR files")
    for b, items in bins.items():
        lo, hi = RT60_BOUNDS[b]
        hi_str = f"{hi:.1f}" if hi != float("inf") else "∞"
        print(f"  {b:6s} (RT60 {lo:.1f}–{hi_str} s): {len(items)} RIRs")

    return bins


# ── Reverberation application ─────────────────────────────────────────────────

def apply_rir(speech: torch.Tensor, rir_np: np.ndarray) -> torch.Tensor:
    """
    Convolve speech with the RIR.  Truncates output to original length
    and peak-normalises to prevent clipping.
    """
    speech_np = speech.squeeze().numpy().astype(np.float64)
    rir_norm  = rir_np / (np.abs(rir_np).max() + 1e-8)
    convolved = fftconvolve(speech_np, rir_norm, mode="full")[: len(speech_np)]
    peak = np.abs(convolved).max()
    if peak > 0:
        convolved = convolved / peak * 0.9
    return torch.from_numpy(convolved.astype(np.float32)).unsqueeze(0)


# ── Trial running ──────────────────────────────────────────────────────────────

def run_trials(model, db, trials: list[dict]) -> list[dict]:
    results = []
    for trial in tqdm(trials):
        rir_np   = trial["rir_np"]
        rt60_bin = trial["rt60_bin"]
        rt60_s   = trial["rt60_s"]
        try:
            t0     = time.perf_counter()
            speech = preprocess_wav(trial["wav_path"])
            reverb = apply_rir(speech, rir_np)
            emb    = get_embedding(model, reverb)
            dist   = score_against_speaker(emb, db[trial["claimed_id"]]["embeddings"])
            lat    = (time.perf_counter() - t0) * 1000

            results.append({
                "wav_path":   trial["wav_path"],
                "claimed_id": trial["claimed_id"],
                "genuine":    trial["genuine"],
                "rt60_bin":   rt60_bin,
                "rt60_s":     rt60_s,
                "rir_file":   trial["rir_file"],
                "distance":   dist,
                "latency_ms": lat,
            })
        except Exception as e:
            print(f"  [warn] Skipping {trial['wav_path']} ({rt60_bin}): {e}")
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_curves(results, eer_baseline, active_bins, out_path, kind="roc"):
    eps = 1e-6
    fig, ax = plt.subplots(figsize=(7, 6))

    for b in active_bins:
        sub = [r for r in results if r["rt60_bin"] == b]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])
        if len(gen) == 0 or len(imp) == 0:
            continue

        lo, hi = RT60_BOUNDS[b]
        hi_str = f"{hi:.1f}" if hi != float("inf") else "∞"
        thresholds, far, frr = compute_roc(gen, imp)
        eer, _ = compute_eer(far, frr, thresholds)
        label  = f"{b} (RT60 {lo:.1f}–{hi_str} s)  EER={eer:.2%}"
        color  = COLORS[b]

        if kind == "roc":
            ax.plot(far, 1 - frr, lw=2, color=color, label=label)
        else:
            far_nd = stats.norm.ppf(np.clip(far, eps, 1 - eps))
            frr_nd = stats.norm.ppf(np.clip(frr, eps, 1 - eps))
            ax.plot(far_nd, frr_nd, lw=2, color=color, label=label)

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

    curve = "ROC" if kind == "roc" else "DET"
    ax.set_title(f"{curve} — Experiment 7: Room Reverberation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp7] {curve} saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir_dir",     required=True,
                        help="Root folder containing RIR .wav files (e.g. OpenSLR SLR28)")
    parser.add_argument("--genuine_csv", default=GENUINE_CSV)
    parser.add_argument("--imp_csv",     default=IMP_CSV)
    parser.add_argument("--db_path",     default="speaker_db.json")
    parser.add_argument("--n_genuine",   type=int, default=None,
                        help="Genuine trials per RT60 bin (default: 100)")
    parser.add_argument("--n_impostor",  type=int, default=None,
                        help="Impostor trials per RT60 bin (default: 100)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
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

    # ── Load and bin RIRs ─────────────────────────────────────────────────────
    rir_files = load_rir_files(args.rir_dir)
    rir_bins  = bin_rirs(rir_files)
    active_bins = [b for b in RT60_BINS if rir_bins[b]]

    if not active_bins:
        print("[exp7] ERROR: No usable RIR files found after binning.")
        sys.exit(1)

    trials = []
    for b in active_bins:
        bin_rirs_list = rir_bins[b]
        for row in rng.sample(all_genuine,  min(n_gen, len(all_genuine))):
            path, rir_np, rt60 = random.choice(bin_rirs_list)
            trials.append({**row, "rt60_bin": b, "rt60_s": rt60,
                           "rir_np": rir_np, "rir_file": path.name})
        for row in rng.sample(all_impostor, min(n_imp, len(all_impostor))):
            path, rir_np, rt60 = random.choice(bin_rirs_list)
            trials.append({**row, "rt60_bin": b, "rt60_s": rt60,
                           "rir_np": rir_np, "rir_file": path.name})

    print(f"[exp7] {len(trials)} total trials  "
          f"({n_gen} genuine + {n_imp} impostor) × {len(active_bins)} RT60 bins")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"[exp7] Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "wav_path":   row["wav_path"],
                    "claimed_id": row["claimed_id"],
                    "genuine":    row["genuine"].lower() == "true",
                    "rt60_bin":   row["rt60_bin"],
                    "rt60_s":     float(row["rt60_s"]),
                    "rir_file":   row["rir_file"],
                    "distance":   float(row["distance"]),
                    "latency_ms": float(row["latency_ms"]),
                })
        active_bins = sorted(set(r["rt60_bin"] for r in results),
                             key=RT60_BINS.index)
    else:
        model   = load_model()
        db      = load_db(args.db_path)
        results = run_trials(model, db, trials)

        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "claimed_id", "genuine",
                                                    "rt60_bin", "rt60_s", "rir_file",
                                                    "distance", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[exp7] Scores saved to {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = ["Experiment 7 - Room Reverberation (RIR Convolution)",
             "=" * 50,
             f"Baseline EER (Exp 1): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
             ""]

    for b in active_bins:
        sub = [r for r in results if r["rt60_bin"] == b]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])
        lo, hi = RT60_BOUNDS[b]
        hi_str = f"{hi:.1f}" if hi != float("inf") else "∞"
        avg_rt60 = np.mean([r["rt60_s"] for r in sub])

        thresholds, far, frr = compute_roc(gen, imp)
        eer, eer_t  = compute_eer(far, frr, thresholds)
        f1pct       = far_at_frr(far, frr, 0.01)
        r1pct       = frr_at_far(far, frr, 0.01)
        avg_lat     = np.mean([r["latency_ms"] for r in sub])
        delta       = eer - eer_baseline

        lines += [
            f"RT60 bin: {b}  ({lo:.1f}–{hi_str} s,  avg RT60={avg_rt60:.3f} s)",
            f"  Trials       : {len(gen)} genuine, {len(imp)} impostor",
            f"  EER          : {eer:.4f} ({eer*100:.2f}%)  delta vs baseline: {delta*100:+.2f}pp",
            f"  EER threshold: {eer_t:.4f}",
            f"  FAR@FRR=1%   : {f1pct:.4f} ({f1pct*100:.2f}%)",
            f"  FRR@FAR=1%   : {r1pct:.4f} ({r1pct*100:.2f}%)",
            f"  Avg latency  : {avg_lat:.1f} ms",
            "",
        ]

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[exp7] Metrics saved to {METRICS_TXT}")

    plot_curves(results, eer_baseline, active_bins, ROC_PNG, kind="roc")
    plot_curves(results, eer_baseline, active_bins, DET_PNG, kind="det")


if __name__ == "__main__":
    main()
