"""
experiments/exp3_downsample.py

Experiment 3 — Downsampling robustness.

Two methods tested for factors 2×, 5×, 10×:
  A. Naive  — keep every Nth sample (no anti-aliasing), update sr metadata
  B. Proper — torchaudio sinc-interpolation resample (bandlimited, anti-aliased)

After downsampling the model's preprocessing upsamples back to 16 kHz before
embedding, so the pipeline stays identical — only the information loss differs.

Extra measurement: effective clip duration vs factor.  Naive downsampling at
factor K on a 3 s clip leaves 3/K seconds of unique audio content (aliased);
proper resampling preserves the full 3 s but band-limits to 8/3.2/1.6 kHz.

Outputs (results/exp3/):
  scores.csv          distance, method, factor, genuine per trial
  metrics.txt         EER per variant + vs exp1 baseline
  roc.png / det.png   curves per variant

Usage:
  python experiments/exp3_downsample.py
  python experiments/exp3_downsample.py --n_genuine 200 --n_impostor 200
  python experiments/exp3_downsample.py --rerun
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
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from enroll import load_model, load_db, preprocess_wav, get_embedding
from auth import score_against_speaker, DEFAULT_THRESHOLD
from experiments.exp1_baseline import (
    compute_roc, compute_eer, far_at_frr, frr_at_far,
    load_genuine_scores, load_impostor_scores,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

GENUINE_CSV = "results/exp1_baseline.csv"
IMP_CSV     = "results/exp1/impostor_scores.csv"
OUT_DIR     = Path("results/exp3")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"
DET_PNG     = OUT_DIR / "det.png"

SAMPLE_RATE = 16000
FACTORS     = [2, 5, 10]
METHODS     = ["naive", "proper"]

# Colours: method × factor
_CMAP = plt.cm.get_cmap("tab10")
COLORS = {
    ("naive",  2):  _CMAP(0),
    ("naive",  5):  _CMAP(1),
    ("naive",  10): _CMAP(2),
    ("proper", 2):  _CMAP(3),
    ("proper", 5):  _CMAP(4),
    ("proper", 10): _CMAP(5),
}


# ── Downsampling ───────────────────────────────────────────────────────────────

def downsample_naive(wav: torch.Tensor, factor: int, orig_sr: int) -> tuple[torch.Tensor, int]:
    """Keep every Nth sample — no anti-aliasing.  Returns (wav, new_sr)."""
    new_sr = orig_sr // factor
    return wav[:, ::factor], new_sr


def downsample_proper(wav: torch.Tensor, factor: int, orig_sr: int) -> tuple[torch.Tensor, int]:
    """Bandlimited sinc resampling.  Returns (wav, new_sr)."""
    new_sr = orig_sr // factor
    wav_down = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=new_sr)
    return wav_down, new_sr


def upsample_to_16k(wav: torch.Tensor, current_sr: int) -> torch.Tensor:
    """Bring back to 16 kHz so preprocess_wav pipeline is unchanged."""
    if current_sr == SAMPLE_RATE:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=current_sr, new_freq=SAMPLE_RATE)


# ── Embedding with perturbation ────────────────────────────────────────────────

def embed_downsampled(wav_path: str, method: str, factor: int) -> torch.Tensor:
    """
    Loads audio, downsamples by factor using method, upsamples back to 16 kHz,
    applies the standard centre-crop/pad, returns raw wav tensor for get_embedding.
    preprocess_wav is reused for crop/pad only — we bypass its load step.
    """
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (C, T)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Downsample
    if method == "naive":
        wav, new_sr = downsample_naive(wav, factor, SAMPLE_RATE)
    else:
        wav, new_sr = downsample_proper(wav, factor, SAMPLE_RATE)

    # Upsample back to 16 kHz (pipeline expects this rate)
    wav = upsample_to_16k(wav, new_sr)

    # Centre-crop / repeat-pad to CHUNK_SEC (3 s)
    chunk_samples = int(3.0 * SAMPLE_RATE)
    if wav.shape[1] >= chunk_samples:
        start = (wav.shape[1] - chunk_samples) // 2
        wav   = wav[:, start: start + chunk_samples]
    else:
        repeats = (chunk_samples // wav.shape[1]) + 1
        wav     = wav.repeat(1, repeats)[:, :chunk_samples]

    return wav  # (1, T)


# ── Trial running ──────────────────────────────────────────────────────────────

def run_trials(model, db, trials: list[dict]) -> list[dict]:
    results = []

    for trial in tqdm(trials):
        method = trial["method"]
        factor = trial["factor"]
        try:
            t0  = time.perf_counter()
            wav = embed_downsampled(trial["wav_path"], method, factor)
            emb = get_embedding(model, wav)
            dist = score_against_speaker(emb, db[trial["claimed_id"]]["embeddings"])
            lat  = (time.perf_counter() - t0) * 1000

            results.append({
                "wav_path":   trial["wav_path"],
                "claimed_id": trial["claimed_id"],
                "genuine":    trial["genuine"],
                "method":     method,
                "factor":     factor,
                "distance":   dist,
                "latency_ms": lat,
            })
        except Exception as e:
            print(f"  [warn] Skipping {trial['wav_path']} ({method} ×{factor}): {e}")

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_curves(results, eer_baseline, out_path, kind="roc"):
    eps = 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=(kind == "det"))
    titles = ["Naive (drop every Nth sample)", "Proper (sinc interpolation)"]

    for ax, method, title in zip(axes, METHODS, titles):
        for factor in FACTORS:
            sub = [r for r in results if r["method"] == method and r["factor"] == factor]
            gen = np.array([r["distance"] for r in sub if     r["genuine"]])
            imp = np.array([r["distance"] for r in sub if not r["genuine"]])
            if len(gen) == 0 or len(imp) == 0:
                continue

            thresholds, far, frr = compute_roc(gen, imp)
            eer, _ = compute_eer(far, frr, thresholds)
            color  = COLORS[(method, factor)]

            if kind == "roc":
                ax.plot(far, 1 - frr, lw=2, color=color,
                        label=f"×{factor}  EER={eer:.2%}")
            else:
                far_nd = stats.norm.ppf(np.clip(far, eps, 1 - eps))
                frr_nd = stats.norm.ppf(np.clip(frr, eps, 1 - eps))
                ax.plot(far_nd, frr_nd, lw=2, color=color,
                        label=f"×{factor}  EER={eer:.2%}")

        if kind == "roc":
            ax.axhline(1 - eer_baseline, color="grey", lw=1, ls="--",
                       label=f"Baseline EER={eer_baseline:.2%}")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("FAR"); ax.set_ylabel("TAR (1 − FRR)")
        else:
            ticks     = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
            tick_vals = stats.norm.ppf(ticks)
            tick_lbls = [f"{t*100:.1f}" for t in ticks]
            eer_nd = stats.norm.ppf(max(eer_baseline, eps))
            ax.scatter([eer_nd], [eer_nd], color="grey", zorder=5,
                       label=f"Baseline EER={eer_baseline:.2%}")
            ax.plot([tick_vals[0], tick_vals[-1]], [tick_vals[0], tick_vals[-1]], "k--", lw=0.8)
            ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls)
            ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls)
            ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)")

        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    curve = "ROC" if kind == "roc" else "DET"
    fig.suptitle(f"{curve} — Experiment 3: Downsampling", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp3] {curve} saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine_csv", default=GENUINE_CSV)
    parser.add_argument("--imp_csv",     default=IMP_CSV)
    parser.add_argument("--db_path",     default="speaker_db.json")
    parser.add_argument("--n_genuine",   type=int, default=None,
                        help="Genuine trials per variant (default: all, ~333 each for 2000 total)")
    parser.add_argument("--n_impostor",  type=int, default=None)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline EER ─────────────────────────────────────────────────────────
    exp1_gen = np.array([s[0] for s in load_genuine_scores(args.genuine_csv)])
    exp1_imp = np.array([s[0] for s in load_impostor_scores(Path(args.imp_csv))])
    _, e1_far, e1_frr = compute_roc(exp1_gen, exp1_imp)
    eer_baseline, _   = compute_eer(e1_far, e1_frr, _)

    # ── Build trial list (each variant gets its own sample) ──────────────────
    rng = random.Random(args.seed)

    all_genuine = []
    with open(args.genuine_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["genuine"].lower() == "true":
                all_genuine.append({"wav_path": row["wav_path"],
                                    "claimed_id": row["claimed_id"], "genuine": True})

    all_impostor = []
    with open(args.imp_csv, newline="") as f:
        for row in csv.DictReader(f):
            all_impostor.append({"wav_path": row["wav_path"],
                                 "claimed_id": row["claimed_id"], "genuine": False})

    n_gen = args.n_genuine  or len(all_genuine)  // (len(FACTORS) * len(METHODS))
    n_imp = args.n_impostor or len(all_impostor) // (len(FACTORS) * len(METHODS))

    trials = []
    for method in METHODS:
        for factor in FACTORS:
            g_sample = rng.sample(all_genuine,  min(n_gen, len(all_genuine)))
            i_sample = rng.sample(all_impostor, min(n_imp, len(all_impostor)))
            for row in g_sample + i_sample:
                trials.append({**row, "method": method, "factor": factor})

    print(f"[exp3] {len(trials)} total trials  "
          f"({n_gen} genuine + {n_imp} impostor) × {len(METHODS)} methods × {len(FACTORS)} factors")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"[exp3] Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "wav_path":   row["wav_path"],
                    "claimed_id": row["claimed_id"],
                    "genuine":    row["genuine"].lower() == "true",
                    "method":     row["method"],
                    "factor":     int(row["factor"]),
                    "distance":   float(row["distance"]),
                    "latency_ms": float(row["latency_ms"]),
                })
    else:
        model   = load_model()
        db      = load_db(args.db_path)
        results = run_trials(model, db, trials)

        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "claimed_id", "genuine",
                                                    "method", "factor", "distance", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[exp3] Scores saved to {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = ["Experiment 3 — Downsampling",
             "=" * 50,
             f"Baseline EER (Exp 1): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
             ""]

    for method in METHODS:
        lines.append(f"Method: {method.upper()}")
        lines.append("-" * 30)
        for factor in FACTORS:
            sub = [r for r in results if r["method"] == method and r["factor"] == factor]
            gen = np.array([r["distance"] for r in sub if     r["genuine"]])
            imp = np.array([r["distance"] for r in sub if not r["genuine"]])

            thresholds, far, frr = compute_roc(gen, imp)
            eer, eer_t = compute_eer(far, frr, thresholds)
            f1pct = far_at_frr(far, frr, 0.01)
            r1pct = frr_at_far(far, frr, 0.01)
            avg_lat = np.mean([r["latency_ms"] for r in sub])
            delta = eer - eer_baseline

            # Effective bandwidth after downsampling
            bw = SAMPLE_RATE / (2 * factor)
            lines += [
                f"  Downsample x{factor}  (effective BW: {bw/1000:.1f} kHz)",
                f"    Trials       : {len(gen)} genuine, {len(imp)} impostor",
                f"    EER          : {eer:.4f} ({eer*100:.2f}%)  delta vs baseline: {delta*100:+.2f}pp",
                f"    EER threshold: {eer_t:.4f}",
                f"    FAR@FRR=1%   : {f1pct:.4f} ({f1pct*100:.2f}%)",
                f"    FRR@FAR=1%   : {r1pct:.4f} ({r1pct*100:.2f}%)",
                f"    Avg latency  : {avg_lat:.1f} ms",
                "",
            ]
        lines.append("")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[exp3] Metrics saved to {METRICS_TXT}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_curves(results, eer_baseline, ROC_PNG, kind="roc")
    _plot_curves(results, eer_baseline, DET_PNG, kind="det")


if __name__ == "__main__":
    main()
