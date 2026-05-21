"""
experiments/exp2_amplitude.py

Experiment 2 — Amplitude scaling robustness.

For each trial, the audio amplitude is multiplied by a factor drawn uniformly
at random from {25, 1, 0.04}.  Scale=25 clips to [-1, 1] (ADC saturation);
scale=0.04 produces a very quiet signal.

Genuine / impostor trial lists are reused from Experiment 1 (same wav paths
and claimed_id assignments), so results are directly comparable.

Outputs (results/exp2/):
  scores.csv          distance, scale_factor, genuine flag per trial
  metrics.txt         EER per scale factor + overall, vs exp1 baseline
  roc.png             ROC curves per scale factor
  det.png             DET curves per scale factor

Usage:
  python experiments/exp2_amplitude.py
  python experiments/exp2_amplitude.py --n_genuine 500 --n_impostor 500
  python experiments/exp2_amplitude.py --rerun
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import csv
import time
import random
import argparse
import numpy as np
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

GENUINE_CSV  = "results/exp1_baseline.csv"
IMP_CSV      = "results/exp1/impostor_scores.csv"
EXP1_METRICS = "results/exp1/metrics.txt"
OUT_DIR      = Path("results/exp2")
SCORES_CSV   = OUT_DIR / "scores.csv"
METRICS_TXT  = OUT_DIR / "metrics.txt"
ROC_PNG      = OUT_DIR / "roc.png"
DET_PNG      = OUT_DIR / "det.png"

SCALE_FACTORS = [25.0, 1.0, 0.04]
SCALE_LABELS  = {25.0: "×25 (loud/clip)", 1.0: "×1 (unchanged)", 0.04: "×0.04 (quiet)"}
COLORS        = {25.0: "#e74c3c", 1.0: "#2ecc71", 0.04: "#3498db"}


# ── Audio perturbation ─────────────────────────────────────────────────────────

def scale_amplitude(wav: "torch.Tensor", factor: float) -> "torch.Tensor":
    import torch
    wav = wav * factor
    wav = torch.clamp(wav, -1.0, 1.0)  # simulate ADC clipping
    return wav


# ── Trial running ──────────────────────────────────────────────────────────────

def run_trials(model, db, trials: list[dict], seed: int = 42) -> list[dict]:
    """
    trials: list of dicts with keys wav_path, claimed_id, genuine (bool)
    Returns list of dicts adding: distance, scale_factor, latency_ms
    """
    rng = random.Random(seed)
    results = []

    for trial in tqdm(trials):
        factor = rng.choice(SCALE_FACTORS)
        try:
            t0  = time.perf_counter()
            wav = preprocess_wav(trial["wav_path"])
            wav = scale_amplitude(wav, factor)
            emb = get_embedding(model, wav)
            dist = score_against_speaker(emb, db[trial["claimed_id"]]["embeddings"])
            lat  = (time.perf_counter() - t0) * 1000

            results.append({
                "wav_path":     trial["wav_path"],
                "claimed_id":   trial["claimed_id"],
                "genuine":      trial["genuine"],
                "scale_factor": factor,
                "distance":     dist,
                "latency_ms":   lat,
            })
        except Exception as e:
            print(f"  [warn] Skipping {trial['wav_path']}: {e}")

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_roc_multi(results: list[dict], eer_baseline: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for factor in SCALE_FACTORS:
        sub = [r for r in results if r["scale_factor"] == factor]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])
        if len(gen) == 0 or len(imp) == 0:
            continue
        _, far, frr = compute_roc(gen, imp)
        eer, _ = compute_eer(far, frr, _)
        ax.plot(far, 1 - frr, lw=2, color=COLORS[factor],
                label=f"{SCALE_LABELS[factor]}  EER={eer:.2%}")

    ax.axhline(1 - eer_baseline, color="grey", lw=1, linestyle="--",
               label=f"Exp1 baseline EER={eer_baseline:.2%}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR (1 − FRR)")
    ax.set_title("ROC — Experiment 2: Amplitude Scaling")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp2] ROC saved to {out_path}")


def plot_det_multi(results: list[dict], eer_baseline: float, out_path: Path):
    eps = 1e-6
    ticks     = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    tick_vals = stats.norm.ppf(ticks)
    tick_lbls = [f"{t*100:.1f}" for t in ticks]

    fig, ax = plt.subplots(figsize=(7, 6))

    for factor in SCALE_FACTORS:
        sub = [r for r in results if r["scale_factor"] == factor]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])
        if len(gen) == 0 or len(imp) == 0:
            continue
        _, far, frr = compute_roc(gen, imp)
        eer, _ = compute_eer(far, frr, _)
        far_nd = stats.norm.ppf(np.clip(far, eps, 1 - eps))
        frr_nd = stats.norm.ppf(np.clip(frr, eps, 1 - eps))
        ax.plot(far_nd, frr_nd, lw=2, color=COLORS[factor],
                label=f"{SCALE_LABELS[factor]}  EER={eer:.2%}")

    eer_nd = stats.norm.ppf(max(eer_baseline, eps))
    ax.scatter([eer_nd], [eer_nd], color="grey", zorder=5,
               label=f"Exp1 EER={eer_baseline:.2%}")
    ax.plot([tick_vals[0], tick_vals[-1]], [tick_vals[0], tick_vals[-1]], "k--", lw=0.8)
    ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls)
    ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET — Experiment 2: Amplitude Scaling")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp2] DET saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine_csv",  default=GENUINE_CSV)
    parser.add_argument("--imp_csv",      default=IMP_CSV)
    parser.add_argument("--db_path",      default="speaker_db.json")
    parser.add_argument("--n_genuine",    type=int, default=None,
                        help="Genuine trials to use (default: all)")
    parser.add_argument("--n_impostor",   type=int, default=None,
                        help="Impostor trials to use (default: all)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--rerun",        action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Exp1 baseline EER (for comparison line on plots) ─────────────────────
    exp1_genuine  = np.array([s[0] for s in load_genuine_scores(args.genuine_csv)])
    exp1_impostor = np.array([s[0] for s in load_impostor_scores(Path(args.imp_csv))])
    _, e1_far, e1_frr = compute_roc(exp1_genuine, exp1_impostor)
    eer_baseline, _ = compute_eer(e1_far, e1_frr, _)

    # ── Build trial list ──────────────────────────────────────────────────────
    rng = random.Random(args.seed)

    genuine_rows = []
    with open(args.genuine_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["genuine"].lower() == "true":
                genuine_rows.append({"wav_path": row["wav_path"],
                                     "claimed_id": row["claimed_id"],
                                     "genuine": True})
    if args.n_genuine:
        genuine_rows = rng.sample(genuine_rows, min(args.n_genuine, len(genuine_rows)))

    impostor_rows = []
    with open(args.imp_csv, newline="") as f:
        for row in csv.DictReader(f):
            impostor_rows.append({"wav_path": row["wav_path"],
                                  "claimed_id": row["claimed_id"],
                                  "genuine": False})
    if args.n_impostor:
        impostor_rows = rng.sample(impostor_rows, min(args.n_impostor, len(impostor_rows)))

    trials = genuine_rows + impostor_rows
    print(f"[exp2] {len(genuine_rows)} genuine + {len(impostor_rows)} impostor trials")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"[exp2] Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "wav_path":     row["wav_path"],
                    "claimed_id":   row["claimed_id"],
                    "genuine":      row["genuine"].lower() == "true",
                    "scale_factor": float(row["scale_factor"]),
                    "distance":     float(row["distance"]),
                    "latency_ms":   float(row["latency_ms"]),
                })
    else:
        model   = load_model()
        db      = load_db(args.db_path)
        results = run_trials(model, db, trials, seed=args.seed)

        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "claimed_id", "genuine",
                                                    "scale_factor", "distance", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[exp2] Scores saved to {SCORES_CSV}")

    # ── Metrics per scale factor ──────────────────────────────────────────────
    lines = ["Experiment 2 — Amplitude Scaling",
             "=" * 40,
             f"Baseline EER (Exp 1): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
             ""]

    for factor in SCALE_FACTORS:
        sub = [r for r in results if r["scale_factor"] == factor]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])

        thresholds, far, frr = compute_roc(gen, imp)
        eer, eer_t = compute_eer(far, frr, thresholds)
        f1pct = far_at_frr(far, frr, 0.01)
        r1pct = frr_at_far(far, frr, 0.01)
        avg_lat = np.mean([r["latency_ms"] for r in sub])
        delta = eer - eer_baseline

        lines += [
            f"Scale factor {factor}  ({SCALE_LABELS[factor]})",
            f"  Trials      : {len(gen)} genuine, {len(imp)} impostor",
            f"  EER         : {eer:.4f} ({eer*100:.2f}%)  Δ vs baseline: {delta*100:+.2f}pp",
            f"  EER threshold: {eer_t:.4f}",
            f"  FAR@FRR=1%  : {f1pct:.4f} ({f1pct*100:.2f}%)",
            f"  FRR@FAR=1%  : {r1pct:.4f} ({r1pct*100:.2f}%)",
            f"  Avg latency : {avg_lat:.1f} ms",
            "",
        ]

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[exp2] Metrics saved to {METRICS_TXT}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_multi(results, eer_baseline, ROC_PNG)
    plot_det_multi(results, eer_baseline, DET_PNG)


if __name__ == "__main__":
    main()
