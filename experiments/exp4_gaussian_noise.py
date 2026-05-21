"""
experiments/exp4_gaussian_noise.py

Experiment 4 — Additive White Gaussian Noise (AWGN).

For each trial, white Gaussian noise is added at a target SNR relative to the
original signal power.  Three SNR levels tested: 40 dB, 20 dB, 10 dB.

SNR formula:
    signal_power = mean(wav ** 2)
    noise_power  = signal_power / 10^(SNR_dB / 10)
    noise        = sqrt(noise_power) * randn(len(wav))

Outputs (results/exp4/):
  scores.csv        distance, snr_db, genuine per trial
  metrics.txt       EER per SNR level + vs exp1 baseline
  roc.png / det.png curves per SNR level

Usage:
  python experiments/exp4_gaussian_noise.py
  python experiments/exp4_gaussian_noise.py --n_genuine 100 --n_impostor 100
  python experiments/exp4_gaussian_noise.py --rerun
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
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from enroll import load_model, load_db, preprocess_wav, get_embedding
from auth import score_against_speaker
from experiments.exp1_baseline import (
    compute_roc, compute_eer, far_at_frr, frr_at_far,
    load_genuine_scores, load_impostor_scores,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

GENUINE_CSV = "results/exp1_baseline.csv"
IMP_CSV     = "results/exp1/impostor_scores.csv"
OUT_DIR     = Path("results/exp4")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"
DET_PNG     = OUT_DIR / "det.png"

SNR_LEVELS = [40, 20, 10]
COLORS     = {40: "#2ecc71", 20: "#e67e22", 10: "#e74c3c"}


# ── Noise addition ─────────────────────────────────────────────────────────────

def add_awgn(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white Gaussian noise to achieve the target SNR (per-clip calibration)."""
    signal_power = (wav ** 2).mean()
    if signal_power < 1e-10:
        return wav
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * noise_power.sqrt()
    return wav + noise


# ── Trial running ──────────────────────────────────────────────────────────────

def run_trials(model, db, trials: list[dict]) -> list[dict]:
    results = []
    for trial in tqdm(trials):
        snr = trial["snr_db"]
        try:
            t0   = time.perf_counter()
            wav  = preprocess_wav(trial["wav_path"])
            wav  = add_awgn(wav, snr)
            emb  = get_embedding(model, wav)
            dist = score_against_speaker(emb, db[trial["claimed_id"]]["embeddings"])
            lat  = (time.perf_counter() - t0) * 1000

            results.append({
                "wav_path":   trial["wav_path"],
                "claimed_id": trial["claimed_id"],
                "genuine":    trial["genuine"],
                "snr_db":     snr,
                "distance":   dist,
                "latency_ms": lat,
            })
        except Exception as e:
            print(f"  [warn] Skipping {trial['wav_path']} (SNR={snr}dB): {e}")
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_curves(results, eer_baseline, out_path, kind="roc"):
    eps = 1e-6
    fig, ax = plt.subplots(figsize=(7, 6))

    for snr in SNR_LEVELS:
        sub = [r for r in results if r["snr_db"] == snr]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])
        if len(gen) == 0 or len(imp) == 0:
            continue

        thresholds, far, frr = compute_roc(gen, imp)
        eer, _ = compute_eer(far, frr, thresholds)
        color  = COLORS[snr]

        if kind == "roc":
            ax.plot(far, 1 - frr, lw=2, color=color,
                    label=f"SNR={snr} dB  EER={eer:.2%}")
        else:
            far_nd = stats.norm.ppf(np.clip(far, eps, 1 - eps))
            frr_nd = stats.norm.ppf(np.clip(frr, eps, 1 - eps))
            ax.plot(far_nd, frr_nd, lw=2, color=color,
                    label=f"SNR={snr} dB  EER={eer:.2%}")

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
        eer_nd = stats.norm.ppf(max(eer_baseline, eps))
        ax.scatter([eer_nd], [eer_nd], color="grey", zorder=5,
                   label=f"Baseline EER={eer_baseline:.2%}")
        ax.plot([tick_vals[0], tick_vals[-1]], [tick_vals[0], tick_vals[-1]], "k--", lw=0.8)
        ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls)
        ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls)
        ax.set_xlabel("FAR (%)")
        ax.set_ylabel("FRR (%)")

    curve = "ROC" if kind == "roc" else "DET"
    ax.set_title(f"{curve} — Experiment 4: Gaussian Noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp4] {curve} saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine_csv", default=GENUINE_CSV)
    parser.add_argument("--imp_csv",     default=IMP_CSV)
    parser.add_argument("--db_path",     default="speaker_db.json")
    parser.add_argument("--n_genuine",   type=int, default=None,
                        help="Genuine trials per SNR level (default: all / n_levels)")
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

    # ── Build trial list ──────────────────────────────────────────────────────
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

    n_gen = args.n_genuine  or len(all_genuine)  // len(SNR_LEVELS)
    n_imp = args.n_impostor or len(all_impostor) // len(SNR_LEVELS)

    trials = []
    for snr in SNR_LEVELS:
        for row in rng.sample(all_genuine,  min(n_gen, len(all_genuine))):
            trials.append({**row, "snr_db": snr})
        for row in rng.sample(all_impostor, min(n_imp, len(all_impostor))):
            trials.append({**row, "snr_db": snr})

    print(f"[exp4] {len(trials)} total trials  "
          f"({n_gen} genuine + {n_imp} impostor) × {len(SNR_LEVELS)} SNR levels")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"[exp4] Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "wav_path":   row["wav_path"],
                    "claimed_id": row["claimed_id"],
                    "genuine":    row["genuine"].lower() == "true",
                    "snr_db":     int(row["snr_db"]),
                    "distance":   float(row["distance"]),
                    "latency_ms": float(row["latency_ms"]),
                })
    else:
        model   = load_model()
        db      = load_db(args.db_path)
        results = run_trials(model, db, trials)

        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "claimed_id", "genuine",
                                                    "snr_db", "distance", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[exp4] Scores saved to {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = ["Experiment 4 - Additive White Gaussian Noise",
             "=" * 50,
             f"Baseline EER (Exp 1): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
             ""]

    for snr in SNR_LEVELS:
        sub = [r for r in results if r["snr_db"] == snr]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])

        thresholds, far, frr = compute_roc(gen, imp)
        eer, eer_t = compute_eer(far, frr, thresholds)
        f1pct  = far_at_frr(far, frr, 0.01)
        r1pct  = frr_at_far(far, frr, 0.01)
        avg_lat = np.mean([r["latency_ms"] for r in sub])
        delta   = eer - eer_baseline

        lines += [
            f"SNR = {snr} dB",
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
    print(f"[exp4] Metrics saved to {METRICS_TXT}")

    plot_curves(results, eer_baseline, ROC_PNG, kind="roc")
    plot_curves(results, eer_baseline, DET_PNG, kind="det")


if __name__ == "__main__":
    main()
