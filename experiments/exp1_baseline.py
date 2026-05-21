"""
experiments/exp1_baseline.py

Experiment 1 — Baseline system evaluation.

Genuine scores  : loaded from the existing results/exp1_baseline.csv
                  (already produced by `auth.py batch_test`)
Impostor scores : computed here — each test-only clip is verified against
                  a randomly assigned enrolled speaker.

Outputs (all written to results/exp1/):
  impostor_scores.csv   raw distances for impostor trials
  metrics.txt           EER, FAR@FRR=1%, FRR@FAR=1%, avg latency
  roc.png               ROC curve
  det.png               DET curve

Usage:
  python experiments/exp1_baseline.py
  python experiments/exp1_baseline.py --rerun_impostors   # recompute even if CSV exists
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import csv
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from enroll import load_model, load_db, preprocess_wav, get_embedding
from auth import cosine_distance, score_against_speaker, DEFAULT_THRESHOLD


# ── Paths ──────────────────────────────────────────────────────────────────────

GENUINE_CSV   = "results/exp1_baseline.csv"
TEST_CSV      = "data/test.csv"
OUT_DIR       = Path("results/exp1")
IMP_CSV       = OUT_DIR / "impostor_scores.csv"
METRICS_TXT   = OUT_DIR / "metrics.txt"
ROC_PNG       = OUT_DIR / "roc.png"
DET_PNG       = OUT_DIR / "det.png"


# ── Score loading ──────────────────────────────────────────────────────────────

def load_genuine_scores(csv_path: str) -> list[tuple[float, float]]:
    """Returns list of (distance, latency_ms) for genuine trials."""
    scores = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["genuine"].lower() == "true":
                scores.append((float(row["distance"]), float(row["latency_ms"])))
    print(f"[exp1] Loaded {len(scores)} genuine scores from {csv_path}")
    return scores


def compute_impostor_scores(model, db: dict, test_csv: str,
                            out_csv: Path, seed: int = 42) -> list[tuple[float, float]]:
    """
    For each impostor row in test_csv (genuine=False, speaker not in DB),
    assigns a random enrolled speaker as the claimed identity and runs verify.
    Writes results to out_csv and returns list of (distance, latency_ms).
    """
    import time

    rng = random.Random(seed)
    enrolled = list(db.keys())

    rows = []
    with open(test_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["genuine"].lower() == "false":
                rows.append(row)

    print(f"[exp1] Computing {len(rows)} impostor scores...")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scores = []

    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["wav_path", "true_speaker",
                                                   "claimed_id", "distance", "latency_ms"])
        writer.writeheader()

        for row in tqdm(rows):
            wav_path   = row["wav_path"]
            true_spk   = row["speaker_id"]
            claimed_id = rng.choice(enrolled)

            try:
                t0  = time.perf_counter()
                wav = preprocess_wav(wav_path)
                emb = get_embedding(model, wav)
                dist = score_against_speaker(emb, db[claimed_id]["embeddings"])
                lat  = (time.perf_counter() - t0) * 1000

                writer.writerow({"wav_path": wav_path, "true_speaker": true_spk,
                                  "claimed_id": claimed_id, "distance": dist,
                                  "latency_ms": lat})
                scores.append((dist, lat))
            except Exception as e:
                print(f"  [warn] Skipping {wav_path}: {e}")

    print(f"[exp1] Impostor scores saved to {out_csv}")
    return scores


def load_impostor_scores(csv_path: Path) -> list[tuple[float, float]]:
    scores = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            scores.append((float(row["distance"]), float(row["latency_ms"])))
    print(f"[exp1] Loaded {len(scores)} impostor scores from {csv_path}")
    return scores


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_roc(genuine_dists, impostor_dists, n_thresholds=1000):
    """
    Sweeps thresholds and returns arrays of (thresholds, FAR, FRR).
    FAR = P(accept | impostor),  FRR = P(reject | genuine)
    """
    all_dists = np.concatenate([genuine_dists, impostor_dists])
    thresholds = np.linspace(all_dists.min(), all_dists.max(), n_thresholds)

    far = np.array([np.mean(impostor_dists <= t) for t in thresholds])
    frr = np.array([np.mean(genuine_dists  >  t) for t in thresholds])

    return thresholds, far, frr


def compute_eer(far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray):
    """Interpolates EER where FAR = FRR. Returns (eer, eer_threshold)."""
    diff = far - frr
    idx  = np.argmin(np.abs(diff))
    # linear interpolation between the two bracketing points
    if idx > 0 and diff[idx - 1] * diff[idx] < 0:
        t = np.interp(0, [diff[idx - 1], diff[idx]],
                         [thresholds[idx - 1], thresholds[idx]])
        eer = np.interp(t, thresholds, (far + frr) / 2)
    else:
        t   = thresholds[idx]
        eer = (far[idx] + frr[idx]) / 2
    return float(eer), float(t)


def far_at_frr(far, frr, target_frr=0.01):
    """FAR when FRR is closest to target_frr."""
    idx = np.argmin(np.abs(frr - target_frr))
    return float(far[idx])


def frr_at_far(far, frr, target_far=0.01):
    """FRR when FAR is closest to target_far."""
    idx = np.argmin(np.abs(far - target_far))
    return float(frr[idx])


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_roc(far, frr, eer, out_path: Path):
    tpr = 1 - frr
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(far, tpr, lw=2, label="ECAPA-TDNN (pretrained)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER = {eer:.2%}")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate = 1 − FRR)")
    ax.set_title("ROC Curve — Experiment 1 Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp1] ROC saved to {out_path}")


def plot_det(far, frr, eer, out_path: Path):
    """DET curve: both axes on normal-deviate scale (standard in speaker recognition)."""
    eps = 1e-6
    far_c = np.clip(far, eps, 1 - eps)
    frr_c = np.clip(frr, eps, 1 - eps)

    far_nd = stats.norm.ppf(far_c)
    frr_nd = stats.norm.ppf(frr_c)

    ticks     = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    tick_vals = stats.norm.ppf(ticks)
    tick_lbls = [f"{t*100:.1f}" for t in ticks]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(far_nd, frr_nd, lw=2, label="ECAPA-TDNN (pretrained)")
    eer_nd = stats.norm.ppf(eer)
    ax.scatter([eer_nd], [eer_nd], color="red", zorder=5, label=f"EER = {eer:.2%}")
    ax.plot([far_nd.min(), far_nd.max()],
            [far_nd.min(), far_nd.max()], "k--", lw=0.8)
    ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls)
    ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curve — Experiment 1 Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp1] DET saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine_csv",     default=GENUINE_CSV)
    parser.add_argument("--test_csv",        default=TEST_CSV)
    parser.add_argument("--db_path",         default="speaker_db.json")
    parser.add_argument("--threshold",       type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--rerun_impostors", action="store_true",
                        help="Recompute impostor scores even if CSV exists")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load or compute scores ────────────────────────────────────────────────
    genuine_scores  = load_genuine_scores(args.genuine_csv)
    genuine_dists   = np.array([s[0] for s in genuine_scores])
    genuine_lats    = np.array([s[1] for s in genuine_scores])

    if IMP_CSV.exists() and not args.rerun_impostors:
        impostor_scores = load_impostor_scores(IMP_CSV)
    else:
        model           = load_model()
        db              = load_db(args.db_path)
        impostor_scores = compute_impostor_scores(model, db, args.test_csv, IMP_CSV)

    impostor_dists = np.array([s[0] for s in impostor_scores])
    impostor_lats  = np.array([s[1] for s in impostor_scores])

    # ── Metrics ───────────────────────────────────────────────────────────────
    thresholds, far, frr = compute_roc(genuine_dists, impostor_dists)
    eer, eer_threshold   = compute_eer(far, frr, thresholds)

    far_at_op  = np.mean(impostor_dists <= args.threshold)
    frr_at_op  = np.mean(genuine_dists  >  args.threshold)
    acc_at_op  = 1 - (far_at_op * len(impostor_dists) +
                      frr_at_op * len(genuine_dists)) / (len(genuine_dists) + len(impostor_dists))

    far_1pct = far_at_frr(far, frr, target_frr=0.01)
    frr_1pct = frr_at_far(far, frr, target_far=0.01)

    avg_lat = np.mean(np.concatenate([genuine_lats, impostor_lats]))

    summary = f"""Experiment 1 — Baseline
========================
Genuine trials  : {len(genuine_dists)}
Impostor trials : {len(impostor_dists)}

EER             : {eer:.4f}  ({eer*100:.2f}%)  @ threshold {eer_threshold:.4f}

Operating point (threshold={args.threshold:.3f}):
  FAR           : {far_at_op:.4f}  ({far_at_op*100:.2f}%)
  FRR           : {frr_at_op:.4f}  ({frr_at_op*100:.2f}%)
  Accuracy      : {acc_at_op:.4f}  ({acc_at_op*100:.2f}%)

FAR @ FRR=1%    : {far_1pct:.4f}  ({far_1pct*100:.2f}%)
FRR @ FAR=1%    : {frr_1pct:.4f}  ({frr_1pct*100:.2f}%)

Avg latency     : {avg_lat:.1f} ms
"""
    print("\n" + summary)

    with open(METRICS_TXT, "w") as f:
        f.write(summary)
    print(f"[exp1] Metrics saved to {METRICS_TXT}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc(far, frr, eer, ROC_PNG)
    plot_det(far, frr, eer, DET_PNG)


if __name__ == "__main__":
    main()
