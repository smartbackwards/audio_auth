"""
experiments/exp5_background_noise.py

Experiment 5 — Environmental background noise.

Real-world noise clips (e.g. UrbanSound8K) are mixed into speech at target
SNR levels: 20 dB, 10 dB, 0 dB.  At 0 dB the noise is as loud as the speech.

Noise mixing:
    speech_power      = mean(speech ** 2)
    current_noise_pwr = mean(noise  ** 2)
    scale             = sqrt(speech_power / (current_noise_pwr * 10^(SNR/10)))
    mixed             = speech + scale * noise

Noise clips shorter than the speech are repeat-tiled; longer ones are
randomly cropped.  Each trial draws a different random noise clip.

Outputs (results/exp5/):
  scores.csv        distance, snr_db, noise_file, genuine per trial
  metrics.txt       EER per SNR level + vs exp1 baseline
  roc.png / det.png curves per SNR level

Usage:
  python experiments/exp5_background_noise.py --noise_dir UrbanSound8K/audio
  python experiments/exp5_background_noise.py --noise_dir UrbanSound8K/audio --n_genuine 100 --n_impostor 100
  python experiments/exp5_background_noise.py --noise_dir UrbanSound8K/audio --rerun
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
import soundfile as sf
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
OUT_DIR     = Path("results/exp5")
SCORES_CSV  = OUT_DIR / "scores.csv"
METRICS_TXT = OUT_DIR / "metrics.txt"
ROC_PNG     = OUT_DIR / "roc.png"
DET_PNG     = OUT_DIR / "det.png"

SAMPLE_RATE = 16000
SNR_LEVELS  = [20, 10, 0]
COLORS      = {20: "#2ecc71", 10: "#e67e22", 0: "#e74c3c"}


# ── Noise loading ──────────────────────────────────────────────────────────────

def load_noise_files(noise_dir: str) -> list[Path]:
    """Recursively find all .wav and .ogg files under noise_dir."""
    root = Path(noise_dir)
    files = list(root.rglob("*.wav")) + list(root.rglob("*.ogg"))
    if not files:
        raise FileNotFoundError(
            f"No .wav/.ogg files found in '{noise_dir}'.\n"
            "Download UrbanSound8K from https://urbansounddataset.weebly.com/urbansound8k.html\n"
            "and pass --noise_dir UrbanSound8K/audio"
        )
    print(f"[exp5] Found {len(files)} noise files in {noise_dir}")
    return files


def load_noise_clip(noise_path: Path, target_len: int) -> torch.Tensor:
    """Load a noise file, resample to 16 kHz, mono-mix, tile/crop to target_len."""
    try:
        data, sr = sf.read(str(noise_path), dtype="float32", always_2d=True)
        noise = torch.from_numpy(data.T)  # (C, T)
    except Exception:
        return None

    if sr != SAMPLE_RATE:
        noise = torchaudio.functional.resample(noise, sr, SAMPLE_RATE)
    if noise.shape[0] > 1:
        noise = noise.mean(dim=0, keepdim=True)

    # Tile if shorter, crop if longer
    if noise.shape[1] < target_len:
        repeats = (target_len // noise.shape[1]) + 1
        noise   = noise.repeat(1, repeats)
    # Random crop
    if noise.shape[1] > target_len:
        start = random.randint(0, noise.shape[1] - target_len)
        noise = noise[:, start: start + target_len]

    return noise  # (1, target_len)


# ── Noise mixing ───────────────────────────────────────────────────────────────

def mix_noise(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Scale noise to target SNR and add to speech."""
    speech_power = (speech ** 2).mean()
    noise_power  = (noise  ** 2).mean()
    if noise_power < 1e-10 or speech_power < 1e-10:
        return speech
    scale = (speech_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
    return speech + scale * noise


# ── Trial running ──────────────────────────────────────────────────────────────

def run_trials(model, db, trials: list[dict], noise_files: list[Path]) -> list[dict]:
    results = []
    for trial in tqdm(trials):
        snr = trial["snr_db"]
        try:
            t0     = time.perf_counter()
            speech = preprocess_wav(trial["wav_path"])

            # Pick a random noise clip
            noise_path = random.choice(noise_files)
            noise = load_noise_clip(noise_path, speech.shape[1])
            if noise is None:
                noise_path = random.choice(noise_files)
                noise = load_noise_clip(noise_path, speech.shape[1])

            mixed = mix_noise(speech, noise, snr)
            emb   = get_embedding(model, mixed)
            dist  = score_against_speaker(emb, db[trial["claimed_id"]]["embeddings"])
            lat   = (time.perf_counter() - t0) * 1000

            results.append({
                "wav_path":   trial["wav_path"],
                "claimed_id": trial["claimed_id"],
                "genuine":    trial["genuine"],
                "snr_db":     snr,
                "noise_file": noise_path.name,
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
        label  = f"SNR={snr} dB  EER={eer:.2%}"

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
        eer_nd = stats.norm.ppf(max(eer_baseline, eps))
        ax.scatter([eer_nd], [eer_nd], color="grey", zorder=5,
                   label=f"Baseline EER={eer_baseline:.2%}")
        ax.plot([tick_vals[0], tick_vals[-1]], [tick_vals[0], tick_vals[-1]], "k--", lw=0.8)
        ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls)
        ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls)
        ax.set_xlabel("FAR (%)")
        ax.set_ylabel("FRR (%)")

    curve = "ROC" if kind == "roc" else "DET"
    ax.set_title(f"{curve} — Experiment 5: Background Noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp5] {curve} saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_dir",   required=True,
                        help="Root folder of noise dataset (e.g. UrbanSound8K/audio)")
    parser.add_argument("--genuine_csv", default=GENUINE_CSV)
    parser.add_argument("--imp_csv",     default=IMP_CSV)
    parser.add_argument("--db_path",     default="speaker_db.json")
    parser.add_argument("--n_genuine",   type=int, default=None,
                        help="Genuine trials per SNR level (default: all / n_levels)")
    parser.add_argument("--n_impostor",  type=int, default=None)
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

    print(f"[exp5] {len(trials)} total trials  "
          f"({n_gen} genuine + {n_imp} impostor) × {len(SNR_LEVELS)} SNR levels")

    # ── Run or load ───────────────────────────────────────────────────────────
    if SCORES_CSV.exists() and not args.rerun:
        print(f"[exp5] Loading existing scores from {SCORES_CSV}")
        results = []
        with open(SCORES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                results.append({
                    "wav_path":   row["wav_path"],
                    "claimed_id": row["claimed_id"],
                    "genuine":    row["genuine"].lower() == "true",
                    "snr_db":     int(row["snr_db"]),
                    "noise_file": row["noise_file"],
                    "distance":   float(row["distance"]),
                    "latency_ms": float(row["latency_ms"]),
                })
    else:
        noise_files = load_noise_files(args.noise_dir)
        model       = load_model()
        db          = load_db(args.db_path)
        results     = run_trials(model, db, trials, noise_files)

        with open(SCORES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "claimed_id", "genuine",
                                                    "snr_db", "noise_file",
                                                    "distance", "latency_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[exp5] Scores saved to {SCORES_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lines = ["Experiment 5 - Environmental Background Noise",
             "=" * 50,
             f"Baseline EER (Exp 1): {eer_baseline:.4f} ({eer_baseline*100:.2f}%)",
             ""]

    for snr in SNR_LEVELS:
        sub = [r for r in results if r["snr_db"] == snr]
        gen = np.array([r["distance"] for r in sub if     r["genuine"]])
        imp = np.array([r["distance"] for r in sub if not r["genuine"]])

        thresholds, far, frr = compute_roc(gen, imp)
        eer, eer_t = compute_eer(far, frr, thresholds)
        f1pct   = far_at_frr(far, frr, 0.01)
        r1pct   = frr_at_far(far, frr, 0.01)
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
    print(f"[exp5] Metrics saved to {METRICS_TXT}")

    plot_curves(results, eer_baseline, ROC_PNG, kind="roc")
    plot_curves(results, eer_baseline, DET_PNG, kind="det")


if __name__ == "__main__":
    main()
