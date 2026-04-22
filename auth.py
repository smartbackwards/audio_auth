"""
auth.py

Speaker verification (1:1) and identification (1:N).

Verification:  given a speaker ID + audio clip → accept or reject
Identification: given only an audio clip → return best matching speaker

Also includes a batch testing mode used by the experiment scripts.

Usage:
  # Verify a specific speaker
  python auth.py verify --speaker_id id10001 --audio_path sample.wav

  # Identify unknown speaker
  python auth.py identify --audio_path sample.wav

  # Batch test (used by test scripts)
  python auth.py batch_test --manifest data/test.csv --output results/baseline.csv
"""

import json
import time
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from enroll import load_model, load_db, preprocess_wav, get_embedding, compute_speaker_centroid

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.25   # cosine distance threshold (tune via ROC analysis)
DB_PATH           = "speaker_db.json"


# ── Core similarity ────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Returns cosine distance in [0, 2]. Lower = more similar."""
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def score_against_speaker(query_emb: np.ndarray,
                           speaker_embeddings: list[np.ndarray]) -> float:
    """
    Returns the minimum cosine distance between the query and all
    stored embeddings for a speaker (nearest-neighbour strategy).
    Alternatively use mean distance — both are valid, NN is slightly more robust.
    """
    distances = [cosine_distance(query_emb, e) for e in speaker_embeddings]
    return min(distances)


# ── Verification (1:1) ─────────────────────────────────────────────────────────

def verify(model, db: dict, speaker_id: str, audio_path: str,
           threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Verifies whether the given audio belongs to speaker_id.

    Returns:
      {
        "accepted":  bool,
        "distance":  float,   # cosine distance (lower = more similar)
        "threshold": float,
        "latency_ms": float,
      }
    """
    t0 = time.perf_counter()

    if speaker_id not in db:
        raise ValueError(f"Speaker '{speaker_id}' not found in database.")

    wav = preprocess_wav(audio_path)
    emb = get_embedding(model, wav)

    distance = score_against_speaker(emb, db[speaker_id]["embeddings"])
    accepted  = distance <= threshold

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "accepted":   accepted,
        "distance":   distance,
        "threshold":  threshold,
        "latency_ms": latency_ms,
    }


# ── Identification (1:N) ───────────────────────────────────────────────────────

def identify(model, db: dict, audio_path: str,
             threshold: float = DEFAULT_THRESHOLD,
             centroids: dict = None) -> dict:
    """
    Identifies who the speaker is (1:N search).

    Two-stage:
      1. Fast centroid comparison to rank speakers
      2. Full NN comparison against top-5 candidates

    Returns:
      {
        "speaker_id":   str or None (None if no speaker clears threshold),
        "display_name": str or None,
        "distance":     float,
        "latency_ms":   float,
        "scores":       list of (speaker_id, distance) sorted ascending
      }
    """
    t0 = time.perf_counter()

    wav = preprocess_wav(audio_path)
    emb = get_embedding(model, wav)

    if centroids:
        # Fast first pass with centroids
        centroid_scores = {
            spk: cosine_distance(emb, c)
            for spk, c in centroids.items()
        }
        top_candidates = sorted(centroid_scores, key=centroid_scores.get)[:10]
    else:
        top_candidates = list(db.keys())

    # Full NN on top candidates
    scores = []
    for spk_id in top_candidates:
        dist = score_against_speaker(emb, db[spk_id]["embeddings"])
        scores.append((spk_id, dist))

    scores.sort(key=lambda x: x[1])
    best_id, best_dist = scores[0]

    latency_ms = (time.perf_counter() - t0) * 1000

    if best_dist <= threshold:
        return {
            "speaker_id":   best_id,
            "display_name": db[best_id]["display_name"],
            "distance":     best_dist,
            "accepted":     True,
            "latency_ms":   latency_ms,
            "scores":       scores,
        }
    else:
        return {
            "speaker_id":   None,
            "display_name": None,
            "distance":     best_dist,
            "accepted":     False,
            "latency_ms":   latency_ms,
            "scores":       scores,
        }


# ── Batch testing ──────────────────────────────────────────────────────────────

def batch_test(model, db: dict, manifest_path: str, output_path: str,
               threshold: float = DEFAULT_THRESHOLD,
               mode: str = "verify") -> None:
    """
    Runs verification or identification on all samples in a manifest CSV.
    Writes results to output_path CSV.

    manifest CSV columns: speaker_id, wav_path
      - For genuine trials: speaker_id matches the true speaker
      - For impostor trials: speaker_id is the claimed identity (different from true speaker)
        mark these with an additional 'genuine' column (True/False) if pre-labelled

    Output CSV: speaker_id, wav_path, true_label, distance, accepted, latency_ms
    """
    import csv

    centroids = compute_speaker_centroid(db) if mode == "identify" else None

    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[auth] Batch {mode} on {len(rows)} samples...")

    results = []
    for row in tqdm(rows):
        wav_path   = row["wav_path"]
        speaker_id = row["speaker_id"]
        genuine    = row.get("genuine", "True").lower() != "false"

        try:
            if mode == "verify":
                result = verify(model, db, speaker_id, wav_path, threshold)
                result["true_speaker"] = speaker_id if genuine else "impostor"
                result["genuine"]      = genuine
                result["wav_path"]     = wav_path
                result["claimed_id"]   = speaker_id
            else:
                result = identify(model, db, wav_path, threshold, centroids)
                result["true_speaker"] = speaker_id
                result["genuine"]      = genuine
                result["wav_path"]     = wav_path
                result["claimed_id"]   = speaker_id
            results.append(result)
        except Exception as e:
            print(f"  [warn] Skipping {wav_path}: {e}")

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["wav_path", "claimed_id", "true_speaker", "genuine",
                  "accepted", "distance", "latency_ms"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"[auth] Results written to {output_path}")

    # Quick summary
    tp = sum(1 for r in results if r["genuine"] and r["accepted"])
    fn = sum(1 for r in results if r["genuine"] and not r["accepted"])
    fp = sum(1 for r in results if not r["genuine"] and r["accepted"])
    tn = sum(1 for r in results if not r["genuine"] and not r["accepted"])

    far = fp / (fp + tn + 1e-8)
    frr = fn / (fn + tp + 1e-8)
    acc = (tp + tn) / len(results)

    print(f"\n── Quick summary (threshold={threshold:.3f}) ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  FAR      : {far:.4f}  (False Accept Rate)")
    print(f"  FRR      : {frr:.4f}  (False Reject Rate)")
    print(f"  Avg latency: {np.mean([r['latency_ms'] for r in results]):.1f} ms")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Verify
    v = subparsers.add_parser("verify")
    v.add_argument("--speaker_id",  required=True)
    v.add_argument("--audio_path",  required=True)
    v.add_argument("--model_dir",   default=None)
    v.add_argument("--db_path",     default=DB_PATH)
    v.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)

    # Identify
    i = subparsers.add_parser("identify")
    i.add_argument("--audio_path",  required=True)
    i.add_argument("--model_dir",   default=None)
    i.add_argument("--db_path",     default=DB_PATH)
    i.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)

    # Batch test
    b = subparsers.add_parser("batch_test")
    b.add_argument("--manifest",    required=True)
    b.add_argument("--output",      required=True)
    b.add_argument("--model_dir",   default=None)
    b.add_argument("--db_path",     default=DB_PATH)
    b.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)
    b.add_argument("--mode",        default="verify", choices=["verify", "identify"])

    args = parser.parse_args()

    model = load_model(args.model_dir)
    db    = load_db(args.db_path)

    if args.command == "verify":
        result = verify(model, db, args.speaker_id, args.audio_path, args.threshold)
        verdict = "✅ ACCEPTED" if result["accepted"] else "❌ REJECTED"
        print(f"\n{verdict}  |  distance={result['distance']:.4f}  |  "
              f"threshold={result['threshold']:.4f}  |  latency={result['latency_ms']:.1f}ms")

    elif args.command == "identify":
        result = identify(model, db, args.audio_path, args.threshold)
        if result["accepted"]:
            print(f"\n✅ Identified: {result['display_name']} ({result['speaker_id']})  "
                  f"|  distance={result['distance']:.4f}  |  latency={result['latency_ms']:.1f}ms")
        else:
            print(f"\n❓ Unknown speaker  |  best distance={result['distance']:.4f}  "
                  f"|  latency={result['latency_ms']:.1f}ms")

    elif args.command == "batch_test":
        batch_test(model, db, args.manifest, args.output, args.threshold, args.mode)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
