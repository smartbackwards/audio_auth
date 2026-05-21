"""
enroll.py

Builds the speaker database from the enrollment manifest.
Stores ONLY embeddings — no raw audio is kept (required by project spec).

Also handles enrolling your own voice recordings (the three group members).

Usage:
  # Enroll from VoxCeleb manifest
  python enroll.py --manifest data/enroll.csv --model_dir results/ecapa_finetuned/save

  # Enroll a single person from custom recordings (e.g. your own voice)
  python enroll.py --enroll_person "Bartek_T" --audio_dir my_recordings/bartek_t/ --model_dir results/ecapa_finetuned/save
"""

import os
import csv
import json
import argparse
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CHUNK_SEC    = 3.0           # seconds to use per utterance for embedding
DB_PATH      = "speaker_db.json"   # embedding database


# ── Model loading ──────────────────────────────────────────────────────────────

FINETUNED_PT = None


def load_model(model_dir: str = FINETUNED_PT):
    """
    Loads the ECAPA-TDNN encoder.

    --model_dir can be:
      - A .pt file produced by finetune.py (e.g. results/ecapa_finetuned/finetuned_encoder.pt)
        → loads pretrained model then overrides encoder weights with the fine-tuned state dict
      - Omitted / not found
        → uses the pretrained HuggingFace checkpoint as-is
    """
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    model = EncoderClassifier.from_hparams(
        source         = "speechbrain/spkrec-ecapa-voxceleb",
        savedir        = "pretrained_models/ecapa",
        run_opts       = {"device": "cpu"},
        local_strategy = LocalStrategy.COPY,
    )

    if model_dir and Path(model_dir).exists():
        if model_dir.endswith(".pt"):
            print(f"[enroll] Loading fine-tuned weights from {model_dir}")
            state = torch.load(model_dir, map_location="cpu")
            model.mods.embedding_model.load_state_dict(state)
        else:
            print(f"[enroll] --model_dir '{model_dir}' is not a .pt file — using pretrained weights.")
    else:
        print("[enroll] No fine-tuned model — using pretrained checkpoint.")

    model.eval()
    return model


# ── Audio preprocessing ────────────────────────────────────────────────────────

def preprocess_wav(wav_path: str, sample_rate: int = SAMPLE_RATE,
                   chunk_sec: float = CHUNK_SEC) -> torch.Tensor:
    """
    Loads, resamples, mono-mixes and centre-crops a wav file.
    Returns a (1, T) tensor ready for the encoder.
    """
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (channels, T)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    chunk_samples = int(chunk_sec * sample_rate)

    if wav.shape[1] >= chunk_samples:
        # Centre crop
        start = (wav.shape[1] - chunk_samples) // 2
        wav   = wav[:, start: start + chunk_samples]
    else:
        # Repeat-pad
        repeats = (chunk_samples // wav.shape[1]) + 1
        wav     = wav.repeat(1, repeats)[:, :chunk_samples]

    return wav  # (1, T)


# ── Embedding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_embedding(model, wav: torch.Tensor) -> np.ndarray:
    """Returns a normalised L2 embedding vector."""
    emb = model.encode_batch(wav)          # (1, 1, emb_dim)
    emb = emb.squeeze()                    # (emb_dim,)
    emb = emb / emb.norm()                 # L2 normalise
    return emb.cpu().numpy()


# ── Database ───────────────────────────────────────────────────────────────────

def load_db(db_path: str = DB_PATH) -> dict:
    if Path(db_path).exists():
        with open(db_path) as f:
            db = json.load(f)
        # Convert lists back to numpy arrays
        for spk in db:
            db[spk]["embeddings"] = [np.array(e) for e in db[spk]["embeddings"]]
        print(f"[enroll] Loaded existing DB with {len(db)} speakers from {db_path}")
        return db
    return {}


def save_db(db: dict, db_path: str = DB_PATH):
    # Convert numpy arrays to lists for JSON serialisation
    serialisable = {}
    for spk, data in db.items():
        serialisable[spk] = {
            "display_name": data["display_name"],
            "embeddings":   [e.tolist() for e in data["embeddings"]],
        }
    with open(db_path, "w") as f:
        json.dump(serialisable, f)
    print(f"[enroll] DB saved: {len(db)} speakers -> {db_path}")


def add_speaker(db: dict, speaker_id: str, display_name: str,
                embeddings: list[np.ndarray]) -> dict:
    """Adds or updates a speaker entry in the database."""
    if speaker_id in db:
        print(f"[enroll] Updating existing speaker: {speaker_id}")
        db[speaker_id]["embeddings"].extend(embeddings)
    else:
        db[speaker_id] = {
            "display_name": display_name,
            "embeddings":   embeddings,
        }
    return db


def compute_speaker_centroid(db: dict) -> dict:
    """
    Pre-computes a mean embedding (centroid) per speaker for fast 1:N lookup.
    Returns {speaker_id: centroid_array}.
    """
    centroids = {}
    for spk_id, data in db.items():
        embs = np.stack(data["embeddings"])     # (N, emb_dim)
        centroid = embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)    # re-normalise
        centroids[spk_id] = centroid
    return centroids


# ── Enrollment from manifest CSV ───────────────────────────────────────────────

def enroll_from_manifest(model, db: dict, manifest_path: str) -> dict:
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by speaker
    from collections import defaultdict
    spk_utts = defaultdict(list)
    for row in rows:
        spk_utts[row["speaker_id"]].append(row["wav_path"])

    print(f"[enroll] Enrolling {len(spk_utts)} speakers from {manifest_path}")

    for spk_id, wav_paths in tqdm(spk_utts.items()):
        embeddings = []
        for wav_path in wav_paths:
            try:
                wav = preprocess_wav(wav_path)
                emb = get_embedding(model, wav)
                embeddings.append(emb)
            except Exception as e:
                print(f"  [warn] Skipping {wav_path}: {e}")

        if embeddings:
            db = add_speaker(db, spk_id, display_name=spk_id, embeddings=embeddings)

    return db


# ── Enrollment from custom recordings ─────────────────────────────────────────

def enroll_person(model, db: dict, speaker_id: str, display_name: str,
                  audio_dir: str) -> dict:
    """
    Enroll a single person from a folder of wav/mp3 files.
    Use this for the three group members' self-recorded samples.

    Recording tips:
      - Record 10+ utterances of ~5 seconds each
      - Vary your sentences (don't repeat the same phrase)
      - Try to match conditions: 16kHz, mono, similar to VoxCeleb style
      - You can use Audacity to record & export as WAV 16kHz mono
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    if not audio_files:
        print(f"[enroll] No audio files found in {audio_dir}")
        return db

    print(f"[enroll] Enrolling {display_name} from {len(audio_files)} files in {audio_dir}")

    embeddings = []
    for audio_path in tqdm(sorted(audio_files)):
        try:
            wav = preprocess_wav(str(audio_path))
            emb = get_embedding(model, wav)
            embeddings.append(emb)
        except Exception as e:
            print(f"  [warn] Skipping {audio_path.name}: {e}")

    if embeddings:
        db = add_speaker(db, speaker_id, display_name=display_name, embeddings=embeddings)
        print(f"[enroll] Added {display_name}: {len(embeddings)} embeddings stored")
    else:
        print(f"[enroll] No valid embeddings for {display_name} — skipping")

    return db


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",     default=None,
                        help="CSV manifest for bulk enrollment (enroll.csv)")
    parser.add_argument("--model_dir",    default=None,
                        help="Path to fine-tuned model save dir")
    parser.add_argument("--db_path",      default=DB_PATH)
    # Single-person enrollment
    parser.add_argument("--enroll_person",default=None,
                        help="Speaker ID for single-person enrollment")
    parser.add_argument("--display_name", default=None,
                        help="Display name (e.g. 'Bartek T.')")
    parser.add_argument("--audio_dir",    default=None,
                        help="Directory of wav files for single-person enrollment")
    args = parser.parse_args()

    model = load_model(args.model_dir)
    db    = load_db(args.db_path)

    if args.manifest:
        db = enroll_from_manifest(model, db, args.manifest)

    if args.enroll_person:
        if not args.audio_dir:
            raise ValueError("--audio_dir required for single-person enrollment")
        display = args.display_name or args.enroll_person
        db = enroll_person(model, db, args.enroll_person, display, args.audio_dir)

    # Pre-compute and store centroids
    print("[enroll] Computing speaker centroids...")
    centroids = compute_speaker_centroid(db)
    centroid_path = args.db_path.replace(".json", "_centroids.npy")
    np.save(centroid_path, centroids)
    print(f"[enroll] Centroids saved: {centroid_path}")

    save_db(db, args.db_path)

    print(f"\n[enroll] Done. {len(db)} speakers in database.")
    print("Next step: run auth.py to verify or identify speakers.")


if __name__ == "__main__":
    main()
