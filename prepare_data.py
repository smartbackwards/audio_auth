"""
prepare_data.py

Splits VoxCeleb1 or VoxCeleb2 into four fully disjoint subsets:
  - train      (used for fine-tuning)
  - val        (used during fine-tuning for monitoring)
  - test       (used for experiment 1 baseline)
  - enrollment (enrolled into the speaker database)

Expected directory layouts (produced by download_voxceleb.py):

  VoxCeleb1:
    voxceleb1/
      wav/
        id10001/
          video_id/
            00001.wav

  VoxCeleb2:
    voxceleb2/
      dev/
        aac/
          id00012/
            video_id/
              00001.m4a  (or .wav)

Usage:
  python prepare_data.py --data_root ./voxceleb1 --dataset vox1
  python prepare_data.py --data_root ./voxceleb2 --dataset vox2
  python prepare_data.py --data_root ./voxceleb1 --dataset vox1 --output_dir ./data
"""

import argparse
import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Split config — per dataset ─────────────────────────────────────────────────
# VoxCeleb1 dev: ~1211 speakers total
SPLITS = {
    "vox1": {
        "n_train":        700,
        "n_val":           51,
        "n_test":         230,
        "n_enroll":       200,
        "n_enroll_utts":   10,
        "n_test_utts":     20,
    },
    # VoxCeleb2 dev: ~5994 speakers total
    "vox2": {
        "n_train":       5000,
        "n_val":          256,
        "n_test":         400,
        "n_enroll":       400,
        "n_enroll_utts":   10,
        "n_test_utts":     20,
    },
}

# Your group members' VoxCeleb speaker IDs — forced into enrollment split.
# Format: ["id1xxxx", "id1xxxx"]  Leave empty if recording yourselves separately.
GROUP_MEMBER_IDS: list[str] = []


# ── Discovery ──────────────────────────────────────────────────────────────────

def discover_speakers_vox1(root: Path) -> dict[str, list[str]]:
    """
    VoxCeleb1 layout: <root>/wav/<id>/<video>/<utt>.wav
    """
    wav_dir = root / "wav"
    if not wav_dir.exists():
        raise FileNotFoundError(
            f"Expected wav/ directory at {wav_dir}\n"
            "Run: python download_voxceleb.py --dataset vox1 --output_dir <root>"
        )
    return _scan_two_level(wav_dir, ("*.wav",))


def discover_speakers_vox2(root: Path) -> dict[str, list[str]]:
    """
    VoxCeleb2 layout: <root>/dev/aac/<id>/<video>/<utt>.[m4a|wav]
    """
    aac_dir = root / "dev" / "aac"
    if not aac_dir.exists():
        raise FileNotFoundError(
            f"Expected dev/aac/ directory at {aac_dir}\n"
            "Run: python download_voxceleb.py --dataset vox2 --output_dir <root>"
        )
    return _scan_two_level(aac_dir, ("*.m4a", "*.wav"))


def _scan_two_level(base: Path, globs: tuple) -> dict[str, list[str]]:
    """Scans <base>/<speaker>/<video>/<utt> and returns {speaker_id: [paths]}."""
    speakers: dict[str, list[str]] = defaultdict(list)
    for spk_dir in sorted(base.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        for video_dir in sorted(spk_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for pattern in globs:
                for audio_file in sorted(video_dir.glob(pattern)):
                    speakers[spk_id].append(str(audio_file))
    print(f"[prepare_data] Found {len(speakers)} speakers, "
          f"{sum(len(v) for v in speakers.values())} total utterances")
    return dict(speakers)


# ── Splitting ──────────────────────────────────────────────────────────────────

def split_speakers(speakers: dict, cfg: dict) -> dict[str, list[str]]:
    """Assigns each speaker to exactly one of: train, val, test, enroll."""
    all_ids = list(speakers.keys())
    random.shuffle(all_ids)

    forced    = [sid for sid in GROUP_MEMBER_IDS if sid in all_ids]
    remaining = [sid for sid in all_ids if sid not in forced]

    total_needed = cfg["n_train"] + cfg["n_val"] + cfg["n_test"] + cfg["n_enroll"]
    if total_needed > len(all_ids):
        raise ValueError(
            f"Need {total_needed} speakers but only {len(all_ids)} available.\n"
            f"Reduce split counts in SPLITS config."
        )

    n_enroll_extra = cfg["n_enroll"] - len(forced)
    enroll_ids = forced + remaining[:n_enroll_extra];  remaining = remaining[n_enroll_extra:]
    test_ids   = remaining[:cfg["n_test"]];            remaining = remaining[cfg["n_test"]:]
    val_ids    = remaining[:cfg["n_val"]];             remaining = remaining[cfg["n_val"]:]
    train_ids  = remaining[:cfg["n_train"]]

    print(f"[prepare_data] Speaker split — "
          f"train: {len(train_ids)}, val: {len(val_ids)}, "
          f"test: {len(test_ids)}, enroll: {len(enroll_ids)}")

    return {"train": train_ids, "val": val_ids,
            "test": test_ids,   "enroll": enroll_ids}


# ── Manifests ──────────────────────────────────────────────────────────────────

def build_subset_manifests(speakers: dict, split: dict,
                            cfg: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict]] = {}

    for subset in ("train", "val"):
        rows = [{"speaker_id": spk, "wav_path": p}
                for spk in split[subset]
                for p in speakers[spk]]
        manifests[subset] = rows
        print(f"[prepare_data] {subset}: {len(rows)} utterances")

    enroll_rows: list[dict] = []
    test_rows:   list[dict] = []

    for spk_id in split["enroll"]:
        utts = speakers[spk_id].copy()
        random.shuffle(utts)
        for wav in utts[:cfg["n_enroll_utts"]]:
            enroll_rows.append({"speaker_id": spk_id, "wav_path": wav})
        for wav in utts[cfg["n_enroll_utts"]:
                        cfg["n_enroll_utts"] + cfg["n_test_utts"]]:
            test_rows.append({"speaker_id": spk_id, "wav_path": wav})

    for spk_id in split["test"]:
        utts = speakers[spk_id].copy()
        random.shuffle(utts)
        for wav in utts[:cfg["n_test_utts"]]:
            test_rows.append({"speaker_id": spk_id, "wav_path": wav,
                              "enrolled": False})

    manifests["enroll"] = enroll_rows
    manifests["test"]   = test_rows

    print(f"[prepare_data] enroll: {len(enroll_rows)} utterances")
    print(f"[prepare_data] test:   {len(test_rows)} utterances")

    for subset, rows in manifests.items():
        if not rows:
            continue
        csv_path = output_dir / f"{subset}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[prepare_data] Written: {csv_path}")

    split_path = output_dir / "speaker_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[prepare_data] Speaker split saved: {split_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root directory of the dataset "
                             "(contains wav/ for vox1, or dev/aac/ for vox2)")
    parser.add_argument("--dataset",   default="vox1", choices=["vox1", "vox2"],
                        help="Which dataset: vox1 (default) or vox2")
    parser.add_argument("--output_dir", default="./data",
                        help="Where to write manifest CSVs")
    args = parser.parse_args()

    root       = Path(args.data_root)
    output_dir = Path(args.output_dir)
    cfg        = SPLITS[args.dataset]

    if args.dataset == "vox1":
        speakers = discover_speakers_vox1(root)
    else:
        speakers = discover_speakers_vox2(root)

    split = split_speakers(speakers, cfg)
    build_subset_manifests(speakers, split, cfg, output_dir)

    print("\n[prepare_data] Done. Next step: run finetune.py")


if __name__ == "__main__":
    main()
