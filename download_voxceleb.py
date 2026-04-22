"""
download_voxceleb.py

Downloads VoxCeleb1 or VoxCeleb2 from HuggingFace (ProgramComputer/voxceleb),
joins the multipart ZIPs, and extracts to the standard directory layout.

  VoxCeleb1 (~30 GB compressed WAV):
    <output_dir>/wav/<id>/<video>/<utt>.wav

  VoxCeleb2 AAC (~100 GB compressed):
    <output_dir>/dev/aac/<id>/<video>/<utt>.m4a

Usage:
  python download_voxceleb.py --dataset vox1 --output_dir ./voxceleb1
  python download_voxceleb.py --dataset vox2 --output_dir ./voxceleb2
  python download_voxceleb.py --dataset vox1 --output_dir ./voxceleb1 --skip_download
"""

import argparse
import sys
import zipfile
from pathlib import Path

HF_REPO = "ProgramComputer/voxceleb"

DATASET_CONFIG = {
    "vox1": {
        "zip_prefix":   "vox1_dev_wav",   # matches vox1_dev_wav_1.zip etc.
        "joined_name":  "vox1_dev_wav.zip",
        "description":  "VoxCeleb1 dev (~30 GB compressed WAV)",
    },
    "vox2": {
        "zip_prefix":   "vox2_aac",        # matches vox2_aac_1.zip, vox2_aac_2.zip
        "joined_name":  "vox2_aac.zip",
        "description":  "VoxCeleb2 dev AAC (~100 GB compressed)",
    },
}


# ── Step 1: list zip parts ─────────────────────────────────────────────────────

def list_parts(repo_id: str, prefix: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        sys.exit("huggingface_hub is not installed — run: pip install huggingface_hub")

    print(f"[download] Listing files in {repo_id} ...")
    all_files = sorted(list_repo_files(repo_id, repo_type="dataset"))
    parts = [f for f in all_files
             if Path(f).name.startswith(prefix) and f.endswith(".zip")]

    if not parts:
        print(f"  All files in repo: {all_files}")
        sys.exit(
            f"No files matching '{prefix}*.zip' found.\n"
            "Check the file listing above and update DATASET_CONFIG if needed."
        )

    print(f"[download] Found {len(parts)} part(s):")
    for p in parts:
        print(f"  {p}")
    return parts


# ── Step 2: download parts ─────────────────────────────────────────────────────

def download_parts(repo_id: str, filenames: list[str],
                   parts_dir: Path) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed — run: pip install huggingface_hub")

    parts_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []

    for filename in filenames:
        dest = parts_dir / Path(filename).name
        if dest.exists():
            print(f"[download] Already exists (skipping): {dest.name}")
            local_paths.append(dest)
            continue

        print(f"[download] Downloading {filename} ...")
        result = hf_hub_download(
            repo_id   = repo_id,
            filename  = filename,
            repo_type = "dataset",
            local_dir = str(parts_dir),
        )
        local_paths.append(Path(result))

    return local_paths


# ── Step 3: join multipart zips ────────────────────────────────────────────────

def join_parts(parts: list[Path], joined_path: Path) -> Path:
    if len(parts) == 1:
        return parts[0]

    if joined_path.exists():
        print(f"[download] Joined file already exists (skipping): {joined_path.name}")
        return joined_path

    parts = sorted(parts)
    print(f"[download] Joining {len(parts)} part(s) → {joined_path.name} ...")
    with open(joined_path, "wb") as out:
        for part in parts:
            print(f"  + {part.name}")
            with open(part, "rb") as f:
                while chunk := f.read(4 * 1024 * 1024):
                    out.write(chunk)

    size_gb = joined_path.stat().st_size / 1e9
    print(f"[download] Joined ZIP: {joined_path.name}  ({size_gb:.1f} GB)")
    return joined_path


# ── Step 4: extract ────────────────────────────────────────────────────────────

def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] Extracting {zip_path.name} → {output_dir} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total   = len(members)
        print(f"[download] {total:,} files to extract ...")
        for i, member in enumerate(members, 1):
            zf.extract(member, output_dir)
            if i % 5000 == 0 or i == total:
                print(f"  {i:,} / {total:,}", end="\r", flush=True)

    print()
    print(f"[download] Extraction complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download VoxCeleb from HuggingFace and extract."
    )
    parser.add_argument("--dataset",       default="vox1", choices=["vox1", "vox2"],
                        help="Which dataset to download: vox1 (default) or vox2")
    parser.add_argument("--output_dir",    required=True,
                        help="Where to extract audio files")
    parser.add_argument("--parts_dir",     default=None,
                        help="Where to save ZIP parts "
                             "(default: <output_dir>/_zips)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading — use existing ZIPs in parts_dir")
    parser.add_argument("--keep_zips",     action="store_true",
                        help="Keep ZIP files after extraction")
    args = parser.parse_args()

    cfg        = DATASET_CONFIG[args.dataset]
    output_dir = Path(args.output_dir)
    parts_dir  = Path(args.parts_dir) if args.parts_dir else output_dir / "_zips"
    joined_zip = parts_dir / cfg["joined_name"]

    print(f"[download] Dataset: {cfg['description']}")

    # Download
    if args.skip_download:
        parts = sorted(parts_dir.glob(f"{cfg['zip_prefix']}*.zip"))
        if not parts:
            sys.exit(f"--skip_download set but no matching ZIPs found in {parts_dir}")
        print(f"[download] Using {len(parts)} existing part(s) in {parts_dir}")
    else:
        filenames = list_parts(HF_REPO, cfg["zip_prefix"])
        parts     = download_parts(HF_REPO, filenames, parts_dir)

    # Join
    joined_zip = join_parts(parts, joined_zip)

    # Extract
    extract_zip(joined_zip, output_dir)

    # Cleanup
    if not args.keep_zips:
        print(f"[download] Cleaning up ZIPs ...")
        for p in parts_dir.glob("*.zip"):
            p.unlink()
        try:
            parts_dir.rmdir()
        except OSError:
            pass

    print(f"\n[download] Done.")
    print(f"Next step:")
    print(f"  python prepare_data.py --dataset {args.dataset} --data_root {output_dir} --output_dir ./data")


if __name__ == "__main__":
    main()
