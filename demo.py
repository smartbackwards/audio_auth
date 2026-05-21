"""
demo.py

Live speaker verification / identification demo.

Modes:
  identify  (default) — speaks into mic, system says who it is
  verify               — claim a speaker ID first, then speak; accept/reject

Usage:
  python demo.py
  python demo.py --mode verify
  python demo.py --mode verify --speaker bartek_m
  python demo.py --duration 5 --threshold 0.45
"""

import os
import argparse
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf

from enroll import load_model, load_db, compute_speaker_centroid
from auth import verify, identify, DEFAULT_THRESHOLD

SAMPLE_RATE = 16000


def record(duration: float, device=None) -> np.ndarray:
    print(f"  Recording {duration:.0f}s ... ", end="", flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.abs(audio).max())
    print(f"done.  rms={rms:.4f}  peak={peak:.4f}")
    if rms < 1e-4:
        print("  [warn] Audio is near-silent — check microphone access / privacy settings.")
    return audio.squeeze()


def run_once(model, db, centroids, mode: str, speaker: str | None,
             duration: float, threshold: float, device=None) -> None:
    enrolled = sorted(db.keys())

    if mode == "verify":
        if speaker:
            claimed = speaker
        else:
            print(f"Enrolled: {', '.join(enrolled)}")
            claimed = input("  Claim speaker ID (or 'q' to quit): ").strip()
            if claimed.lower() == "q":
                raise KeyboardInterrupt
            if claimed not in db:
                print(f"  Unknown speaker '{claimed}'.")
                return
        input(f"  Press Enter then speak as '{db[claimed]['display_name']}' ...")
    else:
        input("  Press Enter then speak ...")

    audio = record(duration, device=device)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        if mode == "verify":
            result = verify(model, db, claimed, tmp_path, threshold)
            verdict = "ACCEPTED" if result["accepted"] else "REJECTED"
            marker  = "✓" if result["accepted"] else "✗"
            name    = db[claimed]["display_name"]
            print(f"\n  {marker}  {verdict}  —  claimed: {name}")
            print(f"     distance={result['distance']:.4f}  "
                  f"threshold={threshold:.4f}  "
                  f"latency={result['latency_ms']:.0f} ms")
        else:
            result = identify(model, db, tmp_path, threshold, centroids)
            if result["accepted"]:
                name = result["display_name"]
                print(f"\n  ✓  IDENTIFIED  —  {name}  ({result['speaker_id']})")
                print(f"     distance={result['distance']:.4f}  "
                      f"threshold={threshold:.4f}  "
                      f"latency={result['latency_ms']:.0f} ms")
                if result.get("scores"):
                    print("     Top candidates:")
                    for spk_id, dist in result["scores"][:4]:
                        tag  = " <<" if spk_id == result["speaker_id"] else ""
                        dname = db[spk_id]["display_name"]
                        print(f"       {dname:22s}  {dist:.4f}{tag}")
            else:
                print(f"\n  ✗  UNKNOWN SPEAKER")
                print(f"     best distance={result['distance']:.4f}  "
                      f"threshold={threshold:.4f}  "
                      f"latency={result['latency_ms']:.0f} ms")
                if result.get("scores"):
                    best_id, best_dist = result["scores"][0]
                    print(f"     Closest match: {db[best_id]['display_name']} "
                          f"(dist={best_dist:.4f})")
    finally:
        os.unlink(tmp_path)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         choices=["identify", "verify"], default="identify")
    parser.add_argument("--speaker",      default=None,
                        help="Fix the claimed speaker ID (verify mode only)")
    parser.add_argument("--duration",     type=float, default=4.0,
                        help="Recording duration in seconds (default 4)")
    parser.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--db_path",      default="speaker_db.json")
    parser.add_argument("--device",       type=int, default=None,
                        help="Input device index (see --list_devices)")
    parser.add_argument("--list_devices", action="store_true",
                        help="List audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print("[demo] Loading model and speaker database ...")
    model = load_model()
    db    = load_db(args.db_path)

    if not db:
        print("[demo] Speaker database is empty — run enroll.py first.")
        return

    centroids = compute_speaker_centroid(db)
    enrolled  = sorted(db.keys())

    # Show which mic will be used
    default_in = sd.query_devices(kind='input')
    print(f"[demo] Microphone: {default_in['name']} (use --device N to change, --list_devices to list)")
    print(f"[demo] {len(enrolled)} enrolled speakers")
    print(f"[demo] Mode: {args.mode}  |  Duration: {args.duration}s  |  "
          f"Threshold: {args.threshold:.3f}")
    print("[demo] Press Ctrl+C to exit.\n")

    while True:
        try:
            run_once(model, db, centroids, args.mode, args.speaker,
                     args.duration, args.threshold, args.device)
        except KeyboardInterrupt:
            print("\n[demo] Bye.")
            break


if __name__ == "__main__":
    main()
