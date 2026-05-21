"""
prepare_clean_data.py

Rebuilds data/enroll.csv, data/val.csv, and data/test.csv using ONLY the 35
locally-available VoxCeleb1 official test-set speakers.  These 40 speakers were
held out from the pretrained ECAPA-TDNN training corpus, so using them here
eliminates the data-leakage concern that affected the old 200-speaker splits
(which came from VoxCeleb1 training speakers).

Per-speaker split:
  enroll : first ENROLL_N utterances
  val    : next VAL_N utterances
  test   : remaining utterances, capped at TEST_MAX

test.csv rows:
  genuine=True  — speaker_id is the true speaker  (one row per test utterance)
  genuine=False — speaker_id is a random OTHER enrolled speaker (same audio)

Usage:
  python prepare_clean_data.py
  python prepare_clean_data.py --enroll_n 8 --test_max 30
"""

import os
import csv
import random
import argparse
from pathlib import Path

# 35 of the 40 official VoxCeleb1-O test speakers present locally
VOXCELEB1_TEST_SPEAKERS = [
    'id10329', 'id10400', 'id10456', 'id10476', 'id10508', 'id10530',
    'id10592', 'id10600', 'id10645', 'id10736', 'id10745', 'id10756',
    'id10762', 'id10786', 'id10797', 'id10802', 'id10824', 'id10862',
    'id10875', 'id10900', 'id10912', 'id10921', 'id10946', 'id10950',
    'id10994', 'id11009', 'id11045', 'id11048', 'id11059', 'id11073',
    'id11083', 'id11112', 'id11142', 'id11175', 'id11251',
]

BASE = Path('voxceleb1/wav')


def gather_wavs(speaker: str) -> list[str]:
    spk_dir = BASE / speaker
    wavs = []
    for root, _, files in os.walk(spk_dir):
        wavs += [os.path.join(root, f) for f in files if f.endswith('.wav')]
    wavs.sort()
    return wavs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enroll_n',  type=int, default=6,  help='Enrollment utterances per speaker')
    parser.add_argument('--val_n',     type=int, default=2,  help='Validation utterances per speaker')
    parser.add_argument('--test_max',  type=int, default=20, help='Max test utterances per speaker')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Verify speakers and collect splits
    enroll_rows = []
    val_rows    = []
    test_genuine = {}  # speaker_id -> [wav_paths]
    skipped = []

    for spk in VOXCELEB1_TEST_SPEAKERS:
        wavs = gather_wavs(spk)
        need = args.enroll_n + args.val_n + 1
        if len(wavs) < need:
            print(f'[warn] {spk}: only {len(wavs)} wavs (need {need}), skipping')
            skipped.append(spk)
            continue

        enroll_wavs = wavs[:args.enroll_n]
        val_wavs    = wavs[args.enroll_n:args.enroll_n + args.val_n]
        test_wavs   = wavs[args.enroll_n + args.val_n:][:args.test_max]

        for w in enroll_wavs:
            enroll_rows.append({'speaker_id': spk, 'wav_path': w})
        for w in val_wavs:
            val_rows.append({'speaker_id': spk, 'wav_path': w, 'genuine': True})
        test_genuine[spk] = test_wavs

    speakers = [s for s in VOXCELEB1_TEST_SPEAKERS if s not in skipped]

    # Build test rows: genuine + impostor per utterance
    test_rows = []
    for spk, test_wavs in test_genuine.items():
        others = [s for s in speakers if s != spk]
        for wav in test_wavs:
            test_rows.append({'speaker_id': spk,                'wav_path': wav, 'genuine': True})
            test_rows.append({'speaker_id': rng.choice(others), 'wav_path': wav, 'genuine': False})

    rng.shuffle(test_rows)

    # Back up old CSVs if they exist
    for name in ('enroll.csv', 'val.csv', 'test.csv'):
        src = Path('data') / name
        if src.exists():
            src.rename(src.with_suffix('.csv.bak'))
            print(f'[info] Backed up old {name} -> {name}.bak')

    # Write new CSVs
    Path('data').mkdir(exist_ok=True)

    with open('data/enroll.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['speaker_id', 'wav_path'])
        w.writeheader(); w.writerows(enroll_rows)

    with open('data/val.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['speaker_id', 'wav_path', 'genuine'])
        w.writeheader(); w.writerows(val_rows)

    with open('data/test.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['speaker_id', 'wav_path', 'genuine'])
        w.writeheader(); w.writerows(test_rows)

    n_genuine   = sum(1 for r in test_rows if r['genuine'])
    n_impostor  = sum(1 for r in test_rows if not r['genuine'])

    print(f'\n=== Clean data prepared ===')
    print(f'  Speakers        : {len(speakers)} (VoxCeleb1 test set, not seen during pre-training)')
    print(f'  enroll.csv      : {len(enroll_rows)} rows  ({args.enroll_n} per speaker)')
    print(f'  val.csv         : {len(val_rows)} rows  ({args.val_n} per speaker)')
    print(f'  test.csv        : {len(test_rows)} rows  ({n_genuine} genuine / {n_impostor} impostor)')
    print()
    print('Next steps:')
    print('  1. python enroll.py --manifest data/enroll.csv')
    print('  2. python enroll.py --enroll_person bartek_m --audio_dir my_recordings/bartek_m/enroll --display_name "Bartek M."')
    print('  3. python enroll.py --enroll_person bartek_t --audio_dir my_recordings/bartek_t/enroll --display_name "Bartek T."')
    print('  4. python enroll.py --enroll_person krzysiek  --audio_dir my_recordings/krzysiek/enroll  --display_name "Krzysiek"')
    print('  5. python auth.py batch_test --manifest data/test.csv --output results/exp1_baseline.csv')
    print('  6. python experiments/exp1_baseline.py --rerun_impostors')
    print('  7. python experiments/exp2_amplitude.py')
    print('  8. python experiments/exp3_downsample.py')
    print('  9. python experiments/exp4_gaussian_noise.py')
    print(' 10. python experiments/exp5_background_noise.py --noise_dir <UrbanSound8K/audio>')
    print(' 11. python experiments/exp6_codec.py')
    print(' 12. python experiments/exp7_reverberation.py --rir_dir <rir_data/>')


if __name__ == '__main__':
    main()
