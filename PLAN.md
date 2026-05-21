# Project Plan — Remaining Work
### Biometria Projekt 2 — Experiments, Analysis, Report

---

## Status

| Step | Done |
|------|------|
| VoxCeleb1 download & splits | ✅ |
| Fine-tuning | ⏭️ skipped (pretrained checkpoint used) |
| Enroll 200 VoxCeleb speakers | ✅ |
| Enroll Bartek M | ✅ |
| Enroll remaining group members | ⏳ waiting for recordings |
| Threshold calibration | ⏳ in progress |
| Experiments 1–7 | ❌ |
| Analysis / metrics | ❌ |
| Report | ❌ |

---

## Remaining Setup

### Finish enrollment (Step 4)

Record 10+ WAV clips each (~5 s, 16 kHz mono). Split into `enroll` (6), `val` (2), `test` (2) subfolders.

```
python enroll.py --enroll_person "bartek_t" --audio_dir my_recordings/bartek_t_enroll
python enroll.py --enroll_person "krzysiek"  --audio_dir my_recordings/krzysiek_enroll
```

### Calibrate threshold (Step 5)

Run val clips through `auth.py verify` for all group members, pick a threshold that
accepts all genuine val clips. Update `DEFAULT_THRESHOLD` in [auth.py](auth.py).

```
python auth.py verify --speaker_id bartek_m --audio_path my_recordings/bartek_m/val/file_9.wav --threshold 0.40
```

---

## Experiments

All experiments use **unseen test samples** (never seen during enrollment or threshold tuning).
Results must be reported with standard biometric metrics: **EER, FAR, FRR, ROC curve**.

> Scripts to write: `experiments/exp{1..7}.py` + `analysis.py`

---

### Experiment 1 — Baseline

**Samples:** ≥ 500, balanced genuine / impostor trials
**Task:** Run `auth.py batch_test` on `data/test.csv`, report metrics.

```
python auth.py batch_test --manifest data/test.csv --output results/exp1_baseline.csv
python analysis.py --results results/exp1_baseline.csv --plot results/exp1_roc.png
```

**Report:** EER, FAR @ fixed FRR, FRR @ fixed FAR, ROC curve. Compare with published
ECAPA-TDNN numbers (VoxCeleb1-O EER ≈ 0.87% for the fine-tuned model; expect higher
without fine-tuning).

---

### Experiment 2 — Amplitude Scaling

**Samples:** ≥ 500
**Task:** For each sample, randomly multiply amplitude by one of `{25, 1, 0.04}` (uniform).
Compare accuracy to Experiment 1.

```
python experiments/exp2_amplitude.py --manifest data/test.csv --output results/exp2_amplitude.csv
```

**Results:** ×0.04 barely degrades (EER 0.22%, +0.02pp) — log-mel features are amplitude-invariant.
×25 clips the signal and causes real degradation (EER 2.72%, +2.53pp, FAR@FRR=1% jumps to 6.32%).

**Potential fix for ×25:** peak-normalise before embedding (`wav /= wav.abs().max()`).
This brings the clipped signal back into the linear range before feature extraction and should
recover most of the lost accuracy. Worth adding as a mitigation experiment.

---

### Experiment 3 — Downsampling

**Samples:** 200
**Task A — Naive (drop samples):** keep every 2nd, 5th, 10th sample (update sample-rate metadata accordingly).
**Task B — Proper (interpolation):** resample 2×, 5×, 10× using `torchaudio.functional.resample`.

For each variant: measure accuracy **and** note how the required clip length changes
(shorter effective duration → system may need longer input).

```
python experiments/exp3_downsample.py
```

**Codecs / factors to test:** ×2, ×5, ×10 for both naive and interpolated.

**Results:**

| Factor | Naive EER | Proper EER |
|--------|-----------|------------|
| ×2  | 0.90%  (+0.71pp) | 1.47%  (+1.27pp) |
| ×5  | 11.75% (+11.55pp) | 9.34%  (+9.15pp) |
| ×10 | 24.22% (+24.03pp) | 19.50% (+19.30pp) |

Key findings:
- **Naive beats proper at ×2**: aliasing folds 4–8 kHz back into 0–4 kHz — distorted but still
  carries speaker identity. Proper hard-removes it entirely.
- **Proper wins at ×5 and ×10**: aliasing becomes too severe at high factors, clean narrow-band
  beats corrupted wide-band.
- **Cliff between ×2 and ×5**: EER jumps from ~1% to ~10% — this is the practical usability limit.
- **×10 is non-functional** (EER ~20–24%, FAR@FRR=1% at 77–89%).
- Downsampling is far more damaging than amplitude scaling (cf. exp2 ×0.04 = +0.02pp).

---

### Experiment 4 — Gaussian Noise

**Samples:** 100
**SNR levels:** 40 dB, 20 dB, 10 dB
**Task:** Add white Gaussian noise calibrated to achieve the target SNR relative to each
original signal. Compare accuracy to Experiment 1.

```
python experiments/exp4_gaussian_noise.py --manifest data/test.csv --output results/exp4_noise.csv
```

**SNR formula:**
```
noise_power = signal_power / 10^(SNR_dB / 10)
noise = sqrt(noise_power) * randn(len(signal))
```

---

### Experiment 5 — Background Noise (Environmental)

**Samples:** 100
**SNR levels:** 20 dB, 10 dB, 0 dB
**Dataset:** [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) or equivalent.
**Task:** Mix speech with a randomly selected background clip at the target SNR.

```
python experiments/exp5_background_noise.py --manifest data/test.csv \
  --noise_dir urban_sound8k/ --output results/exp5_background.csv
```

---

### Experiment 6 — Lossy Compression

**Samples:** 100 (or as many as needed for statistical significance)
**Codecs:** MP3, AAC, Opus — at least 3 bitrate settings each.

| Codec | Bitrates to test |
|-------|-----------------|
| MP3   | 320k, 128k, 32k |
| AAC   | 256k, 96k, 32k  |
| Opus  | 128k, 32k, 8k   |

```
python experiments/exp6_compression.py --manifest data/test.csv --output results/exp6_compression.csv
```

**Tool:** `ffmpeg` for encoding/decoding on the fly.

---

### Experiment 7 — Reverberation

**Samples:** 100
**Task:** Convolve speech with room impulse responses (RIRs). Use
[OpenSLR SLR28](https://www.openslr.org/28/) (MIT IR Survey) or similar.

Test at least 3 different rooms (small/medium/large or varying RT60).

```
python experiments/exp7_reverb.py --manifest data/test.csv \
  --rir_dir rir_data/ --output results/exp7_reverb.csv
```

**Convolution:** `scipy.signal.fftconvolve(speech, rir)`, normalise after.

---

## Analysis Script (`analysis.py`)

Needs to produce for each experiment:

- **ROC curve** (FAR vs TAR)
- **DET curve** (FAR vs FRR, log scale)
- **EER** (where FAR = FRR)
- **FAR @ FRR=1%** and **FRR @ FAR=1%**
- Summary table comparing all experiments

```
python analysis.py --results results/ --output report/figures/
```

---

## Report Checklist

- [ ] Authors: name, student ID, lab slot
- [ ] Authentication method description — model, embedding, cosine distance, threshold
- [ ] Sources cited (ECAPA-TDNN paper, SpeechBrain, VoxCeleb)
- [ ] Original contributions highlighted
- [ ] Dataset section:
  - [ ] Number of speakers (200 VoxCeleb + group members)
  - [ ] Samples per speaker, clip duration
  - [ ] Train / val / enroll / test split sizes and disjointness
- [ ] Enrollment procedure description
- [ ] Authentication procedure description (1:1 verify, 1:N identify)
- [ ] Experiment results with plots (EER, ROC, DET per experiment)
- [ ] Comparison with published baseline (ECAPA-TDNN VoxCeleb1-O)
- [ ] Conclusions

**Deadline:** send to lab instructor ≥ 48 hours before the lab session.
**Demo:** live system demo required during the lab.

---

## Data / Downloads Still Needed

| Resource | Used in | Link |
|----------|---------|------|
| UrbanSound8K | Exp 5 | https://urbansounddataset.weebly.com/urbansound8k.html |
| Room Impulse Responses | Exp 7 | https://www.openslr.org/28/ |

---

## Bonus — Official VoxCeleb1 Trial List

Replace the random-impostor setup with the standard pairwise evaluation to get
a number directly comparable to published ECAPA-TDNN results.

**Trial list files** (plain text, no registration needed):

| List | URL |
|------|-----|
| VoxCeleb1-O (37k pairs) | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt |
| VoxCeleb1-E (581k pairs) | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt |
| VoxCeleb1-H hard (552k pairs) | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt |

**Format:** `label path1 path2`  (label 1 = same speaker, 0 = different)

**Evaluation approach** (different from DB-based verify):
- Embed both clips independently
- Score = cosine distance between the two embeddings
- Sweep thresholds → EER (no DB, no enrollment needed)

Published ECAPA-TDNN EER on VoxCeleb1-O: **0.87%** (fine-tuned).
Pretrained-only will be higher — use VoxCeleb1-O for the comparison.
