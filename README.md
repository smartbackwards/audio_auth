# Speaker Recognition System
### Biometria Projekt 2 — VoxCeleb2 + ECAPA-TDNN

---

## Stack
- **Model**: ECAPA-TDNN (pretrained on VoxCeleb, fine-tuned)
- **Toolkit**: SpeechBrain
- **Dataset**: VoxCeleb1 (default) or VoxCeleb2
- **Similarity**: Cosine distance on L2-normalised embeddings

---

## Setup

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. Get VoxCeleb2

The official VoxCeleb registration page is currently unavailable.
Download via HuggingFace (`ProgramComputer/voxceleb`) instead.

**VoxCeleb1** is recommended — ~1211 speakers, ~30 GB compressed WAV (no conversion needed):
```bash
python download_voxceleb.py --dataset vox1 --output_dir ./voxceleb1
```

VoxCeleb2 is larger (~100 GB compressed AAC, 5994 speakers) — only use if you need scale:
```bash
python download_voxceleb.py --dataset vox2 --output_dir ./voxceleb2
```

The script will:
1. List the matching `*.zip` parts in the HF repo
2. Download them to `<output_dir>/_zips/` with progress (resumable — safe to interrupt and re-run)
3. Join the parts into a single ZIP
4. Extract audio files and delete the ZIPs

<details>
<summary>Alternative: official archives (when the website is available)</summary>

1. Register at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
2. Download and extract so the structure matches:
   - VoxCeleb1: `voxceleb1/wav/<id>/<video>/<utt>.wav`
   - VoxCeleb2: `voxceleb2/dev/aac/<id>/<video>/<utt>.m4a`
</details>

---

## Week 1 Workflow

### Step 1 — Prepare data splits

```bash
# Download VoxCeleb1 (see "Get VoxCeleb" above)
python download_voxceleb.py --dataset vox1 --output_dir ./voxceleb1

# Create the train/val/enroll/test manifests
python prepare_data.py --dataset vox1 --data_root ./voxceleb1 --output_dir ./data
```

This creates four **fully disjoint** CSVs:
| File | Purpose | VoxCeleb1 speakers |
|------|---------|-------------------|
| `data/train.csv` | Fine-tuning | 700 |
| `data/val.csv` | Validation during fine-tuning | 51 |
| `data/enroll.csv` | Enrolled into the speaker DB | 200 |
| `data/test.csv` | All experiments (enrolled + test-only) | 200 + 230 |

---

### Step 2 — Fine-tune (run on GPU machine or Colab)

#### On your mate's GPU / university cluster:
```bash
python finetune.py finetune.yaml
```

#### On Google Colab:
```python
# In a Colab notebook:
!git clone https://github.com/yourrepo/speaker_recognition
%cd speaker_recognition
!pip install -r requirements.txt

# Upload data/ folder or mount Drive
from google.colab import drive
drive.mount('/content/drive')

!python finetune.py finetune.yaml
```

Fine-tuning takes ~2-4 hours on a single GPU for 10 epochs.
The best checkpoint is saved to `results/ecapa_finetuned/save/`.

---

### Step 3 — Enroll 100 speakers from VoxCeleb

```bash
python enroll.py --manifest data/enroll.csv --model_dir results/ecapa_finetuned/save
```

---

### Step 4 — Enroll yourselves (group members)

Record 10+ WAV clips each (~5 seconds, 16kHz mono). Audacity works great.
Save to separate folders, then:

```bash
python enroll.py --enroll_person "bartek_t" --display_name "Bartek T." \
  --audio_dir my_recordings/bartek_t/ --model_dir results/ecapa_finetuned/save

python enroll.py --enroll_person "bartek_m" --display_name "Bartek M." \
  --audio_dir my_recordings/bartek_m/ --model_dir results/ecapa_finetuned/save

python enroll.py --enroll_person "krzysiek" --display_name "Krzysiek" \
  --audio_dir my_recordings/krzysiek/ --model_dir results/ecapa_finetuned/save
```

> ⚠️ The project requires that enrolled samples are disjoint from training data.
> Your own voice recordings are not in VoxCeleb, so they're automatically safe.

---

### Step 5 — Test the system

```bash
# Verify a specific speaker
python auth.py verify --speaker_id bartek_t --audio_path sample.wav

# Identify unknown speaker (1:N)
python auth.py identify --audio_path sample.wav

# Run baseline experiment (Experiment 1)
python auth.py batch_test --manifest data/test.csv \
  --output results/experiment_1_baseline.csv
```

---

## Database design (per spec)

- **Only embeddings are stored** — no raw audio in `speaker_db.json`
- Each speaker entry: `{ display_name, embeddings: [[...], [...], ...] }`
- Centroid vectors pre-computed for fast 1:N lookup

### Privacy / risk note (for report)
- Embeddings are not trivially reversible to audio, but inversion attacks exist
- Risk: if DB leaks, embeddings could be used for replay attacks on other systems
- Mitigation: store embeddings encrypted at rest, rotate after system compromise
- Retention: delete embeddings when user requests removal

---

## Project structure

```
speaker_recognition/
├── requirements.txt       # All dependencies
├── download_voxceleb.py   # Download VoxCeleb2 from HuggingFace → ./voxceleb2/
├── prepare_data.py        # Dataset split → 4 disjoint CSVs
├── finetune.yaml          # SpeechBrain training config
├── finetune.py            # Fine-tuning script (run on GPU)
├── enroll.py              # Build speaker database
├── auth.py                # Verify (1:1) and identify (1:N)
├── data/                  # Generated by prepare_data.py
│   ├── train.csv
│   ├── val.csv
│   ├── enroll.csv
│   ├── test.csv
│   └── speaker_split.json
├── results/               # Experiment outputs
├── speaker_db.json        # Embedding database (generated by enroll.py)
└── my_recordings/         # Your own voice samples (not committed to git)
    ├── bartek_t/
    ├── bartek_m/
    └── krzysiek/
```

---

## Week 2+ (coming next)
- `experiments/` — scripts for all 7 test experiments
- `analysis.py` — ROC curve, FAR/FRR plots, EER calculation
- Report template
