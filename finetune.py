"""
finetune.py

Fine-tunes a pretrained ECAPA-TDNN speaker encoder on VoxCeleb2 train split.
Designed to run on a GPU machine (your mate's or Colab).

Usage (local GPU or Colab):
  python finetune.py finetune.yaml

Colab quick-start:
  !git clone <your repo>
  %cd speaker_recognition
  !pip install speechbrain
  !python finetune.py finetune.yaml
"""

import sys
import csv
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path


# ── Dataset ────────────────────────────────────────────────────────────────────

class VoxDataset(torch.utils.data.Dataset):
    """
    Reads a manifest CSV (speaker_id, wav_path) and returns fixed-length chunks.
    Builds a speaker-to-int label map on the fly.
    """

    def __init__(self, csv_path: str, sample_rate: int, chunk_seconds: float,
                 augment: bool = False):
        self.sample_rate   = sample_rate
        self.chunk_samples = int(chunk_seconds * sample_rate)
        self.augment       = augment
        self.items         = []
        self.spk2idx       = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spk = row["speaker_id"]
                if spk not in self.spk2idx:
                    self.spk2idx[spk] = len(self.spk2idx)
                self.items.append((row["wav_path"], self.spk2idx[spk]))

        print(f"[Dataset] {csv_path}: {len(self.items)} utterances, "
              f"{len(self.spk2idx)} speakers")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, label = self.items[idx]
        wav, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Mix down to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0)  # (T,)

        # Random crop / pad to fixed length
        if wav.shape[0] >= self.chunk_samples:
            start = random.randint(0, wav.shape[0] - self.chunk_samples)
            wav   = wav[start: start + self.chunk_samples]
        else:
            # Repeat-pad
            repeats = (self.chunk_samples // wav.shape[0]) + 1
            wav     = wav.repeat(repeats)[:self.chunk_samples]

        return wav, label


def collate_fn(batch):
    wavs, labels = zip(*batch)
    wavs   = torch.stack(wavs)        # (B, T)
    labels = torch.tensor(labels)     # (B,)
    return wavs, labels


# ── Brain ──────────────────────────────────────────────────────────────────────

class SpeakerBrain(sb.Brain):

    def compute_forward(self, batch, stage):
        wavs, labels = batch
        wavs = wavs.to(self.device)

        # Feature extraction
        feats = self.modules.compute_features(wavs)       # (B, T, 80)
        feats = self.modules.mean_var_norm(feats, torch.ones(feats.shape[0], device=self.device))

        # Embedding
        embeddings = self.modules.encoder(feats)          # (B, emb_dim)
        return embeddings, labels.to(self.device)

    def compute_objectives(self, predictions, batch, stage):
        embeddings, labels = predictions
        loss = self.hparams.classifier(embeddings, labels)
        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.VALID:
            print(f"Epoch {epoch} | Val loss: {stage_loss:.4f}")
            self.checkpointer.save_and_keep_only(
                meta={"val_loss": stage_loss},
                min_keys=["val_loss"],
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    hparams_file = sys.argv[1] if len(sys.argv) > 1 else "finetune.yaml"

    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    # Build datasets
    train_dataset = VoxDataset(
        csv_path       = hparams["train_csv"],
        sample_rate    = hparams["sample_rate"],
        chunk_seconds  = hparams["sentence_len"],
        augment        = True,
    )
    valid_dataset = VoxDataset(
        csv_path       = hparams["valid_csv"],
        sample_rate    = hparams["sample_rate"],
        chunk_seconds  = hparams["sentence_len"],
        augment        = False,
    )

    # Patch classifier output size to match actual number of training speakers
    n_speakers = len(train_dataset.spk2idx)
    print(f"[finetune] Training with {n_speakers} speakers")

    # Build linear classifier head (separate from AAM-softmax wrapper)
    classifier_head = torch.nn.Linear(hparams["emb_dim"], n_speakers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = hparams["batch_size"],
        shuffle     = hparams["shuffle"],
        num_workers = 4,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size  = hparams["batch_size"],
        shuffle     = False,
        num_workers = 2,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    # Load pretrained weights
    print("[finetune] Loading pretrained ECAPA-TDNN from HuggingFace...")
    from speechbrain.pretrained import EncoderClassifier
    pretrained = EncoderClassifier.from_hparams(
        source      = hparams["pretrained_path"],
        savedir     = "pretrained_models/ecapa",
        run_opts    = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    # Transfer encoder weights
    hparams["encoder"].load_state_dict(
        pretrained.mods.embedding_model.state_dict(), strict=False
    )
    print("[finetune] Pretrained weights loaded.")

    # Wire up modules
    modules = {
        "encoder":          hparams["encoder"],
        "classifier":       classifier_head,
        "compute_features": hparams["compute_features"],
        "mean_var_norm":    hparams["mean_var_norm"],
    }

    brain = SpeakerBrain(
        modules     = modules,
        opt_class   = hparams["opt_class"],
        hparams     = hparams,
        run_opts    = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer= hparams["checkpointer"],
    )

    brain.fit(
        epoch_counter = hparams["epoch_counter"],
        train_set     = train_loader,
        valid_set     = valid_loader,
    )

    print(f"\n[finetune] Done. Best checkpoint saved to: {hparams['save_folder']}")
    print("Next step: run enroll.py to build your speaker database.")


if __name__ == "__main__":
    main()
