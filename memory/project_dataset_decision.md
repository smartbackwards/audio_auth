---
name: Dataset/model switch decision pending
description: User is deciding whether to switch model or dataset to fix data leakage concern; to be resolved after exp6/exp7 are coded
type: project
---

User wants to switch either the model or the evaluation dataset to address the data leakage concern (speechbrain/spkrec-ecapa-voxceleb was pretrained on VoxCeleb, and the eval speakers are VoxCeleb speakers). Logically it doesn't sit right with them even if technically defensible.

**Why:** The 0.20% EER baseline looks suspiciously inflated. User prefers clean experimental design.

**Current state:**
- Experiments 1–7 are all coded (exp6/exp7 written 2026-05-05)
- Data split (speaker_split.json) is considered very good — user wants to keep it
- Fine-tuned model exists at results/ecapa_finetuned/finetuned_encoder.pt

**Options on the table:**
- Keep data split, swap pretrained backbone to one NOT trained on VoxCeleb (e.g. WeSpeaker model trained on CN-Celeb)
- Keep model, swap eval data to non-VoxCeleb speakers (CNCeleb1 or scraped YouTube) — but this would break the existing data split

**How to apply:** When user returns to this topic tomorrow, resume the discussion. The preferred direction is swapping the model backbone while keeping the VoxCeleb data split. The PDF (L2.pdf) mentions WeSpeaker and 3D Speaker Toolkit as alternatives to SpeechBrain.
