# Prospector Finetuning — Stand & Plan (2026-11-03)

Ziel: SFT-Stufe 1 (`tsmc`, `0.8b`, Tag `flash20`) nachbauen und trainieren.

## Erledigt

- **Repo-Spiegel + venv**: `~/prospector/haddock` = Mirror von
  lamm-mit/Prospector v0.2.3 (bc6ec9c); pytest 553 passed / 36 skipped.
- **Release-Daten** via `hf download lamm-mit/Prospector` ( HF_Xet-Problem:
  `HF_HUB_DISABLE_XET=1` setzen; `HF_HOME` weg vom root-owning Default).
- **Datensatz sft1**: `data/traj/tsmc_sft_mix_flash20.jsonl` — 5.800
  Transcripts, 21.976.819 Tokens (25 % trainable).
- **T4-Patch (validiert, Kaggle-Kernel `prospector-sft1-t4patch-03` v6)**:
  1. `use_liger_kernel: true` — liger-kernel 0.8.5 (`qwen3_5_lce_forward`)
     eliminiert die fp32-Logit-Allocation (10,4 GiB bei 152k-Vocab × 12k Seq).
  2. `amp: fp16` — GradScaler; T4 hat kein bf16/TF32.
  3. `model_dtype: float16` + RMSNorm/LayerNorm → fp32 (Residual-Stabilität).
  - Smoke grün: `train_loss 0.7771` (ohne GradScaler: 1,11e9 Overflow),
    377 s/Optimizer-Step, Adapter gespeichert.
  - Trainer-Patch: `src/graph_reasoner/train/sft.py` — neue Config-Keys
    `use_liger_kernel` und `amp` in `SFT_CONFIG_KEYS` + `TrainingArguments`.
  - Config: `configs/sft1_t4patch.yaml`.
  - Falle: liger fused CE ist inkompatibel mit HF-DataParallel; 2 GPUs
    brauchen torchrun-DDP (Smoke: `CUDA_VISIBLE_DEVICES=0`).
- **OOM-Befunde**: Original sft1.yaml braucht A100-80GB-Klasse — Kaggle T4
  (OOM 10,39 GiB Alloc) und Colab L4 (22,03 GiB, OOM 6,12 GiB in backward).
  Colab-Guthaben 2,73 CU — unangetastet, L4-Smoke verworfen.

## Plan: volle SFT1 auf Kaggle 2xT4 (Kette statt Miete)

Rechnung: 5.800/32 × 3 Epochen ≈ 1.632 Optimizer-Steps.

- **DDP**: torchrun, 2 Prozesse → beide T4s → ~85 h GPU ≈ 190 s/Step.
- **Checkpoint-Karussell** (Skill `kaggle-longrun-checkpointing`):
  - `save_steps: 50` (Trainer-Resume aus `output_dir` ist eingebaut).
  - Jede Session: Checkpoint-Dataset mounten → trainieren (12 h) →
    `/kaggle/working` Output pullen → als neues Dataset pushen.
  - ~7 Sessions à 12 h ≈ 1 Woche (45 h/Wochen-Quota passt).
  - Letzte Session: `final/` = fertiger Adapter → lokal + HF sichern.
- **Session-Kette**: Kernel v7 mit torchrun-DDP + Resume-Logik; Status-Tracker
  pro Session (welche Session, welcher Checkpoint-Stand).
- **Risiken**: Session-Ablehnung zu Spitzenzeiten, Output-Pull-Fehler,
  Kettenabbruch bei fehlender Nachfolge-Session → Tracker pflegen.

## Alternativen (nicht gewählt)

- Runpod A100-80GB: 18-25 h, ~$30-50 (Secure) / ~$20-30 (Spot). Resume
  eingebaut; bleibt Fallback, falls die Kaggle-Kette bricht.
- Colab L4: OOM; 2,73 CU reichen nicht für volle SFT1 (10-25 CU).

## Artefakte

| Was | Wo |
|---|---|
| Trainer-Patch | `src/graph_reasoner/train/sft.py` |
| T4-Config | `configs/sft1_t4patch.yaml` |
| Kaggle-Kernel v6 (grün) | `launch/t4patch/prospector_t4patch_smoke.py` |
| Colab-Skripte (L4-Versuch) | `launch/t4patch/colab/` |
| Skills (Operator-Build) | `skills/` |
