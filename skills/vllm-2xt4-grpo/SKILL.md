---
name: vllm-2xt4-grpo
description: GRPO-Config für 2×T4 (32 GB) im Prospector-Projekt — num_generations/turn_budget reduzieren, vllm-GPU-Fraktion/kv-cache/Max-Nums-S fraktionierte Colocation. Use when GRPO auf Kaggle 2xT4, "grpo auf t4", OOM im GRPO-Run, oder eine --config-Experimentsdatei für 32-GB-Karten angepasst werden soll.
---

# vLLM-2xT4-GRPO — GRPO unter 32 GB

Spec-Bedarf des Rezepts: 80–96 GB (eine GPU). 2×T4 = 32 GB → alles fraktionieren.
Basis: `~/prospector/lab/configs/grpo_full.yaml` (nicht verändern — Kopie als
`configs/grpo_full_2xt4.yaml` anlegen und als `--config` mitgeben).

## Stellschrauben (in dieser Reihenfolge)

1. **`num_generations: 16 → 4–8`** — größter Speicher-/Zeitfaktor (jede Generation
   ist ein kompletter Episode-Rollout mit vLLM-Batch).
2. **`turn_max_tokens: 1024 → 512`** — halbiert KV-Cache pro Rollout; Gradients
   brauchen den Speicher, nicht die Generierung.
3. **`vllm_gpu_frac: 0.35`** — Fraktion der GPU, die vLLM (Rollout-Engine) neben
   dem Trainer bekommt; auf T4 mit 0.20–0.25 starten, erst erhöhen wenn Trainer-OOM
   weg ist. `vllm_sleep: true` lassen (Engine schlafen legt VRAM zwischen den
   Rollout-Phasen frei).
4. **`vllm_max_num_seqs: 64 → 8–16`** — parallele Sequenzen im Rollout; kleinstmöglich halten.
5. **`batch_size`/accum wie sft1** (fp32-Logits über Vokabular begrenzen per-device
   batch auf 1; kompensieren mit grad-accum).
6. Trainer-seitig: Gradient-Checkpointing an, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (Fleet-erprobt, mtp-lab).

## Reihenfolge beim Debuggen

Erst OOM weg (1→4), dann Qualität retten: num_generations ist GRPO-wirksamster
Knopf — bei 4er-Gruppen Advantage-Schätzung verrauscht; lieber Steps erhöhen als
Gruppen unter 4 fallen lassen. Jede Änderung mit Smoke (Mini-Daten) testen, dann
Session planen (Skill `kaggle-longrun-checkpointing`).
