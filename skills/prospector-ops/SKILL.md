---
name: prospector-ops
description: Projekt-Runbook fürs Prospector-Finetuning (github.com/lamm-mit/Prospector, 4-Stufen-Rezept) — Pfade, Stufen-Reihenfolge, Smoke-Modus, GRPO-Hardware-Bedarf. Use when der Nutzer Prospector trainieren, "prospector run", "sft1", "grpo", "smoke", oder Stufen des Rezepts starten/verstehe will.
---

# Prospector-Ops — Projekt-Runbook

## Pfade (dieser Rechner)

- `~/prospector/haddock` — Code + Release (upstream Mirror, lauffähig)
- `~/prospector/lab` — Trainings-Checkout (configs, scripts, SKILLS hier)
- `~/prospector/hf-cache` — **HF-Cache HIER** (root-cache-Blocker umgangen:
  nie `~/.cache/huggingface` direkt, immer `HF_HOME` auf diesen Pfad zeigen lassen)
- `~/prospector/kaggle/` — Kaggle-Kernel-Artefakte (aktueller 2xT4-Pilot: `prospector-sft1-smoke-01`)

## Das 4-Stufen-Rezept (configs/README.md)

| Stufe | Was | Config | Befehl |
|---|---|---|---|
| 1 sft1 | LoRA-SFT auf Teacher-Demos, **fp32-logits → batch_size 1 × grad_accum 32** | `configs/sft1.yaml` | `scripts/train_sft.py --stage sft1` |
| 2 grpo_full | GRPO über ganze Episoden (medium/hard), **lr 3e-5** | `configs/grpo_full.yaml` | `scripts/train_traj_grpo.py --stage full` |
| 3 sft2 | SFT auf Demos + verifizierte gate-passing Episoden aus Stufe 2 | `configs/sft2.yaml` | `scripts/train_sft.py --stage sft2` |
| 4 grpo | beide Decision-Rounds, **lr 1.5e-5** | `configs/grpo_decision.yaml` | `scripts/train_traj_grpo.py --stage d1` + `--stage d2` |

Orchestrierung: `prospector run --graph tsmc --size 4b …` (Stufen 1–4 + Selection,
"days on one card"), `prospector plan` zeigt trocken was passieren würde.

## Skip-Verhalten (wichtig für Resume)

Jede Stufe **skippt existierende Outputs** (adapter + gespeicherte Evals) und
leitet Dateinamen aus `--graph/--size/--tag` ab. Jeder Adapter schreibt `run.json`
(woraus er trainiert wurde, Chat-Template, Policy-Zahlen). Relaunch = Resume;
Released-Adapter vorhanden → es wird nichts trainiert.

## Smoke-Modus

Vor jedem Langlauf: Smoke (kurze Stufen, Mini-Daten) — aktueller Pilot:
`~/prospector/kaggle/prospector-sft1-smoke-01`. Niemals direkt in 4-Stufen-Durchlauf starten.

## GRPO-Hardware

GRPO braucht **eine CUDA-GPU mit 80–96 GB** (Spec README). Auf 2×T4 (32 GB):
Skill `vllm-2xt4-grpo`; auf Miete: Skill `runpod-a100-grpo`.
Session-Grenzen auf Kaggle: Skill `kaggle-longrun-checkpointing`.
