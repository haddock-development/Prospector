---
name: kaggle-longrun-checkpointing
description: Resumables Training über die 12h-Kaggle-Session-Grenze im Prospector-Projekt — Adapter-Checkpoints als Kaggle-Dataset versionieren, Stufen-Skip zum Wiederaufsetzen nutzen. Use when ein Kaggle-Kernel-Training die Session-Grenze reißt, "12h", "session limit", "checkpoint nach kaggle", oder ein abgebrochener Lauf fortgesetzt werden soll.
---

# Kaggle-Longrun-Checkpointing — über die 12h-Grenze

Fleet-Wissen: `jev-kaggle-ops` (Push/Quota/Output-Fallen), `kaggle-live-logs`
(Logs trotz Sync-Lag). Diese Skill ergänzt das Checkpoint-Protokoll.

## Protokoll

1. **Checkpoint-Rhythmus:** nach jeder abgeschlossenen Stufe (sft1, grpo_full,
   sft2, grpo d1/d2 — nie mitten in einer Stufe, außer der Trainer schreibt
   selbst checkpoint-N) Adapter-Dir + `run.json` als **neues Kaggle-Dataset-
   Version** pushen (Dataset pro Run, Version pro Checkpoint). Kaggle-Output
   allein ist NICHT sicher (Session-Grenze purgt).
2. **Resumen in neuem Kernel:** Checkpoint-Dataset mounten, Pfade wie im
   Original-Run setzen, dann einfach denselben `prospector run`/Stufen-Befehl
   feuern — der Stufen-Skip überspringt komplette existierende Stufen
   (`run.json`-Identität prüfen: gleicher --graph/--size/--tag, sonst trainiert er neu!).
3. **Identität verbürgen:** vor Resume `run.json` des gemounteten Checkpoints
   gegen die erwartete Config diffen (chat template, base model, lora targets).
4. **Budget-Plan:** Stufe grpo_full ist die längste — Session-Laufzeit vorher
   aufteilen (z.B. sft1+smoke in Session A, grpo_full in Session B, sft2+grpo in C);
   Kernel-Deadline via kaggle-live-logs überwachen, nicht --follow.
5. **Transfer-Goldenregel:** Kernel-Output → `kernels output` downloaden ist
   unzuverlässig (jev-kaggle-ops); der Dataset-Push aus dem Kernel heraus ist
   der verlässliche Weg.

## Referenz

Aktueller Pilot: `~/prospector/kaggle/prospector-sft1-smoke-01`.
Verwandt: `prospector-ops` (Stufen/Skip), `vllm-2xt4-grpo` (Speicher statt Zeit).
