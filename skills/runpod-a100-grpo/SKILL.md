---
name: runpod-a100-grpo
description: GRPO-Training fürs Prospector-Rezept (Spec 80–96 GB) auf gemieteter Runpod-A100, gekoppelt an runpod-cost-guard. Use when der Nutzer GRPO "auf miete", "auf runpod", "a100", oder weil 2xT4 zu klein laufen lassen will; enthält Kostendisziplin und Session-Ablauf.
---

# Runpod-A100-GRPO — gemietete 80-GB-Session

Kostendisziplin ist Pflicht: globaler Skill `runpod-cost-guard` VOR und NACH
jedem Lauf (Idle-Pods kosten, Stopped ≠ gratis, Kill-Switch-Check am Sessionende).

## Ablauf

1. **Vorher:** Prospector-Stufen, die NICHT A100 brauchen (sft1, sft2 — LoRA-SFT
   passt auf kleinere Karten/Kaggle, siehe `vllm-2xt4-grpo`) schon erledigt oder
   geparkt; nur GRPO-Stufen (grpo_full, d1/d2) auf die A100.
2. **Pod:** A100 80GB, On-Demand nur für die GRPO-Session; Disk-Größe an
   HF-Cache anpassen (`~/prospector/hf-cache` hochladen oder HF_HOME aufs Pod-Volume).
3. **Setup:** Repo `~/prospector/haddock`/`lab` + Data + Checkpoints rüber
   (Checkpoints = Kaggle-Dataset gemountet oder rsync); Container-Image möglichst
   nah am Lab-Image (`docs/RUNNING.md`-Konvention: Image-ID dokumentieren).
4. **Lauf:** Original-Config `configs/grpo_full.yaml` (keine 2xT4-Fraktionen!),
   `prospector run`/Stufen-Skip nutzen (Skill `prospector-ops`); Watchdog/tail
   über `kaggle-live-logs`-Prinzip: Logdatei pollen statt --follow.
5. **Checkpoint-Disziplin:** adapter+run.json nach jeder Stufe sofort vom Pod
   runter (rsync) UND als Kaggle-Dataset versionieren — der Pod ist wegwerfbar.
6. **Nachher:** runpod-cost-guard Sitrep, Pod stoppen/terminieren, Kosten im
   Run-Verzeichnis notieren (Stunden × Preis), Ergebnisse zurück in `~/prospector/kaggle/`.

## Entscheidungshilfe 2xT4 vs A100

- 2xT4: gratis, aber num_generations↓/turn_budget↓ (Qualitätsrisiko bei GRPO)
  und 12h-Grenzen — gut für Smoke und sft-Stufen.
- A100: echte GRPO-Spec (Original-Config), Stundenkosten — nur für grpo-Stufen
  mit bereits gerauchter Pipeline. Faustregel: erst 2xT4-Smoke grün, dann A100.
