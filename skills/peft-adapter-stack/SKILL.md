---
name: peft-adapter-stack
description: 3 LoRA-Adapter in Load-Order mergen fürs Prospector-Serving (sft1 + grpo_full/d1/d2 Kette) mit run.json-Validierung. Use when der Nutzer einen Adapter-Stack für Eval/Serving aufbauen, "--adapter Kette", "run.json prüfen", oder Serving-Templates validieren will.
---

# PEFT-Adapter-Stack — Load-Order + run.json-Validierung

Serving-Contract: `--adapter` nimmt eine **kommaseparierte Kette** in Reihenfolge
sft1 → grpo (README-Referenz):

```bash
STACK=models/traj_sft_4b_flash20g2/final,\
models/traj_grpo_4b_flash20d1/checkpoint-120,\
models/traj_grpo_4b_flash20d2/checkpoint-120
python scripts/run_agent_eval.py --graph tsmc --model vllm:Qwen/Qwen3.5-4B --adapter $STACK ...
```

## Regeln

1. **Load-Order ist semantisch:** jeder spätere Adapter ist auf den Output des
   vorherigen trainiert (sft2/grpo-Ketten). Niemals umsortieren, niemals Stufen auslassen.
2. **run.json je Adapter validieren VOR dem Start:** gleiche base model / revision,
   kompatible chat_template (das LASTE Adapter-Template zählt — `run_agent_eval.py`
   nimmt es aus dem letzten Adapter der Kette), gleiche lora target modules.
   Mismatch → Serving liefert stille Schrottqualität, kein Fehler.
3. **Template-Drift-Falle:** wenn ein Adapter ein anderes Template aufgezeichnet
   hat als der Base — Eval-Namen entsprechend vermerken und Policy-Zahlen nur
   aus vllm:-Runs vergleichen (README: "policy numbers come from the vllm: driver").
4. **Checkpoint-Stand pinnen:** checkpoint-NNN statt `final` mischen ist okay,
   aber im Eval-Namen dokumentieren (`--name`), sonst sind Ergebnisse nicht zuordenbar.
5. Siehe auch Rezept-Begriffe (gate/grid/pick/coda): `src/graph_reasoner/recipe/README.md`.
