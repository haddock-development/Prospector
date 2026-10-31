---
name: graph-build-pipeline
description: Eigener Knowledge-Graph-Bau aus PDFs fürs Prospector-Finetuning (marker-pdf → Fragmente → Merge → graphml im tsmc_5b10p-Format) für eigene Domänen. Use when der Nutzer einen eigenen Graphen bauen, "graph für domäne X", "PDFs zu graphml", "tsmc-Format", oder eval mit --graph auf eigenem Material laufen will.
---

# Graph-Build-Pipeline — eigene Domänen als Prospector-Graph

Referenz-Format: der mitgelieferte Halbleiter-Graph `tsmc_5b10p` (5 Bücher /
10 Papers) und sein Builder unter `src/graph_reasoner/recipe/` — eigenes Material
nach demselben Schema bauen, damit `--graph <name>` überall funktioniert.

## Pipeline

1. **PDFs → Text:** `marker-pdf` (Fleet: läuft GPU-lastig; auf 4-GB-Karte nur
   Einzeldokument, sonst Kaggle/Colab). Output: Markdown je Dokument.
2. **Fragmente:** Dokumente in die Fragment-Granularität des tsmc-Graphs
   schneiden (Abschnitt-/Absatz-Blöcke mit stabiler ID + Quellenverweis —
   IDs durchgehend stabil halten, Eval-Citation-Gate hängt an den Belegen).
3. **Merge:** Fragmente je Quelle mergen, Kanten/Entity-Typen exakt wie im
   tsmc-Schema übernehmen (Schema-Divergenz = Eval bricht oder vergleicht Äpfel mit Birnen).
4. **Export als graphml** im tsmc_5b10p-Layout, Dateinamen/Split-Konvention
   (Bücher vs Papers) beibehalten.
5. **Abnahme:** `prospector plan --graph <neu>` (trocken), dann
   `run_agent_eval.py --graph <neu> --limit 5` gegen ein kleines Modell —
   erst wenn Zitationen/Citations grün sind, Training auf dem neuen Graphen ansetzen.

## Fallen

- PDF-Extraktion ist der Qualitäts-Bottleneck: Tabellen/Formeln in marker-pdf
  stichprobenartig gegen das Original prüfen, bevor 500 Fragmente drauf trainieren.
- Eigene Domäne = eigene Frageverteilung: Eval-Fragen (Gold) nicht aus den
  Trainings-PDFs paraphrasieren (Contamination — Frozen-Gold-Denkschule siehe deterministic-agents).
- Große Domänen: erst 1 Buch + 2 Papers end-to-end, dann skalieren.
