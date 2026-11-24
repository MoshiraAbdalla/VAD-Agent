# Result Summaries

## VAD-Agent Detection with Offline Temporal Refinement

- UCF-Crime: AUC `0.830438160` (`83.04%`).
- XD-Violence: AP `0.591460854` (`59.15%`), AUC `0.846424071` (`84.64%`).

Frame-level stride scores are assigned at their saved segment start positions. The evaluation commands use `--gap-policy hold`: the preceding score is retained through an absent output, with a zero score before the first available output. Gaussian smoothing returns one value per saved segment. The full configuration fuses `0.1` direct-segment evidence with `0.9` temporally refined evidence and then applies video-position weighting.

## Context and Narrative Memory Agents

| Configuration | AUC (%) |
| --- | ---: |
| Context Agent only | 77.73 |
| Previous reasoning memory | 80.96 |
| Full VAD-Agent | 83.04 |

## Segment Stride

| Stride (frames) | UCF-Crime AUC (%) | XD-Violence AUC (%) |
| ---: | ---: | ---: |
| 300 | 79.74 | 83.68 |
| 150 | 83.04 | 84.04 |
| 64 | 79.82 | 84.64 |

## UCF-Crime Sequential Inference and Offline Temporal Refinement

| Operation | UCF-Crime AUC (%) | Gain (pp) |
| --- | ---: | ---: |
| Initial segment scores (without temporal refinement) | 72.08 | - |
| Initial + similarity retrieval | 75.70 | +3.62 |
| Initial + similarity retrieval + Gaussian smoothing | 82.31 | +6.61 |
| Refinement with position weighting | 82.83 | +0.53 |
| Full refinement with direct-score fusion | 83.04 | +0.21 |

Rows 2–5 use offline temporal refinement.

## Narrative Agent Backbone

| Narrative agent | Reasoning agent | AUC (%) |
| --- | --- | ---: |
| LLaMA | InternVL2-8B | 79.36 |
| InternVL2-8B | InternVL2-8B | 83.04 |

## Paper Sampling Study on the HIVAU UCF Subset

The same 60-video subset is used for all rows.

| Variant | Context frames | Reasoning frames | AUC (%) |
| --- | ---: | ---: | ---: |
| Uniform for both agents | 8 | 8 | 83.98 |
| MIF for both agents | 8 | 8 | 85.48 |
| Uniform context + MIF reasoning | 8 | 8 | 86.86 |

## Full UCF Sampling Evaluation

These rows use all 290 UCF-Crime test videos.

| Variant | Context frames | Reasoning frames | AUC |
| --- | ---: | ---: | ---: |
| Uniform for both agents | 8 | 8 | 0.792794417 |
| MIF for both agents | 8 | 8 | 0.802549996 |
| Uniform context + MIF reasoning | 8 | 8 | 0.800478823 |
| Uniform context + MIF reasoning | 16 | 8 | 0.830438160 |

## Reproduce the Paper Ablations

```powershell
$env:PYTHON = '<path-to-evaluation-env>\python.exe'
.\scripts\run_paper_ablations.ps1
```

Each run saves its metrics, commands and logs under `results/vad_detection/ablations/runs/`.

## VADTree Generalization

- VADTree baseline AUC is `77.94%`.
- VAD-Agent + VADTree AUC is `78.34%`.
- VAD-Agent + VADTree achieves BLEU `0.300`, CIDEr `1.540`, METEOR `0.116`, ROUGE `0.283`, BERTScore `0.879`, and ParaScore `0.879`.


