# Reproduction Guide

This document maps paper rows to the code and cached artifacts in this repository.

## Reference Evaluation Environment

The paper's detection and ablation metrics were computed from the saved
predictions and features using Windows Anaconda with Python 3.13.5, NumPy 2.1.3
linked to Intel MKL 2023.1.0, SciPy 1.15.3, and scikit-learn 1.6.1.
Use the evaluation interpreter for Sections 3 and 9 of the README.

```powershell
conda create -n vad-agent-eval -c defaults python=3.13.5 numpy=2.1.3 scipy=1.15.3 scikit-learn=1.6.1 'blas=1.0=mkl' mkl=2023.1.0
conda activate vad-agent-eval
$env:PYTHON = Join-Path $env:CONDA_PREFIX 'python.exe'
```

## VAD-Agent Detection

Frame-level stride scores are assigned at their saved segment start positions. The evaluation commands use `--gap-policy hold`: the preceding score is retained through an absent output, with a zero score before the first available output. Gaussian smoothing returns one value per saved segment. The full configuration fuses `0.1` direct-segment evidence with `0.9` temporally refined evidence and then applies video-position weighting.

### UCF-Crime

Saved predictions and features:

```text
results/predictions/ucf_crime_mif_stride150/
results/cached_features/ucf_crime_stride150/
```

Generation code: `src/vad_detection/inference/generate_ucf_mif_stride150.py`.

Evaluation command:

```powershell
.\scripts\run_ucf_vad_agent_fused01.ps1
```

Equivalent explicit command:

```powershell
& $env:PYTHON -B src\vad_detection\evaluation\compute_ucf_temporal_refinement.py `
  --ann-root data\annotations\UCF_Eval.json `
  --score-folder results\predictions\ucf_crime_mif_stride150 `
  --vision-folder results\cached_features\ucf_crime_stride150 `
  --projection-mode stride `
  --gap-policy hold `
  --strict-alignment `
  --include-self-sim `
  --apply-position-weight `
  --position-weight-sigma -1 `
  --top-ratio 0.10 `
  --temp 10 `
  --smooth-size 15 `
  --smooth-sigma 10 `
  --fuse-direct-segment-score `
  --direct-segment-weight 0.1
```

Expected output:

```text
AUC=0.830438160
videos_used=290
missing_videos=0
```

### XD-Violence

Saved predictions and features:

```text
results/predictions/xd_violence_mif_stride64/
results/cached_features/xd_violence_stride64/
```

Generation code: `src/vad_detection/inference/generate_xd_mif_stride64.py`.

Evaluation command:

```powershell
.\scripts\run_xd_vad_agent_fused01.ps1
```

Equivalent explicit command:

```powershell
& $env:PYTHON -B src\vad_detection\evaluation\compute_xd_temporal_refinement.py `
  --ann-root data\annotations\xd_test_800.json `
  --score-folder results\predictions\xd_violence_mif_stride64 `
  --vision-folder results\cached_features\xd_violence_stride64 `
  --projection-mode stride `
  --gap-policy hold `
  --strict-alignment `
  --include-self-sim `
  --apply-position-weight `
  --position-weight-sigma -1 `
  --top-ratio 0.15 `
  --temp 10 `
  --smooth-size 30 `
  --smooth-sigma 30 `
  --fuse-direct-segment-score `
  --direct-segment-weight 0.1
```

Expected output:

```text
AUC=0.846424071
AP=0.591460854
videos_used=800
missing_videos=0
```

### Sequential Inference

These commands evaluate the direct segment scores using the same frame-position and gap policy, without retrieval, smoothing, fusion or position weighting:

```powershell
.\scripts\run_ucf_vad_agent_fused01.ps1 -WithoutRefinement
.\scripts\run_xd_vad_agent_fused01.ps1 -WithoutRefinement
```

Table 7 reports 72.08% AUC on UCF-Crime with sequential inference and 83.04%
AUC with full offline temporal refinement.

## Paper Ablations

Run Tables 5 through 9 from the repository root:

```powershell
$env:PYTHON = '<path-to-evaluation-env>\python.exe'
.\scripts\run_paper_ablations.ps1
```

Select individual tables using the same runner:

```powershell
.\scripts\run_paper_ablations.ps1 -Table 5,7
```

`configs/paper_ablations.json` specifies the inputs, evaluation parameters and expected metrics. Each run writes `metrics.json`, `commands.json` and logs to a unique UTC-named directory under `results/vad_detection/ablations/runs/`. The commands read cached predictions and features; no model inference is required.

### Table 5 Context and Narrative Memory Agents

| Configuration | AUC (%) |
| --- | ---: |
| Context Agent only | 77.73 |
| Previous reasoning memory | 80.96 |
| Full VAD-Agent | 83.04 |

Artifact mapping:

| Configuration | Prediction folder | Feature cache |
| --- | --- | --- |
| Context Agent only | `results/predictions/ablation_ucf_context_only` | `results/cached_features/ucf_crime_stride150` |
| Previous reasoning memory | `results/predictions/ablation_ucf_reasoning_summary_only` | `results/cached_features/ucf_crime_stride150` |
| Full VAD-Agent | `results/predictions/ucf_crime_mif_stride150` | `results/cached_features/ucf_crime_stride150` |

### Table 6 Segment Stride

| Stride (frames) | UCF-Crime AUC (%) | XD-Violence AUC (%) |
| ---: | ---: | ---: |
| 300 | 79.74 | 83.68 |
| 150 | 83.04 | 84.04 |
| 64 | 79.82 | 84.64 |

Artifact mapping:

| Dataset | Stride | Prediction folder | Feature cache |
| --- | ---: | --- | --- |
| UCF-Crime | 300 | `results/predictions/ucf_crime_mif_stride300` | `results/cached_features/ucf_crime_stride300` |
| UCF-Crime | 150 | `results/predictions/ucf_crime_mif_stride150` | `results/cached_features/ucf_crime_stride150` |
| UCF-Crime | 64 | `results/predictions/ucf_crime_mif_stride64` | `results/cached_features/ucf_crime_stride64` |
| XD-Violence | 300 | `results/predictions/xd_violence_mif_stride300` | `results/cached_features/xd_violence_stride300` |
| XD-Violence | 150 | `results/predictions/xd_violence_mif_stride150` | `results/cached_features/xd_violence_stride150` |
| XD-Violence | 64 | `results/predictions/xd_violence_mif_stride64` | `results/cached_features/xd_violence_stride64` |

The existing six-row stride wrapper is also available:

```powershell
.\scripts\run_stride_ablation.ps1
```

### Table 7 Sequential Inference and Offline Temporal Refinement

| Operation | UCF-Crime AUC (%) | Gain (pp) |
| --- | ---: | ---: |
| Initial segment scores (without temporal refinement) | 72.08 | - |
| Initial + similarity retrieval | 75.70 | +3.62 |
| Initial + similarity retrieval + Gaussian smoothing | 82.31 | +6.61 |
| Refinement with position weighting | 82.83 | +0.53 |
| Full refinement with direct-score fusion | 83.04 | +0.21 |

All rows use `results/predictions/ucf_crime_mif_stride150` and `results/cached_features/ucf_crime_stride150` with the same frame mapping. Gains are calculated from unrounded AUC values. The initial row uses the direct evaluator. Retrieval-only evaluation uses a smoothing width of one; the Gaussian row uses width 15 and sigma 10. Position weighting and the 0.1 direct-score fusion are enabled only for their respective rows.

### Table 8 Narrative Agent Backbone

| Narrative agent | Reasoning agent | AUC (%) |
| --- | --- | ---: |
| LLaMA | InternVL2-8B | 79.36 |
| InternVL2-8B | InternVL2-8B | 83.04 |

| Narrative agent | Prediction folder | Feature cache |
| --- | --- | --- |
| LLaMA | `results/predictions/ablation_ucf_llama_narrative` | `results/cached_features/ucf_crime_stride150` |
| InternVL2-8B | `results/predictions/ucf_crime_mif_stride150` | `results/cached_features/ucf_crime_stride150` |

### Table 9 Sampling Assignment on the HIVAU UCF Subset

All three rows use the same 60-video list, `data/annotations/hivau_test_ucf_crime_videos.txt`, and the shared `results/cached_features/ucf_crime_stride150` feature cache.

| Variant | Context frames | Reasoning frames | AUC (%) |
| --- | ---: | ---: | ---: |
| Uniform for both agents | 8 | 8 | 83.98 |
| MIF for both agents | 8 | 8 | 85.48 |
| Uniform context + MIF reasoning | 8 | 8 | 86.86 |

| Variant | Prediction folder |
| --- | --- |
| Uniform for both agents | `results/predictions/ablation_ucf_uniform8_uniform8` |
| MIF for both agents | `results/predictions/ablation_ucf_mif8_mif8` |
| Uniform context + MIF reasoning | `results/predictions/ablation_ucf_uniform8_mif8` |

The subset evaluator is `src/vad_detection/evaluation/compute_hivau_subset_auc.py`. The paper runner invokes it with `--projection-mode stride --gap-policy hold --strict-alignment`, retrieval ratio 0.10, temperature 10, Gaussian width 15, sigma 10, position weighting, and direct-score weight 0.1.

## Full UCF Sampling Evaluation

This additional sampling evaluation uses all 290 UCF-Crime test videos rather than the paper's 60-video subset.

| Variant | Context frames | Reasoning frames | AUC |
| --- | ---: | ---: | ---: |
| Uniform for both agents | 8 | 8 | 0.792794417 |
| MIF for both agents | 8 | 8 | 0.802549996 |
| Uniform context + MIF reasoning | 8 | 8 | 0.800478823 |
| Uniform context + MIF reasoning | 16 | 8 | 0.830438160 |

Run the full-UCF sampling configurations with:

```powershell
.\scripts\run_ucf_sampling_ablation.ps1
```

| Variant | Prediction folder | Feature cache |
| --- | --- | --- |
| Uniform for both agents, 8/8 | `results/predictions/ablation_ucf_uniform8_uniform8` | `results/cached_features/ucf_crime_stride150` |
| MIF for both agents, 8/8 | `results/predictions/ablation_ucf_mif8_mif8` | `results/cached_features/ucf_crime_stride150` |
| Uniform context + MIF reasoning, 8/8 | `results/predictions/ablation_ucf_uniform8_mif8` | `results/cached_features/ucf_crime_stride150` |
| Uniform context + MIF reasoning, 16/8 | `results/predictions/ucf_crime_mif_stride150` | `results/cached_features/ucf_crime_stride150` |

Generation code:

```text
src/vad_detection/inference/ablation_ucf_uniform8_uniform8.py
src/vad_detection/inference/ablation_ucf_mif8_mif8.py
src/vad_detection/inference/generate_ucf_mif_stride150.py
```

The 8/8 uniform-context/MIF-reasoning configuration uses the main UCF generator with `--context-k 8 --reasoning-k 8` and a separate output folder.

## HIVAU-70K VAU

The reported BLEU score sums BLEU-1 through BLEU-4. The saved metric files
use `PaScore` as the key for ParaScore.

Main VAD-Agent prediction builder:

```text
src/vau_hivau/inference/build_vad_agent_hivau_video_predictions.py
```

Semantic evaluation:

```powershell
.\scripts\evaluate_hivau_semantics.ps1
```

Saved metric tables:

```text
results/vau_hivau/vad_agent/
results/vau_hivau/vad_agent_holmes/
```

## Holmes Integration

Holmes VAU code:

```text
src/holmes_integration/vau/
```

Holmes detection wrappers:

```powershell
src\holmes_integration\detection\run_ucf_mif_holmes_expmax128.ps1
src\holmes_integration\detection\run_xd_mif_holmes_linear.ps1
```

Cached Holmes artifacts:

```text
results/holmes_mif/
results/holmes_integration/detection/
```

## VADTree Generalization

VAD-Agent integration with VADTree:

```text
src/vadtree_generalization/
```

Reported result JSONs:

```text
results/vadtree_generalization/vadtree_raw_fused_beta0.2.json
results/vadtree_generalization/ours_raw_fused_beta0.2.json
```

ImageBind is used only for selecting MIF visual evidence inside VADTree fine-grained nodes. It is not used as the final anomaly scorer.


