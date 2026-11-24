# Dataset and Checkpoint References

This repository is self-contained for evaluation from the released cached artifacts. It does not include raw video frames or large model checkpoints.

Included artifacts:

- `data/annotations/UCF_Eval.json`
- `data/annotations/xd_test_800.json`
- `data/annotations/name_gt_pair.json`
- `data/annotations/hivau_merge_instruction_test_final.jsonl`
- `results/cached_features/ucf_crime_stride300/` - UCF non-overlap stride ablation.
- `results/cached_features/ucf_crime_stride150/` - UCF main result and sampling ablations.
- `results/cached_features/ucf_crime_stride64/` - UCF stride-64 ablation.
- `results/cached_features/xd_violence_stride300/` - XD non-overlap stride ablation.
- `results/cached_features/xd_violence_stride150/` - XD stride-150 ablation.
- `results/cached_features/xd_violence_stride64/` - XD main result and stride-64 ablation.
- `results/predictions/ucf_crime_mif_stride300/` - UCF stride-300 ablation predictions.
- `results/predictions/ucf_crime_mif_stride150/` - UCF main 16/8 prediction row.
- `results/predictions/ucf_crime_mif_stride64/` - UCF stride-64 ablation predictions.
- `results/predictions/ablation_ucf_uniform8_uniform8/` - UCF sampling ablation, uniform context and uniform reasoning.
- `results/predictions/ablation_ucf_mif8_mif8/` - UCF sampling ablation, MIF context and MIF reasoning.
- `results/predictions/ablation_ucf_uniform8_mif8/` - UCF sampling ablation, 8 uniform context frames and 8 MIF reasoning frames.
- `results/predictions/xd_violence_mif_stride300/` - XD stride-300 ablation predictions.
- `results/predictions/xd_violence_mif_stride150/` - XD stride-150 ablation predictions.
- `results/predictions/xd_violence_mif_stride64/` - XD main stride-64 prediction row.
- `results/holmes_mif/` - cached Holmes MIF/ATS detection artifacts.

External assets needed only for rerunning model inference:

- UCF-Crime extracted frames: `data/ucf/frames` or your local equivalent.
- XD-Violence extracted frames: `data/xd/frames` or your local equivalent.
- InternVL2-8B for VAD-Agent detection and HIVAU VAU inference.
- HolmesVAU-2B and `holmesvau/ATS/anomaly_scorer.pth` for Holmes integration.
- LLaVA-Video-7B-Qwen2, LLaMA, and ImageBind for VADTree generalization.
- `roberta-large` for BERTScore and PaScore during semantic evaluation.

Use `configs/paths.example.json` as a local path template. Do not commit raw datasets or checkpoints.
