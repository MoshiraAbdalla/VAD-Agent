# VADTree Generalization

This module contains the standalone VAD-Agent integration used for the paper's VADTree generalization table.

The experiment reuses VADTree HGTree temporal decomposition and injects VAD-Agent on the fine-grained nodes:

- Context agent: LLaVA-Video-7B-Qwen2 over uniformly sampled frames.
- Reasoning agent: LLaVA-Video-7B-Qwen2 over MIF-selected visual evidence.
- Narrative agent: LLaMA text memory over the temporal sequence.
- MIF selector: ImageBind features are used only to choose anomaly-focused frames inside each HGTree node.

Primary entry points:

```powershell
python -m src.vadtree_generalization.run_agentic_mif --help
python -m src.vadtree_generalization.export_agentic_vid_score --help
python -m src.vadtree_generalization.fuse_coarse_fine_scores --help
```

Reported result artifacts are stored in:

```text
results/vadtree_generalization/vadtree_raw_fused_beta0.2.json
results/vadtree_generalization/ours_raw_fused_beta0.2.json
```

The reported table uses raw coarse-fine fused HGTree scores with `beta=0.2`, without extra temporal refinement.

