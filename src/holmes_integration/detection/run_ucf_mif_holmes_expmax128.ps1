$PY = if ($env:HOLMES_PYTHON) { $env:HOLMES_PYTHON } elseif ($env:PYTHON) { $env:PYTHON } else { 'python' }
$ROOT = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ROOT

New-Item -ItemType Directory -Force -Path results\holmes_integration\detection | Out-Null

& $PY src\holmes_integration\detection\compute_full_dataset_mif_uniform_union_auc.py `
  --dataset ucf `
  --ckpt HolmesVAU-2B `
  --sampler_path holmesvau\ATS\anomaly_scorer.pth `
  --device cuda:0 `
  --ann_root data\annotations\UCF_Eval.json `
  --ucf_root data\ucf\videos `
  --mif_frame_scores_dir results\holmes_mif\ucf\holmescls_mif160_ats_frame_scores `
  --union_cache_dir results\holmes_mif\ucf\mif_uniform_union_frame_scores `
  --out results\holmes_integration\detection\ucf_full_mif_uniform_union_auc_expmax128.json `
  --uniform_stride 16 `
  --projection expmax128 `
  --batch_size 16 `
  --resume

