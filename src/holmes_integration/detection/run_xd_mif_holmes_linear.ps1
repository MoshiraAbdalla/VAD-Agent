$PY = if ($env:HOLMES_PYTHON) { $env:HOLMES_PYTHON } elseif ($env:PYTHON) { $env:PYTHON } else { 'python' }
$ROOT = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $ROOT

New-Item -ItemType Directory -Force -Path results\holmes_integration\detection | Out-Null

& $PY src\holmes_integration\detection\compute_full_dataset_mif_uniform_union_auc.py `
  --dataset xd `
  --ckpt HolmesVAU-2B `
  --sampler_path holmesvau\ATS\anomaly_scorer.pth `
  --device cuda:0 `
  --ann_root data\annotations\xd_test_800.json `
  --xd_root data\xd\videos `
  --name_gt_pair data\annotations\name_gt_pair.json `
  --mif_frame_scores_dir results\holmes_mif\xd\holmescls_mif160_ats_frame_scores `
  --union_cache_dir results\holmes_mif\xd\mif_uniform_union_frame_scores `
  --out results\holmes_integration\detection\xd_full_mif_uniform_union_auc_linear.json `
  --uniform_stride 16 `
  --projection linear `
  --batch_size 16 `
  --resume

