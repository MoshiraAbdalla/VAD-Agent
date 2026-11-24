param([switch]$WithoutRefinement)

$PY = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$ROOT = Split-Path -Parent $PSScriptRoot
cd $ROOT

if ($WithoutRefinement) {
    & $PY -B src\vad_detection\evaluation\compute_direct_segment_frame_metrics.py `
      --dataset xd `
      --ann-root data\annotations\xd_test_800.json `
      --score-folder results\predictions\xd_violence_mif_stride64 `
      --vision-folder results\cached_features\xd_violence_stride64 `
      --projection-mode stride `
      --gap-policy hold `
      --strict-alignment
} else {
    & $PY -B src\vad_detection\evaluation\compute_xd_temporal_refinement.py `
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
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
