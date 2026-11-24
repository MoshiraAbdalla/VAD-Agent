param([switch]$WithoutRefinement)

$PY = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$ROOT = Split-Path -Parent $PSScriptRoot
cd $ROOT

if ($WithoutRefinement) {
    & $PY -B src\vad_detection\evaluation\compute_direct_segment_frame_metrics.py `
      --dataset ucf `
      --ann-root data\annotations\UCF_Eval.json `
      --score-folder results\predictions\ucf_crime_mif_stride150 `
      --vision-folder results\cached_features\ucf_crime_stride150 `
      --projection-mode stride `
      --gap-policy hold `
      --strict-alignment
} else {
    & $PY -B src\vad_detection\evaluation\compute_ucf_temporal_refinement.py `
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
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
