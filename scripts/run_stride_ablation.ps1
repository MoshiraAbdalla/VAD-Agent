$PY = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

Write-Output "`n=== UCF stride 300 ==="
& $PY -B src\vad_detection\evaluation\compute_ucf_temporal_refinement.py `
  --ann-root data\annotations\UCF_Eval.json `
  --score-folder results\predictions\ucf_crime_mif_stride300 `
  --vision-folder results\cached_features\ucf_crime_stride300 `
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "`n=== UCF stride 150 ==="
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "`n=== UCF stride 64 ==="
& $PY -B src\vad_detection\evaluation\compute_ucf_temporal_refinement.py `
  --ann-root data\annotations\UCF_Eval.json `
  --score-folder results\predictions\ucf_crime_mif_stride64 `
  --vision-folder results\cached_features\ucf_crime_stride64 `
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "`n=== XD stride 300 ==="
& $PY -B src\vad_detection\evaluation\compute_xd_temporal_refinement.py `
  --ann-root data\annotations\xd_test_800.json `
  --score-folder results\predictions\xd_violence_mif_stride300 `
  --vision-folder results\cached_features\xd_violence_stride300 `
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "`n=== XD stride 150 ==="
& $PY -B src\vad_detection\evaluation\compute_xd_temporal_refinement.py `
  --ann-root data\annotations\xd_test_800.json `
  --score-folder results\predictions\xd_violence_mif_stride150 `
  --vision-folder results\cached_features\xd_violence_stride150 `
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "`n=== XD stride 64 ==="
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
