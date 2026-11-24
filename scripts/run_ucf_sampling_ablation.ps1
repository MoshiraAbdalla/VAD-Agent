$PY = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$common = @(
  'src\vad_detection\evaluation\compute_ucf_temporal_refinement.py',
  '--ann-root', 'data\annotations\UCF_Eval.json',
  '--vision-folder', 'results\cached_features\ucf_crime_stride150',
  '--projection-mode', 'stride',
  '--gap-policy', 'hold',
  '--strict-alignment',
  '--include-self-sim',
  '--apply-position-weight',
  '--position-weight-sigma', '-1',
  '--top-ratio', '0.10',
  '--temp', '10',
  '--smooth-size', '15',
  '--smooth-sigma', '10',
  '--fuse-direct-segment-score',
  '--direct-segment-weight', '0.1'
)

$rows = @(
  @{Name='Uniform for both agents (8/8)'; Folder='results\predictions\ablation_ucf_uniform8_uniform8'},
  @{Name='MIF for both agents (8/8)'; Folder='results\predictions\ablation_ucf_mif8_mif8'},
  @{Name='Uniform context + MIF reasoning (8/8)'; Folder='results\predictions\ablation_ucf_uniform8_mif8'},
  @{Name='Uniform context + MIF reasoning (16/8)'; Folder='results\predictions\ucf_crime_mif_stride150'}
)

foreach ($row in $rows) {
  Write-Output "`n=== $($row.Name) ==="
  & $PY -B @common --score-folder $row.Folder
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
