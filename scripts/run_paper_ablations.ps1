param(
    [ValidateSet(5, 6, 7, 8, 9)]
    [int[]]$Table = @(5, 6, 7, 8, 9),
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { 'python' }),
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Jobs = Get-Content -LiteralPath (Join-Path $ProjectRoot 'configs\paper_ablations.json') -Raw | ConvertFrom-Json
$Selected = @($Jobs | Where-Object { $Table -contains [int]$_.table })
if ($Selected.Count -eq 0) { throw 'No paper ablations selected.' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $ProjectRoot 'results\vad_detection\ablations\runs' }
if (-not [IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path (Get-Location).Path $OutputRoot }
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$RunDirectory = Join-Path $OutputRoot ([DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffffff'))
New-Item -ItemType Directory -Path $RunDirectory | Out-Null
$Metrics = @()
$Commands = @()
Push-Location -LiteralPath $ProjectRoot
try {
    foreach ($Job in $Selected) {
        $EvaluationArguments = @('-B', [string]$Job.evaluator) + @($Job.arguments)
        if ($Job.mode -eq 'subset') {
            $EvaluationArguments += @('--out', (Join-Path $RunDirectory ($Job.id + '_native.json')))
        }
        $Commands += @{ Table = $Job.table; Id = $Job.id; Python = $Python; Arguments = $EvaluationArguments }
        Write-Output ('Evaluating paper Table ' + $Job.table + ': ' + $Job.id)
        $Lines = @(& $Python @EvaluationArguments | Tee-Object -FilePath (Join-Path $RunDirectory ($Job.id + '.log')))
        $Code = $LASTEXITCODE
        $Lines | Write-Output
        if ($Code -ne 0) { throw "Evaluation failed for $($Job.id), exit code $Code." }
        $Parsed = @{}
        foreach ($Line in $Lines) {
            if ([string]$Line -match '^(AUC|AP|videos_used|missing_videos)=([0-9.]+)$') {
                $Parsed[$Matches[1]] = $Matches[2]
            }
        }
        if (-not $Parsed.ContainsKey('AUC') -or -not $Parsed.ContainsKey('videos_used') -or -not $Parsed.ContainsKey('missing_videos')) {
            throw "Could not read metrics/coverage for $($Job.id)."
        }
        if ([int]$Parsed['videos_used'] -ne [int]$Job.expected_videos -or [int]$Parsed['missing_videos'] -ne 0) {
            throw "Incomplete video coverage for $($Job.id); refusing to report a partial paper result."
        }
        $Auc = [double]::Parse($Parsed['AUC'], [Globalization.CultureInfo]::InvariantCulture)
        $Ap = if ($Parsed.ContainsKey('AP')) { [double]::Parse($Parsed['AP'], [Globalization.CultureInfo]::InvariantCulture) } else { $null }
        $Metrics += [ordered]@{ table = [int]$Job.table; id = [string]$Job.id; dataset = [string]$Job.dataset; auc = $Auc; ap = $Ap; videos_used = [int]$Parsed['videos_used']; missing_videos = 0 }
    }
}
finally {
    $Commands | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunDirectory 'commands.json') -Encoding UTF8
    $Metrics | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunDirectory 'metrics.json') -Encoding UTF8
    Pop-Location
    Write-Output ('results_directory=' + $RunDirectory)
}
