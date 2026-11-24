$env:HF_HUB_OFFLINE='0'
$env:TRANSFORMERS_OFFLINE='0'
$PY = if ($env:HOLMES_PYTHON) { $env:HOLMES_PYTHON } else { 'D:\envs\holmesvau\python.exe' }
$ROOT = Split-Path -Parent $PSScriptRoot
cd $ROOT

function Invoke-HivauSemanticEval {
  param(
    [string] $Predictions,
    [string] $Out,
    [string] $TableCsv,
    [string] $TableMd,
    [string] $SyncedPredictions
  )

  & $PY src\vau_hivau\evaluation\compute_hivau_question_semantic_table.py `
    --predictions $Predictions `
    --instructions data\annotations\hivau_merge_instruction_test_final.jsonl `
    --out $Out `
    --table_csv $TableCsv `
    --table_md $TableMd `
    --synced_predictions_out $SyncedPredictions `
    --meteor_timeout 300 `
    --bertscore `
    --bertscore_model roberta-large `
    --parascore `
    --parascore_model roberta-large `
    --parascore_mode similarity `
    --cider_tokenizer simple
}

Invoke-HivauSemanticEval `
  -Predictions results\vau_hivau\vad_agent\mif_hivau_video_original_task_predictions.json `
  -Out results\vau_hivau\vad_agent\mif_hivau_video_original_task_semantic_table.json `
  -TableCsv results\vau_hivau\vad_agent\mif_hivau_video_original_task_semantic_table.csv `
  -TableMd results\vau_hivau\vad_agent\mif_hivau_video_original_task_semantic_table.md `
  -SyncedPredictions results\vau_hivau\vad_agent\mif_hivau_video_original_task_predictions_task_synced.json

Invoke-HivauSemanticEval `
  -Predictions results\vau_hivau\vad_agent_holmes\hivau_video_question_context_summary_mafats_taskhybrid_predictions.json `
  -Out results\vau_hivau\vad_agent_holmes\hivau_video_question_context_summary_mafats_taskhybrid_semantic_table.json `
  -TableCsv results\vau_hivau\vad_agent_holmes\hivau_video_question_context_summary_mafats_taskhybrid_semantic_table.csv `
  -TableMd results\vau_hivau\vad_agent_holmes\hivau_video_question_context_summary_mafats_taskhybrid_semantic_table.md `
  -SyncedPredictions results\vau_hivau\vad_agent_holmes\hivau_video_question_context_summary_mafats_taskhybrid_predictions_task_synced.json
