#!/bin/zsh
# Resumable official-IEMOCAP chain (test session 5): 10 -> 20 -> 21 (wavlm) -> 12 -> 21 (e2v).
# Safe to rerun after the VM is reclaimed; notebook 21 also skips every seed run whose npz was already pushed.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_official.sh
source "${0:A:h}/chain_lib.sh"
setup

step 10_sensevoice 10_sensevoice_iemocap_off $M/sensevoice_small_finetune/metrics_test.json --ckpt \
  --set 'DATASET="iemocap"'
step 20_sv_transcripts 20_sv_transcripts_iemocap_off cache/iemocap/asr_sensevoice_ft.csv "" --set 'DATASET="iemocap"'
step 21_combined 21_combined_iemocap_off_wavlm $M/combined_gt/metrics_test.json "" --set 'DATASET="iemocap"' \
  --set 'AUDIO="wavlm"' --set 'RUNS=["asr_ctx_e2v", "asr_ctx", "asr_e2v", "asr", "gt_ctx", "gt2asr", "gt"]'
step 12_cascade_fusion 12_cascade_fusion_iemocap_off $M/cascade_fusion_gated_gt/metrics_test.json "" \
  --set 'DATASET="iemocap"' --set 'SEEDS=[0, 1, 2]'
step 21_combined 21_combined_iemocap_off_e2v $M/combined_asr_ctx_e2v/metrics_test.json "" --set 'DATASET="iemocap"' \
  --set 'RUNS=["asr_e2v", "asr_ctx_e2v"]'
echo "ALL DONE $(date +%H:%M)"
