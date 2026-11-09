#!/bin/zsh
# Resumable IEMOCAP check of two ideas on test session 5, 3 seeds each, compared with the existing runs:
#   1. Whisper large v3 encoder instead of frozen WavLM (baseline asr_ctx_wavlm).
#   2. Prosody descriptors in the text input (baselines asr_ctx and asr_ctx_wavlm).
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_sota_ideas.sh
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
step 21_combined 21_combined_iemocap_whisper $M/combined_asr_ctx_whisper/metrics_test.json "" ${=I} \
  --set 'AUDIO="whisper"' --set 'SEEDS=[0, 1, 2]' --set 'RUNS=["asr_ctx_e2v"]'
step 21_combined 21_combined_iemocap_pros $M/combined_asr_ctx_pros_wavlm/metrics_test.json "" ${=I} \
  --set 'AUDIO="wavlm"' --set 'SEEDS=[0, 1, 2]' --set 'RUNS=["asr_ctx_pros", "asr_ctx_pros_e2v"]'

# Leave-one-session-out folds 1-4, seed 0 like the existing CV baselines (fold 5 is the session-5 run above).
for k in 1 2 3 4; do
  S="--set TEST_SESSION=$k"
  step 21_combined 21_combined_iemocap_whisper_f$k $M/combined_asr_ctx_whisper_f$k/metrics_test.json "" ${=I} ${=S} \
    --set 'AUDIO="whisper"' --set 'SEEDS=[0]' --set 'RUNS=["asr_ctx_e2v"]'
  step 21_combined 21_combined_iemocap_pros_f$k $M/combined_asr_ctx_pros_wavlm_f$k/metrics_test.json "" ${=I} ${=S} \
    --set 'AUDIO="wavlm"' --set 'SEEDS=[0]' --set 'RUNS=["asr_ctx_pros", "asr_ctx_pros_e2v"]'
done
echo "ALL DONE $(date +%H:%M)"
