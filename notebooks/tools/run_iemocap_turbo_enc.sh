#!/bin/zsh
# IEMOCAP, seed 42, 5 folds: frozen encoder of Whisper large-v3-turbo as the audio branch, so one Whisper checkpoint
# could serve both ASR and audio. Runs the configs that read audio (full, w/o context, w/o text) on the Whisper-FT
# transcripts of run_iemocap_whisper_asr.sh. Results: combined_wasr_ctx_whisper_turbo*, combined_wasr_whisper_turbo*,
# combined_whisper_turbo_only*.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_turbo_enc.sh
RUNS_REPO=jasonhoang6201/iemocap-runs-s42
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
K=cache/iemocap/combined

for k in 5 1 2 3 4; do
  f=_f$k; [[ $k == 5 ]] && f=""
  # The audio-only config writes no prediction npz, so its metrics file marks the step as done.
  step 21_combined 21_wasr_turbo_s42$f $M/combined_whisper_turbo_only$f/metrics_test.json "" ${=I} --set TEST_SESSION=$k \
    --set 'ASR="whisper"' --set 'AUDIO="whisper_turbo"' --set 'RUNS=["asr_ctx_e2v", "asr_e2v", "e2v_only"]'
done
echo "ALL DONE $(date +%H:%M)"
