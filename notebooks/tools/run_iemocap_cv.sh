#!/bin/zsh
# Resumable IEMOCAP follow-up chain:
#   1. WavLM cascade baseline on test session 5 (3 seeds), the leak-free audio counterpart of the e2v cascade.
#   2. Leave-one-session-out folds 1-4 (fold 5 is the existing session-5 run), 1 seed each: per fold SenseVoice is
#      fine-tuned again without the test session, so test transcripts never come from a model that heard that session.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_cv.sh
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
step 12_cascade_fusion 12_cascade_iemocap_wavlm $M/cascade_fusion_gated_gt_wavlm/metrics_test.json "" \
  ${=I} --set 'AUDIO="wavlm"' --set 'SEEDS=[0, 1, 2]'

for k in 1 2 3 4; do
  S="--set TEST_SESSION=$k"
  csv=cache/iemocap/asr_sensevoice_ft_f$k.csv
  # Fold checkpoints are not synced to HF (2.7 GB each): if the VM was replaced between 10 and 20, redo 10.
  if [[ ! -f $NB/$csv && -f $NB/$M/sensevoice_small_finetune_f$k/metrics_test.json ]] && \
     [[ $(vm "import os\nprint('@@CKPT', os.path.exists('$R/checkpoints/sensevoice_iemocap_f$k/model.pt'))\n" 60) != *"@@CKPT True"* ]]; then
    rm $NB/$M/sensevoice_small_finetune_f$k/metrics_test.json
  fi
  step 10_sensevoice 10_sensevoice_iemocap_f$k $M/sensevoice_small_finetune_f$k/metrics_test.json "" ${=I} ${=S}
  step 20_sv_transcripts 20_sv_transcripts_iemocap_f$k $csv "" ${=I} ${=S}
  step 21_combined 21_combined_iemocap_f$k $M/combined_gt_f$k/metrics_test.json "" ${=I} ${=S} \
    --set 'AUDIO="wavlm"' --set 'SEEDS=[0]' \
    --set 'RUNS=["asr_ctx_e2v", "asr_ctx", "asr_e2v", "asr", "gt_ctx", "gt"]'
  step 12_cascade_fusion 12_cascade_iemocap_f$k $M/cascade_fusion_gated_gt_wavlm_f$k/metrics_test.json "" ${=I} ${=S} \
    --set 'AUDIO="wavlm"' --set 'SEEDS=[0]'
done
echo "ALL DONE $(date +%H:%M)"
