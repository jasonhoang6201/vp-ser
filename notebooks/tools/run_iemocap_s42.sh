#!/bin/zsh
# IEMOCAP, seed 42, leave-one-session-out over 5 folds (session 5 first). Resumable: rerun the same command.
#   Per fold: 10 SenseVoice zero-shot + fine-tune -> 20 transcripts for every turn -> 21 Whisper-audio system and
#   its ablations -> 21 WavLM-audio system. Then 15 Whisper turbo zero-shot + fine-tune on every fold (WER only).
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_s42.sh
RUNS_REPO=jasonhoang6201/iemocap-runs-s42
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
K=cache/iemocap/combined

for k in 5 1 2 3 4; do
  f=_f$k; [[ $k == 5 ]] && f=""
  S="--set TEST_SESSION=$k"
  csv=cache/iemocap/asr_sensevoice_ft$f.csv
  # Fold checkpoints are not synced to HF (2.7 GB each): if the VM was replaced between 10 and 20, redo 10.
  if [[ ! -f $NB/$csv && -f $NB/$M/sensevoice_small_finetune$f/metrics_test.json ]] && \
     [[ $(vm "import os\nprint('@@CKPT', os.path.exists('$R/checkpoints/sensevoice_iemocap$f/model.pt'))\n" 60) != *"@@CKPT True"* ]]; then
    rm $NB/$M/sensevoice_small_finetune$f/metrics_test.json
  fi
  step 10_sensevoice 10_sv_s42$f $M/sensevoice_small_finetune$f/metrics_test.json "" ${=I} ${=S}
  step 20_sv_transcripts 20_sv_s42$f $csv "" ${=I} ${=S}
  step 21_combined 21_whisper_s42$f $K/gt_ctx${f}_s42.npz "" ${=I} ${=S} --set 'AUDIO="whisper"'
  step 21_combined 21_wavlm_s42$f $K/asr_ctx_wavlm${f}_s42.npz "" ${=I} ${=S} \
    --set 'AUDIO="wavlm"' --set 'RUNS=["asr_ctx_e2v"]'
done

for k in 5 1 2 3 4; do
  f=_f$k; [[ $k == 5 ]] && f=""
  step 15_whisper_ft 15_whisper_s42$f $M/whisper_turbo_finetune$f/metrics_test.json "" ${=I} --set TEST_SESSION=$k
done
echo "ALL DONE $(date +%H:%M)"
