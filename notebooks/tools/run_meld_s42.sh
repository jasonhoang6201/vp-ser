#!/bin/zsh
# MELD (official train/dev/test, 7 classes), seed 42: the same 4 steps as IEMOCAP, on a second Colab VM.
#   0 data (once: build meld_16k.tar on the VM, keep it in a private HF repo)
#   1-2 WER: 10 SenseVoice zero-shot + fine-tune, 15 Whisper turbo zero-shot + fine-tune (+ transcripts)
#   3-4 the ASR with the lower fine-tuned WER feeds 21: Whisper-audio system and ablations, then WavLM audio.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_meld_s42.sh
DS=meld
SESSION=eaasr-meld
RUNS_REPO=jasonhoang6201/meld-runs-s42
source "${0:A:h}/chain_lib.sh"
setup

if [[ $(vm "import os\nprint('@@DATA', os.path.exists('$R/data/meld_16k.tar'))\n" 60) != *"@@DATA True"* ]]; then
  echo "start 00_data_meld $(date +%H:%M)"
  (cd $NB && tmo 14400 python3 tools/colab_run.py -s $SESSION --log runs/00_meld_s42.log src/00_data_meld.py --bg \
     > runs/00_meld_s42.out 2>&1)
  need "import subprocess\nr = subprocess.run('$SYNC push-data', shell=True, capture_output=True, text=True)\nprint(r.stdout[-400:], r.stderr[-300:])\n" 3600 "DATA PUSHED"
  echo "done 00_data_meld $(date +%H:%M)"
fi

I='--set DATASET="meld"'
K=cache/meld/combined

step 10_sensevoice 10_sv_meld_s42 $M/sensevoice_small_finetune/metrics_test.json "" ${=I}
Z=""; [[ -f $NB/$M/whisper_turbo_zeroshot/metrics_test.json ]] && Z="--set ZERO_SHOT=False"
step 15_whisper_ft 15_whisper_meld_s42 cache/meld/asr_whisper_ft.csv "" ${=I} ${=Z}

# Step 2 decides the ASR block for steps 3-4.
wer() { python3 -c "import json; print(json.load(open('$NB/$M/$1/metrics_test.json'))['asr']['wer'])"; }
sv=$(wer sensevoice_small_finetune); wt=$(wer whisper_turbo_finetune)
if (( wt <= sv )); then ASR=whisper; P=wasr; else ASR=sensevoice; P=asr; fi
echo "ASR $ASR (fine-tuned WER sensevoice $sv, whisper $wt)"
[[ $ASR == sensevoice ]] && step 20_sv_transcripts 20_sv_meld_s42 cache/meld/asr_sensevoice_ft.csv "" ${=I}

step 21_combined 21_whisper_meld_s42 $K/gt_ctx_s42.npz "" ${=I} --set "ASR=\"$ASR\"" --set 'AUDIO="whisper"'
step 21_combined 21_wavlm_meld_s42 $K/${P}_ctx_wavlm_s42.npz "" ${=I} --set "ASR=\"$ASR\"" \
  --set 'AUDIO="wavlm"' --set 'RUNS=["asr_ctx_e2v"]'
echo "ALL DONE $(date +%H:%M)"
