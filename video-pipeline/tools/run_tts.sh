#!/bin/zsh
# Voice one clip on the Colab CPU session `tts`: runs the code cells of tts_vieneu.ipynb with CLIP set, pulls the wavs back.
# Usage (from video-pipeline): zsh tools/run_tts.sh 01-van-de [--voices]
# The session must exist: ../venv/bin/colab new -s tts  (CPU runtime, no --gpu)
set -e
cd "${0:A:h}/.."
CLIP=$1
C=${0:A:h}/../../venv/bin/colab
# `colab exec --timeout` does not cover a dropped connection, so every call also gets a hard alarm.
tmo() { perl -e 'alarm shift; exec @ARGV' "$@"; }

# Upload answers 500 when the target directory does not exist yet.
printf 'import os\nos.makedirs("/content/tts", exist_ok=True)\nprint("ok")\n' | tmo 120 $C exec -s tts --timeout 60 | tail -1
up() { tmo 120 $C upload -s tts $1 $2 < /dev/null 2>&1 | grep -q Uploaded || { echo "upload failed: $1"; exit 1; }; }
up narration/$CLIP.json /content/tts/$CLIP.json
# Cells 1..4 are install, config, setup, synth; cell 6 renders the voice samples.
cells=(1 2 3 4)
[[ $2 == --voices ]] && cells+=(6)
# The cells run as a plain script in a fresh process: a kernel that already imported a newer onnxruntime
# keeps it after the pin is installed, and fails to load the model.
mkdir -p build
python3 - "$CLIP" $cells > build/tts_$CLIP.py <<'EOF'
import json, re, sys
nb = json.load(open('tts_vieneu.ipynb'))
clip, idx = sys.argv[1], [int(i) for i in sys.argv[2:]]
out = ['import subprocess']
for i in idx:
    src = ''.join(nb['cells'][i]['source']).replace("CLIP = '01-van-de'", f"CLIP = {clip!r}")
    out += [re.sub(r'^!(.*)$', lambda m: f'subprocess.run(f"""{m.group(1)}""", shell=True, check=True)', l) for l in src.split('\n')]
print('\n'.join(out))
EOF
up build/tts_$CLIP.py /content/tts/run_$CLIP.py
printf 'import subprocess\nr = subprocess.run(["python3", "/content/tts/run_%s.py"], capture_output=True, text=True)\nprint(r.stdout[-4000:]); print(r.stderr[-2500:])\n' $CLIP \
  | tmo 900 $C exec -s tts --timeout 840 2>&1 | grep -v "^\s*$" | grep -v "it/s\]" | tail -40

mkdir -p audio
tmo 300 $C download -s tts /content/tts/$CLIP.tgz audio/$CLIP.tgz < /dev/null | tail -1
tar -xzf audio/$CLIP.tgz -C audio && rm audio/$CLIP.tgz
if [[ $2 == --voices ]]; then
  tmo 300 $C download -s tts /content/tts/voices.tgz audio/voices.tgz < /dev/null | tail -1
  tar -xzf audio/voices.tgz -C audio && rm audio/voices.tgz
fi
echo "done $(date +%H:%M)"
