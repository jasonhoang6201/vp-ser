# Shared helpers for resumable Colab chains (sourced by run_iemocap_*.sh).
# State comes back from the private HF repos on a fresh VM, finished steps are skipped (a step counts as finished
# once its last output has been pulled to notebooks/), and a rerun attaches to a step's job if it is still alive.
cd "${0:A:h}/../.."
C=venv/bin/colab
NB=notebooks
R=/content/vp-ck
M=results/iemocap

# `colab exec --timeout` does not cover a dropped connection (seen hanging 10+ min), so wrap every call in an alarm.
tmo() { perl -e 'alarm shift; exec @ARGV' "$@"; }
vm() { printf "$1" | tmo $(( ${2:-600} + 60 )) $C exec -s eaasr --timeout ${2:-600} 2>&1 | tail -${3:-1}; }
# A fresh VM drops the first calls now and then: retry until the output shows `marker`, else stop the chain.
need() {
  local code=$1 secs=$2 marker=$3 out
  for t in 1 2 3 4 5; do
    out=$(vm "$code" $secs 20)
    if echo "$out" | grep -q "$marker"; then echo "$out" | tail -${4:-1}; return 0; fi
    sleep 15
  done
  echo "SETUP FAILED ($marker): rerun this script"; exit 1
}
up() {
  for t in 1 2 3 4 5; do
    tmo 300 $C upload -s eaasr $1 $2 < /dev/null 2>&1 | grep -q Uploaded && return 0
    sleep 10
  done
  echo "SETUP FAILED (upload $1): rerun this script"; exit 1
}
pull() {
  # A dropped connection here would fail a step whose job already finished, so retry before giving up.
  # Download only after a confirmed tar: the previous step's archive is still on the VM and would pass for this one.
  local out
  for t in 1 2 3 4 5; do
    out=$(vm "import subprocess\nsubprocess.run('cd $R && tar -czf /content/iem_out.tgz --exclude=*.npy --exclude=wavlm_*.npz --exclude=whisper_*.npz --exclude=e2v_*.npz results/iemocap cache/iemocap 2>/dev/null', shell=True)\nprint('tar ok')\n" 300)
    echo "$out"
    if [[ $out == *"tar ok"* ]]; then
      $C download -s eaasr /content/iem_out.tgz $NB/iem_out.tgz < /dev/null 2>&1 | tail -1
      tar -xzf $NB/iem_out.tgz -C $NB && rm $NB/iem_out.tgz && echo "pulled after $1 $(date +%H:%M)" && return 0
    fi
    sleep 30
  done
}
# Detached and polled with short calls: a single long exec has hung for 16+ min after the upload had already ended.
push() {
  local out started=0
  for t in 1 2 3 4 5; do
    [[ $(push_bg "--wait $1") == *"bg push"* ]] && { started=1; break; }
    sleep 20
  done
  [[ $started == 1 ]] || { echo "push not started"; return; }
  for t in {1..20}; do
    sleep 20
    out=$(vm "import subprocess\nprint('@@PUSHING', subprocess.run(['pgrep', '-f', 'vm_hf_sync.py push'], capture_output=True).returncode == 0)\n" 45)
    [[ $out == *"@@PUSHING False"* ]] && { echo "pushed"; return; }
  done
  echo "push not confirmed"
}
# Detached on the VM so a slow upload never blocks the kernel that colab_run polls through.
push_bg() { vm "import subprocess\nsubprocess.Popen('python3 $R/tools/vm_hf_sync.py push $1 > /content/push.log 2>&1', shell=True, start_new_session=True)\nprint('bg push')\n" 60; }

# step NOTEBOOK LOGNAME MARKER CKPT_FLAG [--set ...]
# marker: a file under notebooks/ that only exists once the step has finished and been pulled.
step() {
  local name=$1 log=$2 marker=$3 ckpt=$4; shift 4
  if [[ -f $NB/$marker ]]; then echo "skip $log (done)"; return; fi
  echo "start $log $(date +%H:%M)"
  # Push every 10 min while the step runs, so a reclaimed VM loses at most 10 min of finished seed runs.
  ( while sleep 600; do push_bg > /dev/null; done ) &
  local syncer=$!
  # A rerun after a local network drop finds the job still alive on the VM: attach instead of starting it over.
  # Earlier steps are skipped by their markers, so a live job with this name belongs to this step.
  local job=(src/$name.py --bg "$@")
  [[ $(vm "import os\ntry:\n    pid = int(open('$R/jobs/$name.pid').read())\n    print('@@ALIVE', open(f'/proc/{pid}/stat').read().split()[2] != 'Z')\nexcept Exception:\n    print('@@ALIVE False')\n" 60) == *"@@ALIVE True"* ]] \
    && job=(--attach $name) && echo "attach $log"
  # 4 h cap: the longest step takes ~2.5 h, so a step still polling after that is tailing a dead VM.
  (cd $NB && tmo 14400 python3 tools/colab_run.py -s eaasr --log runs/${log}.log $job > runs/${log}.out 2>&1)
  kill $syncer 2>/dev/null
  grep -aq "Traceback" $NB/runs/${log}.log && echo "TRACEBACK in $log"
  push $ckpt
  pull $log
  if [[ ! -f $NB/$marker ]]; then
    echo "STEP FAILED $log $(date +%H:%M): rerun this script to resume"; exit 1
  fi
  echo "done $log $(date +%H:%M)"
}

setup() {
  $C status -s eaasr < /dev/null 2>&1 | grep -q "not found" && $C new -s eaasr --gpu L4 < /dev/null 2>&1 | tail -1
  need "import os\nfor d in ['$R/notebooks/eaasr', '$R/tools', '/root/.cache/huggingface']: os.makedirs(d, exist_ok=True)\nprint('dirs ok')\n" 60 "dirs ok"
  for f in $NB/eaasr/*.py; do up $f $R/notebooks/eaasr/$(basename $f); done
  up $NB/tools/vm_hf_sync.py $R/tools/vm_hf_sync.py
  up ~/.cache/huggingface/token /root/.cache/huggingface/token
  echo "files + token ok"
  need "import subprocess\nr = subprocess.run('python3 $R/tools/vm_hf_sync.py restore', shell=True, capture_output=True, text=True)\nprint(r.stdout[-600:], r.stderr[-300:])\n" 1200 "RESTORED" 8
  echo "setup done $(date +%H:%M)"
}
