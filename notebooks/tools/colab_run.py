"""Run a notebook source (src/*.py, percent format) on a Colab CLI session, cell by cell.

Without Drive: data, caches and results live under /content/vp-ck on the VM, so the
same session must be reused across notebooks, and results downloaded before `colab stop`.

Usage:
  python notebooks/tools/colab_run.py src/12_cascade_fusion.py -s eaasr --bg   # long jobs: detached on the VM
  python notebooks/tools/colab_run.py src/12_cascade_fusion.py -s eaasr --set N_TEST=300
  python notebooks/tools/colab_run.py src/13_ent.py -s eaasr --start 9      # resume from cell 9
  python notebooks/tools/colab_run.py src/13_ent.py --list
  python notebooks/tools/colab_run.py --sync -s eaasr                       # upload eaasr/ only
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_notebooks import parse  # noqa: E402

NB = Path(__file__).resolve().parents[1]
COLAB = str(NB.parent / "venv/bin/colab")
REMOTE_ROOT = "/content/vp-ck"
ENV = [f"EAASR_NB={REMOTE_ROOT}/notebooks", f"EAASR_DRIVE={REMOTE_ROOT}"]


def colab_exec(session, code, timeout):
    """Stream one cell's output; return False if the cell raised."""
    cmd = [COLAB, "exec", "-s", session, "--timeout", str(timeout)]
    for e in ENV:
        cmd += ["--env", e]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    p.stdin.write(code)
    p.stdin.close()
    failed = False
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if "Traceback (most recent call last)" in line or re.match(r"^\w*Error: ", line):
            failed = True
    return p.wait() == 0 and not failed


def sync(session):
    files = sorted((NB / "eaasr").glob("*.py"))
    code = "import os; os.makedirs('%s/notebooks/eaasr', exist_ok=True)" % REMOTE_ROOT
    colab_exec(session, code, 60)
    for f in files:
        subprocess.run([COLAB, "upload", "-s", session, str(f), f"{REMOTE_ROOT}/notebooks/eaasr/{f.name}"],
                       check=True)
    # Kernel state persists between execs, so a stale eaasr module must be reloaded.
    colab_exec(session, "import importlib, sys\nif 'eaasr.common' in sys.modules: importlib.reload(sys.modules['eaasr.common'])", 60)
    print(f"[colab_run] synced {len(files)} files")


def colab_exec_capture(session, code, timeout=120, retries=5):
    """Short exec with retries, for polling; the CLI websocket drops now and then."""
    cmd = [COLAB, "exec", "-s", session, "--timeout", str(timeout)]
    r = None
    for attempt in range(1, retries + 1):
        # Backoff so a local network outage of several minutes is ridden out instead of failing the job.
        wait = min(60, 10 * attempt)
        try:
            r = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            time.sleep(wait)
            continue
        if r.returncode == 0 and "Connection was lost" not in r.stdout + r.stderr:
            return r.stdout
        # Only this exact message means the VM is gone; DNS or socket errors look similar but are local.
        if f"Session '{session}' not found" in r.stdout + r.stderr:
            raise RuntimeError(f"session {session!r} is gone (VM reclaimed)")
        time.sleep(wait)
    if r is None:
        raise RuntimeError("colab exec timed out on every retry")
    raise RuntimeError(f"colab exec kept failing:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def run_background(session, name, cells, start, overrides, poll, local_log):
    """Run the cells as one `ipython` job on the VM, detached from the CLI connection, and tail its log."""
    parts = []
    for i, lines in enumerate(cells):
        if i < start:
            continue
        code = "\n".join(lines)
        for k, expr in overrides.items():
            code = re.sub(rf"^{k}\s*=.*$", f"{k} = {expr}", code, flags=re.M)
        parts.append(f"print({f'######## cell {i}: {lines[0][:90]}'!r}, flush=True)\n{code}\n")
    parts.append("print('[JOB DONE]', flush=True)\n")
    script = "\n".join(parts)
    jobs = f"{REMOTE_ROOT}/jobs"
    ipy, log = f"{jobs}/{name}.ipy", f"{jobs}/{name}.log"
    launch = f"""
import os, subprocess
os.makedirs({jobs!r}, exist_ok=True)
open({ipy!r}, "w").write({script!r})
env = dict(os.environ, PYTHONUNBUFFERED="1", EAASR_NB="{REMOTE_ROOT}/notebooks", EAASR_DRIVE="{REMOTE_ROOT}",
           MPLBACKEND="Agg")
p = subprocess.Popen(["ipython", "--no-banner", "--colors=NoColor", {ipy!r}], stdout=open({log!r}, "wb"),
                     stderr=subprocess.STDOUT, cwd="/content", env=env, start_new_session=True)
open({jobs!r} + "/{name}.pid", "w").write(str(p.pid))
print("@@PID", p.pid)
"""
    out = colab_exec_capture(session, launch)
    pid = int(re.search(r"@@PID (\d+)", out).group(1))
    print(f"[colab_run] job {name} pid {pid}, log {log}", flush=True)
    return tail_job(session, name, pid, poll, local_log)


def attach(session, name, poll, local_log, pid=None):
    """Re-attach to a job started earlier (e.g. after the local process was killed); replays its log from the start."""
    jobs = f"{REMOTE_ROOT}/jobs"
    if pid is None:
        out = colab_exec_capture(session, f"print('@@PID', open('{jobs}/{name}.pid').read())")
        pid = int(re.search(r"@@PID (\d+)", out).group(1))
    print(f"[colab_run] attaching to job {name} pid {pid}", flush=True)
    return tail_job(session, name, pid, poll, local_log, mode="w")


def tail_job(session, name, pid, poll, local_log, mode="a"):
    log = f"{REMOTE_ROOT}/jobs/{name}.log"
    off, status = 0, None
    with open(local_log, mode) as lf:
        while True:
            poll_code = f"""
import os
f = open({log!r}, "rb"); f.seek({off}); data = f.read(400000); f.close()
print(data.decode("utf-8", "replace"), end="")
print("\\n@@OFF", {off} + len(data))
try:
    alive = open("/proc/{pid}/stat").read().split()[2] != "Z"
except FileNotFoundError:
    alive = False
print("@@ALIVE", alive)
"""
            out = colab_exec_capture(session, poll_code, retries=30)
            m_off = re.search(r"@@OFF (\d+)", out)
            alive = "@@ALIVE True" in out
            chunk = out[: out.rfind("\n@@OFF")] if m_off else ""
            if chunk:
                chunk = re.sub(r"[^\n]*\r(?=[^\n])", "", chunk)  # keep only the last tqdm refresh per line
                lf.write(chunk); lf.flush()
                sys.stdout.write(chunk); sys.stdout.flush()
            new_off = int(m_off.group(1)) if m_off else off
            if "[JOB DONE]" in chunk:
                status = 0
            if not alive and new_off == off:
                status = 1 if status is None else status
                break
            got, off = new_off - off, new_off
            if got < 400000:
                time.sleep(poll)
    print(f"\n[colab_run] job {name} {'finished' if status == 0 else 'FAILED (see log)'}", flush=True)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("-s", "--session")
    ap.add_argument("--set", action="append", default=[], help="NAME=python_expr, replaces a config line")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--timeout", type=float, default=8 * 3600)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--bg", action="store_true", help="run as a detached ipython job on the VM and tail its log")
    ap.add_argument("--name", help="job name for --bg (default: source stem)")
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--log", help="local log file for --bg")
    ap.add_argument("--attach", metavar="NAME", help="re-attach to a running/finished --bg job by name")
    ap.add_argument("--pid", type=int, help="with --attach, for jobs started before .pid files existed")
    a = ap.parse_args()

    if a.attach:
        return attach(a.session, a.attach, a.poll, a.log or str(NB / "runs" / f"{a.attach}.log"), a.pid)
    if a.sync:
        sync(a.session)
        if not a.src:
            return 0
    src = Path(a.src) if Path(a.src).exists() else NB / a.src
    cells = [lines for kind, lines in parse(src.read_text()) if kind == "code"]
    overrides = dict(s.split("=", 1) for s in a.set)
    if a.bg and not a.list:
        name = a.name or src.stem
        local_log = a.log or str(NB / "runs" / f"{name}.log")
        return run_background(a.session, name, cells, a.start, overrides, a.poll, local_log)
    for i, lines in enumerate(cells):
        if a.list:
            print(f"[{i}] {lines[0][:100]}")
            continue
        if i < a.start or (a.stop is not None and i >= a.stop):
            continue
        code = "\n".join(lines)
        for name, expr in overrides.items():
            code = re.sub(rf"^{name}\s*=.*$", f"{name} = {expr}", code, flags=re.M)
        print(f"\n######## cell {i}: {lines[0][:90]}", flush=True)
        if not colab_exec(a.session, code, a.timeout):
            print(f"\n[colab_run] cell {i} failed; resume with --start {i}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
