"""Keep a Colab VM's IEMOCAP state in private HuggingFace dataset repos, so a reclaimed VM can resume.

  python3 vm_hf_sync.py restore          data tar + manifest, then every saved cache/result/checkpoint
  python3 vm_hf_sync.py push [--ckpt] [--wait]   upload cache/results (+ SenseVoice-FT checkpoint with --ckpt)

Both repos must stay private: IEMOCAP's license forbids redistribution.
"""
import fcntl
import os
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = "/content/vp-ck"
DATA_REPO = "jasonhoang6201/iemocap-16k"
RUNS_REPO = "jasonhoang6201/iemocap-runs"
DIRS = ["cache/iemocap", "results/iemocap"]
CKPT = "checkpoints/sensevoice_iemocap"

api = HfApi()


def restore():
    os.makedirs(f"{ROOT}/data", exist_ok=True)
    if not os.path.exists(f"{ROOT}/data/iemocap_16k.tar"):
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=f"{ROOT}/data")
    if api.repo_exists(RUNS_REPO, repo_type="dataset"):
        snapshot_download(RUNS_REPO, repo_type="dataset", local_dir=ROOT)
    for d in DIRS + [CKPT]:
        p = f"{ROOT}/{d}"
        print(d, len(os.listdir(p)) if os.path.isdir(p) else "-")
    assert os.path.getsize(f"{ROOT}/data/iemocap_16k.tar") > 1_000_000_000
    print("RESTORED")


def push(ckpt, wait):
    # Periodic background pushes skip if one is still running; the final push of a step waits for it.
    lock = open("/content/push.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
        print("push already running")
        return
    api.create_repo(RUNS_REPO, repo_type="dataset", private=True, exist_ok=True)
    assert api.dataset_info(RUNS_REPO).private, "runs repo is public, refusing to upload"
    for d in DIRS + ([CKPT] if ckpt else []):
        if os.path.isdir(f"{ROOT}/{d}"):
            # Embedding matrices in .npy are rebuilt in minutes and would dominate the upload.
            api.upload_folder(folder_path=f"{ROOT}/{d}", path_in_repo=d, repo_id=RUNS_REPO, repo_type="dataset",
                              ignore_patterns=["*.npy"], commit_message=f"sync {d}")
    print("pushed", "with ckpt" if ckpt else "")


if __name__ == "__main__":
    restore() if sys.argv[1] == "restore" else push("--ckpt" in sys.argv, "--wait" in sys.argv)
