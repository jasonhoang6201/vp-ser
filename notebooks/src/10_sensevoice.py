# gpu: A100
# %% [markdown]
# # 10. Baseline SenseVoice-Small (an-2024-sensevoice)
#
# SenseVoice-Small là một model duy nhất, trong một lần chạy sinh ra cả transcript lẫn token cảm xúc
# (`<|HAPPY|>`, `<|SAD|>`, ...). Kiến trúc: encoder non-autoregressive, train bằng CTC, khoảng 234M tham số.
#
# Hai thí nghiệm:
# - **A. Zero-shot**: dùng model gốc, không train thêm.
# - **B. Fine-tune** trên tập train của dataset (dùng công cụ train của FunASR), chọn checkpoint theo dev, đánh giá trên test.
#
# Số trong paper (IEMOCAP, zero-shot): UA 70.5 (Small). Paper không báo số trên MELD cho bản Small.
# Runtime: GPU (A100 hoặc L4 đều được).

# %%
DATASET = "meld"      # "meld" hoặc "iemocap"
RUN_FINETUNE = True
MAX_EPOCH = 10
LR = 2e-4             # mặc định của finetune.sh chính thức; nếu dev bị tệ đi sau epoch 1, thử 5e-5
TEST_SESSION = 5      # IEMOCAP: session để test (cross-validation 5 fold chạy lần lượt 1..5)
FOLD = "" if DATASET == "meld" or TEST_SESSION == 5 else f"_f{TEST_SESSION}"

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install -U funasr whisper-normalizer soundfile
# Chỉ clone để lấy model.py. Không cài requirements.txt của repo: file đó ghim numpy<=1.26, làm hỏng
# pandas/torch trên Colab (numpy 2).
!test -d /content/SenseVoice || git clone -q https://github.com/FunAudioLLM/SenseVoice /content/SenseVoice

# %%
import json, os, re, shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from eaasr.common import CKPT_DIR, LABELS, canon_emotion, evaluate, get_splits, gpu_info, stage_dataset

gpu_info()
SV_DIR = "/content/SenseVoice"
MODEL_DIR = snapshot_download("FunAudioLLM/SenseVoiceSmall", local_dir="/content/sv_model")
if not os.path.exists(f"{MODEL_DIR}/model.py"):
    shutil.copy(f"{SV_DIR}/model.py", MODEL_DIR)
os.chdir(SV_DIR)

man = stage_dataset(DATASET)
splits = get_splits(man, DATASET, test_session=TEST_SESSION)
print({k: len(v) for k, v in splits.items()})

# %%
from funasr import AutoModel

TAG = re.compile(r"<\|([^|]*)\|>")

def parse_sv(raw):
    tags = TAG.findall(raw)
    emo = next((canon_emotion(t) for t in tags if canon_emotion(t)), None)
    return TAG.sub("", raw).strip(), emo

def run_sensevoice(model, df, batch_size=64):
    res = model.generate(input=df["path"].tolist(), cache={}, language="en", use_itn=True,
                         batch_size=batch_size, ban_emo_unk=True)
    assert len(res) == len(df), (len(res), len(df))
    out = df.copy()
    parsed = [parse_sv(r["text"]) for r in res]
    out["raw"] = [r["text"] for r in res]
    out["hyp_text"] = [p[0] for p in parsed]
    out["hyp_emotion"] = [p[1] for p in parsed]
    return out

# %% [markdown]
# ## A. Zero-shot

# %%
model = AutoModel(model=MODEL_DIR, trust_remote_code=True, remote_code=f"{SV_DIR}/model.py",
                  device="cuda:0", disable_update=True)
test_zs = run_sensevoice(model, splits["test"])
print(test_zs[["text", "raw"]].head(5).to_string())
m_zs = evaluate(f"sensevoice_small_zeroshot{FOLD}", DATASET, test_zs,
                notes="SenseVoice-Small, no training, ban_emo_unk=True")

# %%
# Token cảm xúc SenseVoice sinh ra, trước khi map sang nhãn của dataset (để thấy lớp nào không có tương ứng).
test_zs["raw"].str.extract(r"<\|[^|]*\|><\|([^|]*)\|>")[0].value_counts()

# %% [markdown]
# ## B. Fine-tune
# Tạo file `jsonl` theo đúng định dạng README của SenseVoice: mỗi dòng có `source` (đường dẫn wav),
# `target` (transcript), `emo_target` (token cảm xúc), ...

# %%
TO_TAG = {"neutral": "<|NEUTRAL|>", "joy": "<|HAPPY|>", "anger": "<|ANGRY|>", "sadness": "<|SAD|>",
          "fear": "<|FEARFUL|>", "disgust": "<|DISGUSTED|>", "surprise": "<|SURPRISED|>"}
WORK = Path(f"/content/sv_ft_{DATASET}{FOLD}")
WORK.mkdir(exist_ok=True)

def write_jsonl(df, path):
    n = 0
    with open(path, "w") as f:
        for _, r in df.iterrows():
            if not r.text.strip():
                continue
            f.write(json.dumps({
                "key": r.id, "source": r.path, "source_len": int(r.duration * 100),
                "target": r.text, "target_len": len(r.text.split()),
                "text_language": "<|en|>", "emo_target": TO_TAG[r.emotion],
                "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>",
            }) + "\n")
            n += 1
    print(path, n)

write_jsonl(splits["train"], WORK / "train.jsonl")
write_jsonl(splits["dev"], WORK / "dev.jsonl")
!head -n 2 {WORK}/train.jsonl

# %%
import funasr
bin_dir = Path(funasr.__file__).parent / "bin"
train_tool = next(p for p in [bin_dir / "train_ds.py", bin_dir / "train.py"] if p.exists())
OUT = WORK / "out"
print("train tool:", train_tool)

cmd = f"""torchrun --nnodes 1 --nproc_per_node 1 --master_port 29511 {train_tool} \
++model="{MODEL_DIR}" ++trust_remote_code=true ++remote_code="{SV_DIR}/model.py" \
++train_data_set_list="{WORK}/train.jsonl" ++valid_data_set_list="{WORK}/dev.jsonl" \
++dataset_conf.data_split_num=1 ++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000 ++dataset_conf.sort_size=1024 ++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch={MAX_EPOCH} ++train_conf.log_interval=50 ++train_conf.resume=false \
++train_conf.validate_interval=500 ++train_conf.save_checkpoint_interval=500 \
++train_conf.keep_nbest_models=5 ++train_conf.avg_nbest_model=3 ++train_conf.use_deepspeed=false \
++optim_conf.lr={LR} ++output_dir="{OUT}" """
if RUN_FINETUNE:
    !{cmd} 2>&1 | tee {WORK}/train.log | grep -E "loss|acc|Validate|Error|error" | tail -n 400

# %%
!ls -lh {OUT}
cks = sorted(OUT.glob("model.pt*"), key=os.path.getmtime)
pick = next((c for key in ["avg", "best"] for c in cks if key in c.name), cks[-1] if cks else None)
print("checkpoint chọn:", pick)

# %%
ft = AutoModel(model=MODEL_DIR, init_param=str(pick), trust_remote_code=True,
               remote_code=f"{SV_DIR}/model.py", device="cuda:0", disable_update=True)
test_ft = run_sensevoice(ft, splits["test"])
m_ft = evaluate(f"sensevoice_small_finetune{FOLD}", DATASET, test_ft,
                notes=f"SenseVoice-Small fine-tuned {MAX_EPOCH} ep, lr {LR}, ckpt {pick.name}")

# %%
dst = CKPT_DIR / f"sensevoice_{DATASET}{FOLD}"
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(pick, dst / pick.name)
shutil.copy(WORK / "train.log", dst / "train.log")
print("saved to", dst)
# FunASR reports no dev acc for SenseVoice, so keep_nbest never prunes: each 2.7 GB checkpoint
# stays on disk (about 55 GB for 10 epochs) and fills the VM before the next notebook.
for c in OUT.glob("model.pt*"):
    c.unlink()
