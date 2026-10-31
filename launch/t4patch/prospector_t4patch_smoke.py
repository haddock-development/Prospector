import os, subprocess, time

LOG = "/kaggle/working/smoke_v3.log"

def log(m):
    with open(LOG, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def sh(cmd, capture=False):
    log(f"\n### $ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=capture)
    out = ((r.stdout or "") + (r.stderr or ""))[-2500:]
    if out.strip(): log(out)
    if r.returncode != 0:
        log(f"!!! FAILED({r.returncode}): {cmd}")
        raise SystemExit(1)
    return r

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # DP-incompatibility of liger fused CE; full run uses torchrun DDP
os.environ["HF_HOME"] = "/kaggle/tmp/hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["UV_CACHE_DIR"] = "/kaggle/tmp/uv-cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TMP = "/kaggle/tmp"
REPO = f"{TMP}/Prospector"
os.makedirs(TMP, exist_ok=True)

r = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, text=True, capture_output=True)
log("GPU: " + r.stdout.strip())

if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 https://github.com/haddock-development/Prospector {REPO}", capture=True)

# --- T4-Patch: use_liger_kernel-Schalter in den SFT-Trainer ---
import base64
patch = base64.b64decode(
    "aW1wb3J0IHBhdGhsaWIKcCA9IHBhdGhsaWIuUGF0aCgic3JjL2dyYXBoX3JlYXNvbmVyL3RyYWluL3NmdC5weSIpCnMgPSBwLnJlYWRfdGV4dCgpCmFzc2VydCAidXNlX2xpZ2VyX2tlcm5lbCIgbm90IGluIHMsICJiZXJlaXRzIGdlcGF0Y2h0IgpvbGRfa2V5cyA9ICcic2F2ZV9zdGVwcyIsICJzYXZlX3N0cmF0ZWd5IiwgInNlZWQiLCcKbmV3X2tleXMgPSAnInNhdmVfc3RlcHMiLCAic2F2ZV9zdHJhdGVneSIsICJzZWVkIiwgInVzZV9saWdlcl9rZXJuZWwiLCAiYW1wIiwnCmFzc2VydCBvbGRfa2V5cyBpbiBzCnMgPSBzLnJlcGxhY2Uob2xkX2tleXMsIG5ld19rZXlzKQpvbGRfYXJncyA9ICIgICAgICAgIHJlcG9ydF90bz1bXSxcbiAgICAgICAgc2VlZD0iCm5ld19hcmdzID0gKCIgICAgICAgIHJlcG9ydF90bz1bXSxcbiIKICAgICAgICAgICAgIiAgICAgICAgIyBUNC1wYXRjaDogbGlnZXIgZnVzZWQgbGluZWFyIGNyb3NzLWVudHJvcHkgKGtlaW4gZnAzMi1Mb2dpdC1cbiIKICAgICAgICAgICAgIiAgICAgICAgIyBNYXRlcmlhbGlzaWVydW5nKSArIG9wdGlvbmFsZXIgQU1QLU1vZHVzIChcImZwMTZcInxcImJmMTZcIik6XG4iCiAgICAgICAgICAgICIgICAgICAgICMgZnAzMiBNYXN0ZXItV2VpZ2h0cywgcmVkdWNlZC1wcmVjaXNpb24gTWF0bXVscyB2aWEgQXV0b2Nhc3QuXG4iCiAgICAgICAgICAgICIgICAgICAgIHVzZV9saWdlcl9rZXJuZWw9Ym9vbChjb25maWcuZ2V0KFwidXNlX2xpZ2VyX2tlcm5lbFwiLCBGYWxzZSkpLFxuIgogICAgICAgICAgICAiICAgICAgICBiZjE2PXN0cihjb25maWcuZ2V0KFwiYW1wXCIsIFwiXCIpKSA9PSBcImJmMTZcIiBhbmQgdXNlX2JmMTYsXG4iCiAgICAgICAgICAgICIgICAgICAgIGZwMTY9c3RyKGNvbmZpZy5nZXQoXCJhbXBcIiwgXCJcIikpID09IFwiZnAxNlwiLFxuICAgICAgICBzZWVkPSIpCmFzc2VydCBvbGRfYXJncyBpbiBzCnMgPSBzLnJlcGxhY2Uob2xkX2FyZ3MsIG5ld19hcmdzKQpvbGRfZmxhZ3MgPSAiIiIgICAgICAgIGJmMTY9dXNlX2JmMTYgYW5kIGR0eXBlIGlzIHRvcmNoLmJmbG9hdDE2LAogICAgICAgIGZwMTY9KG5vdCB1c2VfYmYxNikgYW5kIGR0eXBlIGlzIHRvcmNoLmZsb2F0MTYsCiIiIgphc3NlcnQgb2xkX2ZsYWdzIGluIHMKcyA9IHMucmVwbGFjZShvbGRfZmxhZ3MsICIiKQojIGZwMTYtR2V3aWNodGU6IE5vcm1lbiBpbiBmcDMyIHN0YWJpbGlzaWVyZW4gZGVuIFJlc2lkdWFsc3Ryb20gKFQ0LVJlemVwdCkKb2xkX3BlZnQgPSAiICAgIG1vZGVsID0gZ2V0X3BlZnRfbW9kZWwobW9kZWwsIGxvcmEpIgpuZXdfcGVmdCA9ICIiIiAgICBpZiBkdHlwZSBpcyB0b3JjaC5mbG9hdDE2OgogICAgICAgIGZvciBtIGluIG1vZGVsLm1vZHVsZXMoKToKICAgICAgICAgICAgaWYgaXNpbnN0YW5jZShtLCAodG9yY2gubm4uUk1TTm9ybSwgdG9yY2gubm4uTGF5ZXJOb3JtKSk6CiAgICAgICAgICAgICAgICBtLnRvKHRvcmNoLmZsb2F0MzIpCiAgICBtb2RlbCA9IGdldF9wZWZ0X21vZGVsKG1vZGVsLCBsb3JhKSIiIgphc3NlcnQgb2xkX3BlZnQgaW4gcwpzID0gcy5yZXBsYWNlKG9sZF9wZWZ0LCBuZXdfcGVmdCkKcC53cml0ZV90ZXh0KHMpCnByaW50KCJwYXRjaGVkIG9rIikK"
).decode()

open(f"{REPO}/apply_patch.py", "w").write(patch)
sh(f"cd {REPO} && python3 apply_patch.py")

sh("curl -LsSf https://astral.sh/uv/install.sh | sh", capture=True)
os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
sh(f"cd {REPO} && uv venv --python 3.12 .venv", capture=True)
sh(f"cd {REPO} && uv pip install --python .venv/bin/python -e '.[dev,train]' -c launch/constraints-gpu.txt liger-kernel==0.8.5", capture=True)

PYBIN = f"{REPO}/.venv/bin"
sh(f"{PYBIN}/hf download lamm-mit/Prospector --local-dir {REPO} --include 'data/traj/tsmc_sft_mix_flash20.jsonl' --include 'data/raw/*' --include 'data/benchmarks/tsmc/*'", capture=True)
sh(f"{PYBIN}/python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3.5-0.8B\")'", capture=True)

# gepatchte Config ablegen
_sft1 = open(f"{REPO}/configs/sft1.yaml").read().replace("model_dtype: float32", "model_dtype: float16")
cfg = _sft1.rstrip() + "\n\nuse_liger_kernel: true\namp: fp16\n"
open(f"{REPO}/configs/sft1_t4patch.yaml", "w").write(cfg)

sh(f"cd {REPO} && {PYBIN}/prospector plan --graph tsmc --size 0.8b | head -3", capture=True)

log("\n=== SFT smoke (3 steps, liger patch) ===")
t0 = time.time()
sh(f"cd {REPO} && {PYBIN}/python scripts/train_sft.py --graph tsmc --size 0.8b --stage sft1 --config configs/sft1_t4patch.yaml --smoke")
log(f"\nSMOKE-DONE in {time.time()-t0:.0f}s")
log("PROSPECTOR-T4PATCH-SMOKE-OK")
