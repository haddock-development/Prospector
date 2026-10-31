import os, subprocess, time

LOG = "/content/prospector_setup.log"

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

os.environ["HF_HOME"] = "/content/hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["UV_CACHE_DIR"] = "/content/uv-cache"
REPO = "/content/Prospector"

r = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, text=True, capture_output=True)
log("GPU: " + r.stdout.strip())

if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 https://github.com/haddock-development/Prospector {REPO}", capture=True)
sh("curl -LsSf https://astral.sh/uv/install.sh | sh", capture=True)
os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
sh(f"cd {REPO} && uv venv --python 3.12 .venv", capture=True)
sh(f"cd {REPO} && uv pip install --python .venv/bin/python -e '.[dev,train]' -c launch/constraints-gpu.txt", capture=True)

PYBIN = f"{REPO}/.venv/bin"
sh(f"{PYBIN}/hf download lamm-mit/Prospector --local-dir {REPO} --include 'data/traj/tsmc_sft_mix_flash20.jsonl' --include 'data/raw/*' --include 'data/benchmarks/tsmc/*'", capture=True)
sh(f"{PYBIN}/python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3.5-0.8B\")'", capture=True)
sh(f"cd {REPO} && {PYBIN}/prospector plan --graph tsmc --size 0.8b | head -5", capture=True)

log("\n=== SFT smoke (3 steps) ===")
t0 = time.time()
sh(f"cd {REPO} && {PYBIN}/python scripts/train_sft.py --graph tsmc --size 0.8b --stage sft1 --smoke")
log(f"SMOKE-DONE in {time.time()-t0:.0f}s")
log("PROSPECTOR-L4-SMOKE-OK")
