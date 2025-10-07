# Prime-RL Disaggregated Setup

## 1️ Motivation

Prime-RL originally assumes a single-machine setup, where the trainer, orchestrator, and inference engine all share the same filesystem, so the orchestrator can update weights by passing a local file path.

In our deployment:
- **Trainer / Orchestrator** → runs on *Leo* (local GPU server)
- **Inference Server** → runs on *NSCC* (remote cluster node)
- The two machines **do not share a filesystem**

Therefore, the default `/update_weights` endpoint (which loads weights via local path) doesn’t work.  
So we need to extend it to support **HTTP file upload** via multipart/form-data.

---

## 2️ Summary of Changes

Three files were lightly modified to support this setup:

| File | Main change |
|------|-------------|
| `src/prime_rl/inference/vllm/server.py` | Change `/update_weights` endpoint to support multipart upload of weight file |
| `src/prime_rl/inference/vllm/worker.py` | Update model loader to point to new transferred weight file |
| `prime_rl/orchestrator/client.py` | Modify client upload logic. Send weight files via HTTP multipart (instead of JSON path) |


## 3 Running Instructions
### On Inference Server
```bash
uv run inference @ examples/reverse_text/rl/infer.toml \
  --host 0.0.0.0 --port 30000 \
  --model.name Logan586/Qwen3-0.6B-Reverse-Text-SFT
```

### On Trainer & Orchestrator Server
```bash
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --model.name Logan586/Qwen3-0.6B-Reverse-Text-SFT \
  --wandb.project prime-rl-disaggregated \
  --wandb.name prime-rl-disaggregated-reverse-text \
  --orchestrator.client.base-url "http://127.0.0.1:30000/v1"
```

### SSH tunnel Setup
```bash
ssh -N -J <jump_host_user>@<jump_host_domain> <remote_user>@<remote_node> -L 30000:localhost:30000
```
```bash
ssh -N -R 30000:localhost:30000 <local_user>@<local_machine_domain>
```