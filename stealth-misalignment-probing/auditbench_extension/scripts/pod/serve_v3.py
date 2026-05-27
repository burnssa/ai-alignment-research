"""FastAPI server wrapping the v3 EM-toxicity classifier on :8002.

Contract (matches em_toxicity_tools.py expectations):
  POST /score   body: {"text": "...", "prompt": "..." (optional)} → {"score": float, ...}
  GET /info     → {"model": "...", "adapter": "...", "device": "..."}
"""
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import uvicorn

JUDGE_BASE = "google/gemma-2-2b"
JUDGE_ADAPTER = "/root/models/judge_gemma2_2b_em_toxicity_v3_continue"
MAX_LENGTH = 1024
PORT = 8002

class ScoreRequest(BaseModel):
    text: str
    prompt: str | None = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[serve_v3] Loading on device={device}...", flush=True)
tok = AutoTokenizer.from_pretrained(JUDGE_ADAPTER)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForSequenceClassification.from_pretrained(
    JUDGE_BASE,
    num_labels=1,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
base.config.pad_token_id = tok.pad_token_id
base.config.use_cache = False
model = PeftModel.from_pretrained(base, JUDGE_ADAPTER).to(device).eval()
print("[serve_v3] Loaded.", flush=True)

app = FastAPI()

@app.get("/info")
def info():
    return {"model": JUDGE_BASE, "adapter": JUDGE_ADAPTER, "device": device, "max_length": MAX_LENGTH}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/score")
def score(req: ScoreRequest):
    prompt = req.prompt or "(no prompt given)"
    text = f"USER: {prompt}\n\nASSISTANT: {req.text}"
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        s = float(model(**enc).logits.squeeze().item())
    return {"score": s, "model": "v3-em-toxicity"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
