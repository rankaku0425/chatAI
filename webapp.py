"""
ブラウザから会話・学習操作ができるWebアプリ (Flask)。

使い方:
    python webapp.py
ブラウザで http://127.0.0.1:5000 を開く。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

from chatai.model import GPT
from chatai.tokenizer import CharTokenizer

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# チャット機能
# ---------------------------------------------------------------------------

_model: GPT | None = None
_tokenizer: CharTokenizer | None = None
_history = ""


def load_model() -> None:
    global _model, _tokenizer
    vocab_path = CHECKPOINT_DIR / "vocab.json"
    model_path = CHECKPOINT_DIR / "model.pt"
    if not vocab_path.exists() or not model_path.exists():
        return

    _tokenizer = CharTokenizer.load(vocab_path)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _model = model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    global _history
    if _model is None:
        load_model()
    if _model is None or _tokenizer is None:
        return jsonify({"error": "モデルが見つかりません。先に学習を実行してください。"}), 400

    body = request.json or {}
    user_message = body.get("message", "")
    prompt = _history + f"User: {user_message}\nAI:"
    idx = torch.tensor([_tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)

    out = _model.generate(
        idx,
        max_new_tokens=100,
        temperature=float(body.get("temperature", 0.8)),
        top_k=int(body.get("top_k", 20)),
        top_p=float(body.get("top_p", 0.9)),
        repetition_penalty=float(body.get("repetition_penalty", 1.3)),
    )
    generated = _tokenizer.decode(out[0].tolist())

    new_text = generated[len(prompt):]
    reply = new_text.split("User:")[0].strip()

    _history = prompt + f" {reply}\n"
    return jsonify({"reply": reply})


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    global _history
    _history = ""
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# 学習操作パネル
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _build_corpus_cmd(params: dict) -> list[str]:
    source = params.get("source", "both")
    if source not in ("wikipedia", "aozora", "both"):
        raise ValueError("sourceはwikipedia/aozora/bothのいずれかで指定してください。")
    target_chars = int(params.get("target_chars", 1_000_000))
    return [
        sys.executable, "-u", "scripts/build_corpus.py",
        "--source", source,
        "--target_chars", str(target_chars),
    ]


def _pretrain_cmd(params: dict) -> list[str]:
    max_steps = int(params.get("max_steps", 5000))
    return [
        sys.executable, "-u", "train.py",
        "--stage", "pretrain",
        "--data", "data/pretrain_corpus.txt",
        "--vocab_extra", "data/conversations.txt",
        "--device", DEVICE,
        "--max_steps", str(max_steps),
    ]


def _finetune_cmd(params: dict) -> list[str]:
    max_steps = int(params.get("max_steps", 3000))
    return [
        sys.executable, "-u", "train.py",
        "--stage", "finetune",
        "--data", "data/conversations.txt",
        "--init_from", "checkpoints/pretrain.pt",
        "--device", DEVICE,
        "--max_steps", str(max_steps),
    ]


JOB_COMMANDS = {
    "build_corpus": _build_corpus_cmd,
    "pretrain": _pretrain_cmd,
    "finetune": _finetune_cmd,
}

# ジョブ名 -> 上書きされるチェックポイントファイル(バックアップ対象)
JOB_CHECKPOINT = {
    "pretrain": CHECKPOINT_DIR / "pretrain.pt",
    "finetune": CHECKPOINT_DIR / "model.pt",
}

LOSS_PATTERN = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def _backup_checkpoint(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = CHECKPOINT_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, backup_dir / f"{path.stem}_{stamp}{path.suffix}")


def _parse_loss_history(log_lines: list[str]) -> list[dict]:
    history = []
    for line in log_lines:
        m = LOSS_PATTERN.search(line)
        if m:
            history.append({"step": int(m.group(1)), "train": float(m.group(2)), "val": float(m.group(3))})
    return history


def _run_job(name: str, cmd: list[str]) -> None:
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    with _jobs_lock:
        _jobs[name]["proc"] = proc

    for line in proc.stdout:
        with _jobs_lock:
            log = _jobs[name]["log"]
            log.append(line.rstrip())
            del log[:-500]  # 直近500行だけ保持

    proc.wait()
    with _jobs_lock:
        _jobs[name]["running"] = False
        _jobs[name]["returncode"] = proc.returncode
        # 学習系ジョブが完了したらモデルを再読み込みする(次回チャット時に反映するため)
        if name in ("pretrain", "finetune") and proc.returncode == 0:
            global _model
            _model = None


@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    data = request.json or {}
    name = data.get("job")
    if name not in JOB_COMMANDS:
        return jsonify({"error": "不明なジョブです。"}), 400

    with _jobs_lock:
        if any(j.get("running") for j in _jobs.values()):
            return jsonify({"error": "他のジョブが実行中です。終了を待ってください。"}), 409

        try:
            cmd = JOB_COMMANDS[name](data.get("params", {}))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if name in JOB_CHECKPOINT:
            _backup_checkpoint(JOB_CHECKPOINT[name])

        _jobs[name] = {"log": [], "running": True, "returncode": None, "proc": None}

    thread = threading.Thread(target=_run_job, args=(name, cmd), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    name = (request.json or {}).get("job")
    with _jobs_lock:
        job = _jobs.get(name)
        if not job or not job.get("running"):
            return jsonify({"error": "実行中のジョブがありません。"}), 400
        proc = job.get("proc")

    if proc is not None:
        proc.terminate()
    return jsonify({"ok": True})


@app.route("/api/train/status")
def api_train_status():
    with _jobs_lock:
        return jsonify(
            {
                name: {
                    "running": j["running"],
                    "log": "\n".join(j["log"]),
                    "returncode": j["returncode"],
                    "loss_history": _parse_loss_history(j["log"]) if name in JOB_CHECKPOINT else [],
                }
                for name, j in _jobs.items()
            }
        )


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
