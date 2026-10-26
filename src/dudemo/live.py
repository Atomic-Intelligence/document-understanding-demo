"""The GPU engines of data/cached/, run in the notebook instead of loaded: PaddleOCR-VL and Qwen VL, general and
fine-tuned. Each one starts a vLLM server, reads every page with it, writes its outputs in the format of
their saved copies, and stops the server again, so only one server is on the GPU at a time.

The settings are those of the saved outputs:
- PaddleOCR-VL: PaddleOCRVL(pipeline_version="v1.6"), with its VLM on PaddleOCR's vLLM server
  (`paddleocr genai_server`); the layout model runs in the notebook.
- Qwen VL: vLLM's OpenAI server with float16, a context of 8192 tokens, the model's repetition penalty and
  greedy decoding; the page__doctags prompt (data/prompts/) in one user turn before the image, as the
  fine-tuned model was trained.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from . import data

PROMPT = (data.DATA / "prompts" / "page__doctags.txt").read_text(encoding="utf-8")
PAGES = sorted(p.stem for p in (data.DATA / "pages").glob("*.png"))
QWEN = {"qwen_base": ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3")}
FINETUNED = "DUDEMO_FINETUNED"          # where the fine-tuned weights are: a folder, or a Hugging Face repo id
GPU_GB = {"paddleocr_vl": 10, "qwen_base": 13, "qwen_finetuned": 13}     # what each server takes on the GPU
START_TIMEOUT_S = 900

_server: dict = {}                      # the one server running: process, port, engine


def _gpu_memory_gb() -> tuple[float, float]:
    """Total and free memory of the GPU the servers run on (the first of CUDA_VISIBLE_DEVICES)."""
    index = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        row = subprocess.run(["nvidia-smi", "-i", index, "--query-gpu=memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("the GPU engines need a GPU, and there is none here") from None
    total, free = (int(x) / 1024 for x in row.strip().split(","))
    return total, free


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _command(engine: str, port: int, share: float, folder: Path) -> list[str]:
    """The command that starts the server of an engine, taking `share` of the GPU's memory."""
    if engine == "paddleocr_vl":
        config = folder / "paddleocr_vl_server.yaml"
        config.write_text(f"gpu-memory-utilization: {share:.2f}\n")
        cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex"))
        weights = cache / "official_models" / "PaddleOCR-VL-1.6"     # where the pipeline keeps its VLM
        return [sys.executable, "-m", "paddleocr", "genai_server", "--model_name", "PaddleOCR-VL-1.6-0.9B",
                "--backend", "vllm", "--host", "127.0.0.1", "--port", str(port), "--backend_config", str(config),
                *(["--model_dir", str(weights)] if weights.exists() else [])]
    if engine == "qwen_finetuned":
        if not os.environ.get(FINETUNED):
            raise RuntimeError(f"the fine-tuned weights are not public: set {FINETUNED} to their folder or "
                               "Hugging Face repo id")
        weights, revision = os.environ[FINETUNED], None
    else:
        weights, revision = QWEN[engine]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", weights,
            *(["--revision", revision] if revision else []), "--served-model-name", engine, "--dtype", "float16",
            "--max-model-len", "8192", "--max-num-batched-tokens", "2048", "--max-num-seqs", "1",
            "--gpu-memory-utilization", f"{share:.2f}", "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
            "--seed", "0", "--host", "127.0.0.1", "--port", str(port)]


def start(engine: str, folder: Path):
    """Starts the vLLM server of `engine`, after stopping any other, and waits until it answers."""
    if _server.get("engine") == engine:
        return
    stop()
    total, free = _gpu_memory_gb()
    need = GPU_GB[engine]
    if free < need:
        raise RuntimeError(f"{data.NAMES[engine]}'s server needs {need} GB of GPU memory, and {free:.0f} GB is free")
    port = _free_port()
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"{engine}_server.log"
    began = time.perf_counter()
    process = subprocess.Popen(_command(engine, port, need / total, folder), stdout=log.open("w"),
                               stderr=subprocess.STDOUT, start_new_session=True)
    _server.update(process=process, port=port, engine=engine)
    while True:
        if process.poll() is not None:
            stop()
            raise RuntimeError(f"{data.NAMES[engine]}'s server stopped while starting; its log is {log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except OSError:
            if time.perf_counter() - began > START_TIMEOUT_S:
                stop()
                raise RuntimeError(f"{data.NAMES[engine]}'s server did not start in {START_TIMEOUT_S} s; "
                                   f"its log is {log}") from None
            time.sleep(1)
    print(f"Started {data.NAMES[engine]}'s vLLM server in {time.perf_counter() - began:.0f} s "
          f"({need} GB of GPU memory).")


@atexit.register
def stop():
    """Stops the server that runs, if one does."""
    process = _server.get("process")
    _server.clear()
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def qwen(engine: str, page: str) -> str:
    """Qwen VL's DocTags answer for a page, as the model wrote it, from its running server."""
    image = base64.b64encode(data.page_path(page).read_bytes()).decode()
    body = {"model": engine, "temperature": 0.0, "seed": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}]}
    request = urllib.request.Request(f"http://127.0.0.1:{_server['port']}/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=900).read())["choices"][0]["message"]["content"]


def paddle(pipeline, page: str) -> dict:
    """PaddleOCR-VL's blocks for a page, in the format of data/cached/paddleocr_vl/: the page size and, per
    block in reading order, its label, its box and what the VLM read."""
    result = list(pipeline.predict(str(data.page_path(page))))[0]
    with tempfile.TemporaryDirectory() as folder:
        result.save_to_json(save_path=folder)
        res = json.loads(next(Path(folder).glob("*_res.json")).read_text(encoding="utf-8"))
    blocks = [{"label": b["block_label"], "box": b["block_bbox"], "content": b["block_content"]}
              for b in res["parsing_res_list"]]
    return {"width": res["width"], "height": res["height"], "blocks": blocks}


def run(engine: str, page: str, folder: Path):
    """Starts the server of `engine`, reads every page with it into `folder` (each file named as its saved
    copy), prints how long each took, and stops the server. `page` is the one asked for; the others are
    read too, so the server starts only once."""
    start(engine, folder.parent)
    folder.mkdir(parents=True, exist_ok=True)
    pipeline = None
    try:
        if engine == "paddleocr_vl":
            from paddleocr import PaddleOCRVL
            pipeline = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server",
                                   vl_rec_server_url=f"http://127.0.0.1:{_server['port']}/v1")
        for name in [page] + [p for p in PAGES if p != page]:
            began = time.perf_counter()
            if pipeline is not None:
                output = json.dumps(paddle(pipeline, name), ensure_ascii=False, indent=1) + "\n"
                (folder / f"{name}.json").write_text(output, encoding="utf-8")
            else:
                (folder / f"{name}.doctags.txt").write_text(qwen(engine, name), encoding="utf-8")
            print(f"{data.NAMES[engine]} read {name} in {time.perf_counter() - began:.0f} s.")
    finally:
        del pipeline
        stop()
