"""The models of data/cached/, run in the notebook instead of loaded: PaddleOCR-VL and Qwen VL, general and
fine-tuned. Each writes its output on a page in the format of its saved copy, so the notebook reads both the
same way. They need a GPU; on a CPU a page takes many minutes.

The settings are those of the saved outputs:
- PaddleOCR-VL: PaddleOCRVL(pipeline_version="v1.6"), every other setting at its default, one predict() per page.
- Qwen VL: the page__doctags prompt (data/prompts/), in one user turn before the image, as the fine-tuned
  model was trained; float16; greedy decoding with the model's repetition penalty of 1.05; an answer of up
  to 8192 tokens minus the prompt. The saved outputs came from vLLM, so a run here can differ in places.

Only one model is on the GPU at a time, so the three fit a 16 GB GPU one after the other.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path

from . import data

PROMPT = (data.DATA / "prompts" / "page__doctags.txt").read_text(encoding="utf-8")
MAX_LEN = 8192                          # the context the saved outputs were served with: prompt and answer
REPETITION_PENALTY = 1.05               # the model's generation_config.json
QWEN = {"qwen_base": ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3")}
FINETUNED = "DUDEMO_FINETUNED"          # where the fine-tuned weights are: a folder, or a Hugging Face repo id

_loaded: dict = {}                      # the one model on the GPU: {name: model}


def available() -> list[str]:
    """The models that can run here: PaddleOCR-VL if paddleocr is installed, the general Qwen VL, and the
    fine-tuned one if its weights are set in DUDEMO_FINETUNED."""
    import importlib.util
    models = ["paddleocr_vl"] if importlib.util.find_spec("paddleocr") else []
    return models + ["qwen_base"] + (["qwen_finetuned"] if os.environ.get(FINETUNED) else [])


def _weights(model: str) -> tuple[str, str | None]:
    if model == "qwen_finetuned":
        if not os.environ.get(FINETUNED):
            raise RuntimeError(f"the fine-tuned weights are not public: set {FINETUNED} to their folder or "
                               "Hugging Face repo id")
        return os.environ[FINETUNED], None
    return QWEN[model]


def _load(name: str, make):
    """The model `name`, loaded with `make()` after the one on the GPU before it is freed."""
    if name not in _loaded:
        for other in list(_loaded):
            del _loaded[other]
        gc.collect()
        _free_gpu()
        _loaded[name] = make()
    return _loaded[name]


def _free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except ImportError:
        pass


def _qwen_model(model: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path, revision = _weights(model)
    gpu = torch.cuda.is_available()
    weights = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, revision=revision, dtype=torch.float16 if gpu else torch.float32)
    return weights.to("cuda" if gpu else "cpu").eval(), AutoProcessor.from_pretrained(path, revision=revision)


def qwen(model: str, page: str, max_new_tokens: int | None = None) -> str:
    """Qwen VL's DocTags answer for a page, as the model wrote it. `max_new_tokens` cuts the answer short,
    for a quick look at its first blocks."""
    import torch
    weights, processor = _load(model, lambda: _qwen_model(model))
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[data.page_image(page)], return_tensors="pt").to(weights.device)
    prompt = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = weights.generate(**inputs, max_new_tokens=max_new_tokens or MAX_LEN - prompt, do_sample=False,
                               repetition_penalty=REPETITION_PENALTY)
    return processor.batch_decode(out[:, prompt:], skip_special_tokens=True)[0]


def paddle(page: str) -> dict:
    """PaddleOCR-VL's blocks for a page, in the format of data/cached/paddleocr_vl/: the page size and, per
    block in reading order, its label, its box and what the VLM read."""
    from paddleocr import PaddleOCRVL
    pipeline = _load("paddleocr_vl", lambda: PaddleOCRVL(pipeline_version="v1.6"))
    result = list(pipeline.predict(str(data.page_path(page))))[0]
    with tempfile.TemporaryDirectory() as folder:
        result.save_to_json(save_path=folder)
        res = json.loads(next(Path(folder).glob("*_res.json")).read_text(encoding="utf-8"))
    blocks = [{"label": b["block_label"], "box": b["block_bbox"], "content": b["block_content"]}
              for b in res["parsing_res_list"]]
    return {"width": res["width"], "height": res["height"], "blocks": blocks}


def run(model: str, page: str, folder: Path) -> Path:
    """Runs `model` on `page`, writes its output into `folder` as its saved copy is written, prints how
    long it took, and returns the file."""
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if model == "paddleocr_vl":
        path = folder / f"{page}.json"
        path.write_text(json.dumps(paddle(page), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        path = folder / f"{page}.doctags.txt"
        path.write_text(qwen(model, page), encoding="utf-8")
    print(f"{data.NAMES[model]} read {page} in {time.perf_counter() - start:.0f} s (model loading included "
          f"the first time).")
    return path
