# Document understanding demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atomic-Intelligence/document-understanding-demo/blob/main/notebooks/demo.ipynb)

The notebook demo of the AI2Future workshop "Can We Rely on Extracted Documents? Parsing Methods
Compared" (Filip Pavičić, Atomic Intelligence).

It follows one page of a synthetic lab report through a document-parsing pipeline: the regions a parser
finds on the page, the Markdown it writes, the errors it makes and the checks that catch some of them.
Every engine's output on the pages is included in the repository, so the notebook runs anywhere in
seconds. Each one can also run live: Docling on a CPU or a GPU, PaddleOCR-VL and Qwen2.5-VL (as released
and fine-tuned by us) on a GPU.

## What the notebook shows

1. **The page**: a synthetic Croatian lab report with tables, key-value fields and units.
2. **Docling on the page**: the regions it finds, the tables it reads and the Markdown it writes.
3. **The same page, two other parsers**: PaddleOCR-VL and general Qwen VL, next to Docling.
4. **Fine-tuning Qwen VL**: the general and the fine-tuned model side by side, what fine-tuning fixed and
   what it did not.
5. **Three kinds of error**: visibly broken, structure wrong, and silent (a wrong value that is
   well-formed and looks right).
6. **Checks after extraction**: simple rules on Docling's tables.
7. **Two parsers, rules and a person**: the whole second report read by two parsers, their
   disagreements settled by rules and by a person, and the score against the ground truth.
8. **Your own document**: Docling on a file of your choice.

## Run on Colab

Open the notebook with the badge above and run the cells from the top. The setup clones this repository
and installs what the engines chosen live need: with all of them cached, only a few small packages. The
free CPU runtime is enough; a GPU is needed only for running PaddleOCR-VL or Qwen VL live (see "Live or
cached").

## Run locally

Python 3.11 to 3.13. With every engine cached, any machine will do:

```bash
git clone https://github.com/Atomic-Intelligence/document-understanding-demo.git
cd document-understanding-demo
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[jupyter]"
jupyter lab notebooks/demo.ipynb
```

To run Docling live, add its stack; it needs about 8 GB of free RAM, and downloads its layout, table and
OCR models the first time, in section 2.1:

```bash
# Linux only: the CPU build of torch. Without this, pip fetches the CUDA build (several GB).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[docling]"
```

## Live or cached

The first cell of the notebook chooses, for each engine, whether it runs in the notebook or what it wrote
last time is loaded from `data/cached/`:

```python
LIVE = {
    "docling": False,
    "paddleocr_vl": False,      # True starts a vLLM server: a GPU with 10 GB free
    "qwen_base": False,         # True starts a vLLM server: a GPU with 13 GB free
    "qwen_finetuned": False,    # the same, and its weights in DUDEMO_FINETUNED
}
```

All are cached by default, so the notebook runs anywhere in seconds. A cached engine prints when its
output was written and how long it took (`data/cached/runs.json`); a live one prints how long it takes
here.

Docling runs live in the notebook, on a CPU or a GPU. PaddleOCR-VL and Qwen VL each start a vLLM server on
the GPU when the notebook first needs them, one at a time: the server reads both reports and stops again.
On an A40, a server takes 1 to 1.5 minutes to start; then PaddleOCR-VL reads a page in about 2 s and Qwen
VL in 30 to 45 s.

- **Colab:** choose a GPU runtime (Runtime → Change runtime type) before setting a GPU engine live. The
  setup cell then also installs vLLM, which replaces Colab's torch with 2.8, and PaddleOCR-VL if it is
  live; that takes a few minutes. This is not tested on Colab yet.
- **Locally, on a GPU machine:**

  ```bash
  pip install -e ".[jupyter,docling,live]"  # vLLM 0.10.2 for the servers, with torch 2.8 for CUDA 12
  # PaddleOCR-VL's server needs flash-attn; its wheels are on GitHub (this one: Python 3.12)
  pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  # paddlepaddle-gpu pins exact CUDA 12.6 libraries; it runs on torch's newer ones, so it goes in
  # without its dependencies, and its few others after it
  pip install paddlepaddle-gpu==3.2.0 --no-deps -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
  pip install httpx networkx opt-einsum protobuf
  ```

The fine-tuned model's weights are not published: set `DUDEMO_FINETUNED` to their folder or Hugging Face
repo id to run it. With the saved outputs' settings, Qwen VL's live answers are the same as the saved
ones; PaddleOCR-VL's saved outputs came from its native engine, so with the server a few words can differ.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/demo.ipynb` | The demo notebook |
| `notebooks/demo_with_outputs.ipynb` | The same notebook run once, with every output saved, to look through without running it ([open in Colab](https://colab.research.google.com/github/Atomic-Intelligence/document-understanding-demo/blob/main/notebooks/demo_with_outputs.ipynb)) |
| `src/dudemo/` | Helpers the notebook imports: loading pages and saved outputs, drawing regions, rendering Markdown |
| `data/` | The pages, their ground truth and the saved outputs of the GPU models (see `data/README.md`) |
| `tests/` | Tests for the helpers and the notebook |

## Models and data

| | Licence | How it is used here |
|---|---|---|
| [Docling](https://github.com/docling-project/docling) | MIT; models Apache 2.0 and CDLA-Permissive 2.0 | Runs in the notebook |
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | Apache 2.0 | Saved outputs |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Qwen Research License | Saved outputs, as released and with our fine-tuning |
| Lab reports | Synthetic, made by Atomic Intelligence's page generator | Pages and ground truth |

The lab reports are generated. Names, results and record numbers on them are not real patient data.

## Licence

To be decided; see `LICENSE`.
