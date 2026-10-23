# Document understanding demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atomic-Intelligence/document-understanding-demo/blob/main/notebooks/demo.ipynb)

The notebook demo of the AI2Future workshop "Can We Rely on Extracted Documents? Parsing Methods
Compared" (Filip Pavičić, Atomic Intelligence).

It follows one page of a synthetic lab report through a document-parsing pipeline: the regions a parser
finds on the page, the Markdown it writes, the errors it makes and the checks that catch some of them.
Docling runs live on a CPU. PaddleOCR-VL and Qwen2.5-VL (as released and fine-tuned by us) need a GPU,
so their outputs on the same page are included in the repository.

## What the notebook shows

1. **The page**: a synthetic Croatian lab report with tables, key-value fields and units.
2. **Docling on the page**: the regions it finds, coloured by type, next to the Markdown it writes.
3. **The same page, three other parsers**: PaddleOCR-VL, general Qwen VL and fine-tuned Qwen VL.
4. **Three kinds of error**: visibly broken, structure wrong, and a wrong value that passes as correct.
5. **General and fine-tuned Qwen VL**: the regions each one finds and the ones it misses.
6. **Checks after extraction**: simple rules on Docling's table, and what a second parser adds.
7. **A table across pages**: the header is lost after a page break, and comes back when both parts are
   read together.
8. **Your own document**: Docling on a file of your choice.

## Run on Colab

Open the notebook with the badge above and run the cells from the top. The first cell clones this
repository and installs the dependencies. The free CPU runtime is enough; no GPU is needed.

## Run locally

Python 3.11 to 3.13, a CPU and about 8 GB of free RAM.

```bash
git clone https://github.com/Atomic-Intelligence/document-understanding-demo.git
cd document-understanding-demo
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Linux only: the CPU build of torch. Without this, pip fetches the CUDA build (several GB).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[jupyter]"
jupyter lab notebooks/demo.ipynb
```

Docling downloads its layout, table and OCR models on first use. The setup cell of the notebook does
this, so later cells run without waiting for downloads.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/demo.ipynb` | The demo notebook |
| `src/dudemo/` | Helpers the notebook imports: loading pages and outputs, drawing regions, checks |
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
