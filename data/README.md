# data

Everything the notebook reads. Nothing here is written by the notebook; models run live write to
`outputs/live/` instead.

| Folder | Contents |
|---|---|
| `pages/` | The page images: synthetic lab reports from Atomic Intelligence's page generator |
| `ground_truth/` | The generator's record of each page: every region's type, box and text |
| `cached/<engine>/` | What each engine wrote last time: `docling` (its DoclingDocument, with the page image), `paddleocr_vl`, `qwen_base` (Qwen2.5-VL-3B as released) and `qwen_finetuned` (the same model with our LoRA) |
| `cached/runs.json` | How each of them was written: the date, and the seconds per page on each device it was timed on |
| `prompts/` | The prompt both Qwen VL models were given (`page__doctags`), for running them live |

The saved outputs come from the evaluation in Atomic Intelligence's research repository, where every
model read the same page images.
