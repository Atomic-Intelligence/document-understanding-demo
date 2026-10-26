# data

Everything the notebook reads. Nothing here is written by the notebook.

| Folder | Contents |
|---|---|
| `pages/` | The page images: synthetic lab reports from Atomic Intelligence's page generator |
| `ground_truth/` | The generator's record of each page: every region's type, box and text |
| `cached/<model>/` | Saved outputs of the models that need a GPU: `paddleocr_vl`, `qwen_base` (Qwen2.5-VL-3B as released) and `qwen_finetuned` (the same model with our LoRA) |

The saved outputs come from the evaluation in Atomic Intelligence's research repository, where every
model read the same page images.
