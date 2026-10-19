# RIS-LAD data and evaluation

## Annotation format

Use the official RIS-LAD train, validation and test splits. SAM3-I annotations
contain COCO-style `images`, `annotations` and `categories` arrays.

Each image record includes `id`, `file_name`, `height`, `width` and:

```json
"text_inst_input": {
  "concept": ["object category"],
  "simple_query": ["the referring expression"],
  "complex_query": ["the referring expression"]
}
```

Each annotation contains a unique `id`, `image_id`, `category_id`, `bbox`,
`area`, `iscrowd` and `segmentation`. Segmentation uses COCO RLE with `size`
and compressed `counts`. Image metadata and RLE dimensions describe the input
canvas. Resolve `file_name` relative to the image root supplied to the launcher.
Keep annotation IDs consistent between candidate prediction and evaluation.

## Observation cache

`tools/prepare_internal_split.py` divides training images into a 90/10 split
using seed 123. Referring queries from the same RGB image stay together.

`tools/build_cache.py` caches frozen SAM3-I features and candidate observations.
Training crop proposals comprise target-centered, shifted and random regions
at scales 0.30, 0.45, 0.60 and 0.75. The launcher samples two random crops per
query. Training annotations supervise candidate quality and target beliefs.
At inference, the model posterior and candidate mask disagreement determine
the crop action.

## Segmentation metrics

RIS-LAD evaluation uses native GT mask dimensions. Binary masks with different
dimensions are aligned using integer nearest-neighbor indexing. Image-level
prediction unions the returned candidate masks; DRA-SAM prediction uses the
mask selected by its posterior decision.

For each referring query, IoU is intersection divided by union. mIoU is the
arithmetic mean across queries, and oIoU is total intersection divided by total
union. Precision@t is the fraction of queries satisfying IoU ≥ t. Thresholds
are 0.5, 0.6, 0.7, 0.8 and 0.9. Empty predictions contribute zero intersection
and the GT area to union.
