# Configuration reference

## Backbone training

The YAML files in `sam3/sam3/train/configs/dra_sam/` define the SAM3-I backbone
training stages. Stage1_2 trains for 8 epochs; Stage3 trains for 4 epochs. The
model input resolution is 1008 and the backbone training batch size is 1.

The launcher sets the following environment variables:

| Variable | Purpose |
|---|---|
| `DRA_ROOT` | Repository root |
| `DRA_ANNOTATIONS` | Annotation directory |
| `DRA_IMAGES` | RGB image root |
| `DRA_INIT_CHECKPOINT` | Initialization checkpoint |

Stage3 initializes from
`outputs/rislad/stage1_2_continue/checkpoints/checkpoint_4.pt`. The `--init` argument
selects an explicit initialization checkpoint for either stage.

## EvidenceScorer training

The training configuration is `configs/rislad.yaml`.

| Parameter | Value |
|---|---:|
| Candidate/text dimension | 256 |
| Geometry dimension | 7 |
| Hidden dimension | 192 |
| Epochs | 30 |
| Batch size | 4 |
| Seed | 123 |
| Optimizer | AdamW |
| Learning rate | 0.0002 |
| Weight decay | 0.05 |
| Warmup | 1 epoch |
| Schedule | Cosine |
| Gradient clipping norm | 1.0 |
| AMP dtype | BF16 |
| Target belief temperature | 0.1 |
| Prior loss weight | 2/3 |
| Posterior loss weight | 4/3 |

The relative posterior/prior loss weight is 2. Target belief temperature
controls the supervision distribution, while posterior temperature controls
the inference belief distribution. The training posterior temperature is 1.0.

## Inference

The action grid contains 2,304 crop actions: 24 × 24 centers and four scales
(0.30, 0.45, 0.60 and 0.75). Referential utility combines candidate competition
and unresolved spatial mass to select the observation region. Local visual
evidence updates the candidate posterior, whose decision determines the output
mask. The unresolved-state decision uses the image-level union fallback.

| Parameter | Value |
|---|---:|
| Posterior temperature | 1.2 |
| Unknown utility weight | 1.0 |
| Detection threshold | 0.5 |

Use `--tau`, `--unknown-weight` and `--batch-size` to set the corresponding
inference parameters. `--config` selects an EvidenceScorer training YAML, and
`--dry-run` displays the launch configuration.
