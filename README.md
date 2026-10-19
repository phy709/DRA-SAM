# DRA-SAM

DRA-SAM performs referring image segmentation through candidate belief modeling,
referential utility and active local observation. An EvidenceScorer integrates
visual, language and geometric evidence, while an analytic utility selects the
observation region. Local evidence updates the candidate posterior to determine
the segmentation output.

## Project structure

| Path | Purpose |
|---|---|
| `sam3/sam3/refzoom/analytic_rai.py` | EvidenceScorer, referential utility and posterior update |
| `sam3/sam3/refzoom/analytic_rai_loss.py` | EvidenceScorer training objective |
| `sam3/sam3/train/configs/dra_sam/` | SAM3-I backbone training configurations |
| `configs/rislad.yaml` | EvidenceScorer and inference configuration |
| `scripts/run.py` | Training, prediction and evaluation commands |
| `tools/` | Observation caching, model training and segmentation metrics |

See [directory structure](DIRECTORY.md), [data format](docs/DATA.md) and
[configuration reference](docs/CONFIGURATION.md).

## Installation

Runtime: Linux, Python 3.10, PyTorch 2.5.1 and CUDA 12.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./sam3
```

## Data preparation

Organize the official RIS-LAD splits as SAM3-I annotations and RGB images:

```text
data/rislad/
├── annotations/
│   ├── sam3i_train.json
│   ├── sam3i_val.json
│   └── sam3i_test.json
└── images/
    └── <file_name from annotation>

checkpoints/sam3i_pretrained.pt
```

Image paths and annotation locations are supplied through command-line
arguments. Run the commands below from the repository root; `--gpu` specifies
the CUDA device.

## Training

### SAM3-I backbone

```bash
python scripts/run.py train-backbone --stage stage1_2 \
  --annotations data/rislad/annotations --images data/rislad/images \
  --init checkpoints/sam3i_pretrained.pt --gpu 0

python scripts/run.py train-backbone --stage stage3 \
  --annotations data/rislad/annotations --images data/rislad/images --gpu 0
```

Stage1_2 trains for 8 epochs. Stage3 initializes from the trained Stage1_2
checkpoint and trains for 4 epochs. The training configurations specify the
adapter setup, optimizer, losses and image transforms.

### EvidenceScorer

```bash
python scripts/run.py cache \
  --annotations data/rislad/annotations --images data/rislad/images \
  --sam-checkpoint outputs/rislad/stage3/checkpoints/checkpoint_4.pt \
  --out outputs/rislad/cache --batch-size 8 --gpu 0

python scripts/run.py train-evidence \
  --train-cache outputs/rislad/cache/train/records \
  --val-cache outputs/rislad/cache/val/records \
  --out outputs/rislad/evidence --batch-size 4 --gpu 0
```

The cache contains frozen SAM3-I features and candidate observations.
EvidenceScorer training uses a 90/10 internal split of the training images,
seed 123, and 30 epochs. Training crop proposals and target beliefs are
supervised by the training annotations.

## Evaluation

Generate image-level candidate predictions:

```bash
python scripts/run.py global \
  --ann data/rislad/annotations/sam3i_test.json --images data/rislad/images \
  --sam-checkpoint outputs/rislad/stage3/checkpoints/checkpoint_4.pt \
  --out outputs/rislad/test/global --gpu 0
```

Evaluate DRA-SAM using the candidate predictions and trained EvidenceScorer:

```bash
python scripts/run.py evaluate \
  --ann data/rislad/annotations/sam3i_test.json --images data/rislad/images \
  --sam-checkpoint outputs/rislad/stage3/checkpoints/checkpoint_4.pt \
  --evidence-checkpoint outputs/rislad/evidence/checkpoint_30.pt \
  --baseline outputs/rislad/test/global/global_all.json \
  --out outputs/rislad/test/dra_sam --tau 1.2 --unknown-weight 1.0 \
  --gpu 0
```

For validation, use `sam3i_val.json` and a corresponding output directory.
Results include mIoU, oIoU, precision at IoU thresholds 0.5–0.9 and per-query
records. `--save-masks` exports segmentation masks. Inference parameters are
defined in `configs/rislad.yaml`.

## RIS-LAD results

The paper configuration uses EvidenceScorer epoch 30, posterior temperature
1.2 and unknown utility weight 1.0.

| Split | mIoU | oIoU | P@0.5 |
|---|---:|---:|---:|
| Val | 54.145 | 56.657 | 59.026 |
| Test | 52.338 | 53.451 | 58.053 |

## Acknowledgments

DRA-SAM builds on [SAM3-I](https://github.com/debby-0527/SAM3-I) and
[SAM3](https://github.com/facebookresearch/sam3).
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licensing.
