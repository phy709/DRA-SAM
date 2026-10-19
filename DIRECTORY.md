# Project structure

```text
DRA-SAM/
├── README.md
├── DIRECTORY.md
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
├── SOURCE_MANIFEST.json
├── .gitignore
├── .gitattributes
├── configs/
│   └── rislad.yaml
├── scripts/
│   ├── run.py
│   └── inference.py
├── tools/
│   ├── build_cache.py
│   ├── data_utils.py
│   ├── prepare_internal_split.py
│   ├── rai_data.py
│   ├── train_evidence.py
│   ├── metrics.py
│   └── evaluate.py
├── docs/
│   ├── DATA.md
│   └── CONFIGURATION.md
└── sam3/
    ├── LICENSE
    ├── README.md
    ├── pyproject.toml
    ├── assets/
    │   └── bpe_simple_vocab_16e6.txt.gz
    └── sam3/
        ├── __init__.py
        ├── logger.py
        ├── model_builder.py
        ├── refzoom/
        │   ├── __init__.py
        │   ├── analytic_rai.py
        │   ├── analytic_rai_loss.py
        │   └── sam3_adapter.py
        ├── train/
        │   ├── train.py
        │   ├── trainer.py
        │   ├── configs/dra_sam/
        │   │   ├── rislad_stage1_2.yaml
        │   │   ├── rislad_stage1_2_continue.yaml
        │   │   └── rislad_stage3.yaml
        │   ├── data/
        │   ├── loss/
        │   ├── optim/
        │   ├── transforms/
        │   └── utils/
        ├── model/
        ├── eval/
        ├── perflib/
        └── sam/
```
