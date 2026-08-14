# IRIS — Semiconductor Image Restoration

Phase 1 is the dataset audit.

Expected dataset:
```
train/
├── NoisyLR/
│   ├── 000040.npy
│   └── ...
└── GT/
    ├── 000040.npy
    └── ...
```

Files with the same filename are paired.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Audit
```bash
python scripts/dataset_audit.py --data_root "C:\path\to\train\train"
```

The report is saved to `results/dataset_audit.json`.

Do not upload the full dataset to GitHub.
