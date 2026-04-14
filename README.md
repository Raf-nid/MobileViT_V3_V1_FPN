# MobileViT-FPN-PixelShuffle

Research codebase for **ultrasound FMC-style** reconstruction experiments: MobileViT-style backbones, FPN / UNet-style decoders, **pixel shuffle** heads, optional **Mixture-of-Experts (MoE)** and **DANN** variants. Documentation and primary code comments are in **English**.

## Related conference paper

This repository supports work presented in connection with the [**Acoustical Society of America — Honolulu 2025**](https://acousticalsociety.org/honolulu-2025/):

**Enabling low-cost full matrix capture acquisition using MobileViT with binary ultrasonic data.**

**Authors:** Rafael Niddam (École de technologie supérieure, 1100 Notre-Dame St. W., A-1340, Montréal, QC H3C 1K3, Canada, Guillaume Painchaud-April (Evident Sci., QC, Canada), Alain Le Duff (Evident Sci., QC, Canada), and Pierre Belanger (École de technologie supérieure, Montréal, QC, Canada).

**Contact:** [Rafael Niddam on LinkedIn](https://www.linkedin.com/in/rafael-niddam/)

## Features

- **Models**: MobileViT v3 family (FPN, UNet, SegFormer, SPT, LSA, pixel variants), Lite MobileViT blocks, MobileNet v2 FPN/UNet, MoE and DANN wrappers (see `models/__init__.py` registry). The **final FMC pixel-shuffle segmentation model** is `MobileViTv3_v1_dynamicFPNpixel2` implemented in `models/segmentation/mobilevit_v3_pixel2.py`.
- **Training**: `training/train.py` (argparse, `--list-models`), plus `training/train_moe.py` and `training/train_dann.py` for specialized setups.
- **Evaluation & analysis**: use **`evaluation/evaluate_loop2.py`** as the main FMC validation script; **`evaluation/evaluate_moe2.py`** for MoE. A trailing `2` in a script name marks the final version in that line (see `evaluation/__init__.py`).
- **Utilities**: losses, NCC, plotting helpers in `utils/`.

## Repository layout

```text
MobileViT-FPN-PixelShuffle/
├── configs/          # Hyperparameters and training defaults
├── evaluation/       # Evaluation and post-hoc analysis
├── models/           # Backbones, decoders, segmentation heads, MoE
├── training/         # Training entry points
├── utils/            # Metrics, losses, FMC visualization helpers
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

**Requirements**

- Python **3.10+**
- **CUDA** GPU recommended for training (CPU may work for tiny tests)

**Install**

```bash
python -m venv .venv
```

Activate:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- Linux / macOS: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

## Usage

**List registered model names**

```bash
python training/train.py --list-models
```

**Train** (example; adjust `--model` and `--batch_size` as needed)

```bash
python training/train.py --model MobileViTv3_v1_dynamicFPNpixel2 --batch_size 16
```

**Evaluate** (final workflow: loop presets + project paths)

```bash
python evaluation/evaluate_loop2.py --preset 0
# Optional: checkpoint and folder under data/ (e.g. Test_dataset) or absolute path
python evaluation/evaluate_loop2.py --preset 0 --checkpoint path/to/Model.pth --test-dir Test_dataset
```

Preset 0 reads `.mat` files from **`data/Test_dataset/`**. Other presets use **`data/<subdir>/`** (see `test_subdir` in `evaluate_loop2.py`). Weights: `weights/`, or `--checkpoint`. Older scripts: `evaluation/evaluate.py`, `evaluation/evaluate_loop.py`.

Additional trainers (older, script-style imports—may require aligning `sys.path` or project root with your layout):

- `python training/train_moe.py`
- `python training/train_dann.py`

## Configuration & data paths

- Defaults live in `configs/` (e.g. `config_mobileunet.py`).
- Training: `data/train_dataset` / `data/valid_dataset` (see `training/train.py`). Evaluation: `evaluate_loop2.py` — preset 0 uses `data/Test_dataset/`; other presets use `data/<subdir>/` + `weights/` (or `--checkpoint` / `--test-dir`).

## License

This project is released under the **MIT License**; see [`LICENSE`](LICENSE).
