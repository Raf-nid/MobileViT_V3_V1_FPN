# -*- coding: utf-8 -*-
"""
Primary FMC validation script (recommended).

**Use this file** for batched inference, NCC/MSE metrics, and MATLAB exports.

Naming in this repo: a trailing ``2`` in a script name (e.g. ``evaluate_loop2`` vs
``evaluate_loop``, ``evaluate_moe2`` vs ``evaluate_moe``) usually marks an **improved**
successor—prefer the ``*2`` entry point unless you need the older script for comparison.

**Setup**
    - Test ``.mat`` files (HDF5 with ``FMC`` / ``Bin``): under ``data/<subdir>`` (preset 0 uses ``data/Test_dataset``). Override with ``--test-dir`` (absolute or relative to ``data/``).
    - Weights: place checkpoints under ``weights/`` (see ``checkpoint`` in each preset) or pass
      ``--checkpoint`` to override.

**Run**
    python evaluation/evaluate_loop2.py --preset 0
    python evaluation/evaluate_loop2.py --preset 3 --checkpoint /path/to/Model.pth --test-dir Test_dataset
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Project imports (run from repo root: python evaluation/evaluate_loop2.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import configs.config_mobileunet as config
from models import get_model
from utils.utils import ncc

DATA_ROOT = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def _resolve_data_dir(subdir_or_abs: str) -> str:
    """Absolute path, or ``data/<subdir>`` when relative (e.g. ``Test_dataset``, ``Verif_...``)."""
    p = Path(subdir_or_abs)
    if p.is_absolute():
        return str(p)
    return str(DATA_ROOT / subdir_or_abs)


def _resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return WEIGHTS_DIR / path_str


# Presets migrated from the legacy monolithic script. Place checkpoints under weights/ using the
# paths below (or symlink). Original lab filenames are kept in comments for traceability.
PRESETS: list[dict] = [
    {
        "run_tag": "Test_vitesse",
        "batch_size": 1,
        "model": "MobileViTv3_v1_dynamicFPNpixel2",
        "image_size": (4096, 1024),
        "mode": "xx_small4",
        "patch_size": (32, 32),
        "test_subdir": "Test_dataset",
        "checkpoint": "presets/p00_Test_vitesse_Model.pth",
    },
    {
        "run_tag": "MbViTPixel2_p4x4_XXS4_5MHz_FF8_NW_W_Normal_batchsize16_NW_plus_Wedge_BruitFixe_BruitDuet",
        "batch_size": 16,
        "model": "MobileViTv3_v1_dynamicFPNpixel2",
        "image_size": (4096, 64),
        "mode": "xx_small4",
        "patch_size": (4, 4),
        "test_subdir": "Verif_5MHz_FF8_NW_W_Amplitude_True",
        "checkpoint": "presets/p01_Model.pth",
    },
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


class MatDataset(Dataset):
    """HDF5 ``.mat`` with datasets ``FMC`` and ``Bin`` (v7.3)."""

    def __init__(self, directory: str, device: torch.device):
        self.files = sorted(glob.glob(os.path.join(directory, "*.mat")))
        if not self.files:
            raise ValueError(f"No .mat files found in directory: {directory}")
        self.filenames = [os.path.basename(f) for f in self.files]
        self.device = device

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        filepath = self.files[idx]
        base = os.path.splitext(self.filenames[idx])[0]
        truncated = base + ".mat"
        try:
            with h5py.File(filepath, "r") as f:
                fmc = torch.tensor(f["FMC"][()].astype("float32"))
                bin_data = torch.tensor(f["Bin"][()].astype("float32"))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            raise
        fmc = fmc.permute(1, 0)
        bin_data = bin_data.permute(1, 0)
        fmc = fmc.unsqueeze(0).to(self.device)
        bin_data = bin_data.unsqueeze(0).to(self.device)
        return fmc, bin_data, truncated


def safe_normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    max_val = np.max(np.abs(data))
    return data / max_val if max_val > eps else data


def create_and_save_plot(
    data: np.ndarray,
    title: str,
    filename: str,
    cmap: str = "seismic",
    vmin=None,
    vmax=None,
    colorbar_label: str = "Amplitude",
) -> None:
    plt.figure(figsize=(3, 5))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("Element axis")
    plt.ylabel("Time Increment")
    plt.yticks([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
    width = data.shape[1]
    if width == 1024:
        plt.xticks([0, 512, 1024])
    elif width == 64:
        plt.xticks([0, 64])
    else:
        plt.xticks([0, 256])
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(filename, dpi=1200, bbox_inches="tight")
    plt.close()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_forward(model, bin_batch):
    """Run one forward pass; return (batch_time_ms, output)."""
    if torch.cuda.is_available():
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        out = model(bin_batch)
        t_end.record()
        torch.cuda.synchronize()
        return float(t_start.elapsed_time(t_end)), out
    t0 = time.perf_counter()
    out = model(bin_batch)
    _sync_cuda()
    return (time.perf_counter() - t0) * 1000.0, out


def run_preset(
    preset: dict,
    device: torch.device,
    date_str: str,
    checkpoint_path: Path | None,
    test_dir_override: str | None,
) -> None:
    run_tag = preset["run_tag"]
    batch_size = int(preset["batch_size"])
    num_epochs = getattr(config, "num_epochs", "unknown")

    plt_dir = f"Evaluation_{run_tag}_{date_str}_{num_epochs}_epochs"
    os.makedirs(plt_dir, exist_ok=True)
    print(f"Output directory: {plt_dir}")

    test_data_dir = _resolve_data_dir(test_dir_override or preset["test_subdir"])
    if not os.path.isdir(test_data_dir):
        raise FileNotFoundError(f"Test data directory does not exist: {test_data_dir}")

    ckpt = checkpoint_path if checkpoint_path is not None else _resolve_checkpoint(preset["checkpoint"])
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Place weights under {WEIGHTS_DIR} or pass --checkpoint."
        )

    model = get_model(
        preset["model"],
        image_size=preset["image_size"],
        mode=preset["mode"],
        num_classes=1000,
        patch_size=preset["patch_size"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded weights from {ckpt}")

    val_dataset = MatDataset(directory=test_data_dir, device=device)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation samples: {len(val_dataset)}")

    loss_fn = nn.MSELoss()
    model.eval()
    val_losses: list[float] = []
    val_accuracy: list[float] = []

    print("Starting evaluation...")
    start_time = datetime.datetime.now()
    total_inference_time_ms = 0.0
    first_fmc_time_ms: float | None = None
    png_keep_ratio = 0.0

    matlab_dir = os.path.join(plt_dir, f"Matlab_{run_tag}")
    os.makedirs(matlab_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (amp_v, bin_v, truncated_names) in enumerate(
            tqdm(val_loader, desc="Validation")
        ):
            try:
                amp_v = amp_v.to(device)
                bin_v = bin_v.to(device)

                batch_ms, recon_v = _timed_forward(model, bin_v)

                total_inference_time_ms += batch_ms
                if first_fmc_time_ms is None:
                    first_fmc_time_ms = batch_ms / bin_v.size(0)

                loss_val = loss_fn(amp_v, recon_v)
                ncc_val = ncc(amp_v, recon_v)
                val_losses.append(loss_val.item())
                val_accuracy.append(ncc_val.item())

                print(
                    f"Batch {batch_idx + 1}: loss={loss_val.item():.6f}, NCC={ncc_val.item():.6f}"
                )

                for i in range(recon_v.size(0)):
                    base_name = truncated_names[i].replace(".mat", "")
                    amp_np = amp_v[i, 0].cpu().numpy()
                    rec_np = recon_v[i, 0].cpu().numpy()
                    bin_np = bin_v[i, 0].cpu().numpy()

                    if np.random.rand() < png_keep_ratio:
                        create_and_save_plot(
                            safe_normalize(amp_np),
                            "",
                            os.path.join(plt_dir, f"{base_name}_Amp.png"),
                            "seismic",
                            -1,
                            1,
                            "Amplitude",
                        )
                        create_and_save_plot(
                            safe_normalize(rec_np),
                            "",
                            os.path.join(plt_dir, f"{base_name}_Rec.png"),
                            "seismic",
                            -1,
                            1,
                            "Amplitude",
                        )
                        create_and_save_plot(
                            np.abs(amp_np - rec_np),
                            "",
                            os.path.join(plt_dir, f"{base_name}_Err.png"),
                            "inferno",
                            0,
                            1,
                            "Error",
                        )
                        create_and_save_plot(
                            bin_np,
                            "",
                            os.path.join(plt_dir, f"{base_name}_Bin.png"),
                            "seismic",
                            0,
                            1,
                            "Binary",
                        )

                    sio.savemat(os.path.join(matlab_dir, f"{base_name}_Amp.mat"), {"Amp": amp_np})
                    sio.savemat(os.path.join(matlab_dir, f"{base_name}_Rec.mat"), {"Rec": rec_np})

                del amp_v, bin_v, recon_v
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error on batch {batch_idx}: {e}")
                continue

    duration = datetime.datetime.now() - start_time
    print(f"Total validation wall time: {duration}")
    if first_fmc_time_ms is not None:
        print(f"First-FMC forward time (approx): {first_fmc_time_ms:.2f} ms")
    print(
        f"Sum of batch forward times: {total_inference_time_ms:.2f} ms "
        f"({total_inference_time_ms / 1000:.3f} s)"
    )

    if val_losses:
        print(f"Mean loss: {np.mean(val_losses):.6f} (std {np.std(val_losses):.6f})")
        print(f"Mean NCC: {np.mean(val_accuracy):.6f} (std {np.std(val_accuracy):.6f})")
    else:
        print("No batches completed successfully.")
        return

    print(f"Done. Outputs in: {plt_dir}")
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"Peak VRAM: {vram:.3f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="FMC validation (recommended entry point).")
    parser.add_argument(
        "--preset",
        type=int,
        default=0,
        help=f"Index into PRESETS (0..{len(PRESETS) - 1}).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Override test .mat directory (absolute path, or name/folder relative to data/, e.g. Test_dataset).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda:0 or cpu (default: cuda if available else cpu).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random).")
    args = parser.parse_args()

    if not (0 <= args.preset < len(PRESETS)):
        raise SystemExit(f"--preset must be in 0..{len(PRESETS) - 1}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")

    seed_value = args.seed if args.seed is not None else int(np.random.randint(1e6, int(1e9)))
    set_seed(seed_value)
    print(f"Seed: {seed_value}")

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_preset(PRESETS[args.preset], device, date_str, ckpt, args.test_dir)


if __name__ == "__main__":
    main()
