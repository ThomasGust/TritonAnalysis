"""Train an Ultralytics YOLO detector for European green crabs."""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_workspace import workspace_paths  # noqa: E402


DEFAULT_BASE_MODEL = "yolo11n.pt"


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for YOLO training. Install it with:\n"
            "  python -m pip install ultralytics\n"
            "or reinstall TritonAnalysis requirements."
        ) from exc
    return YOLO


def latest_synthetic_dataset() -> Path | None:
    """Return the most recent generated crab YOLO data.yaml, if available."""
    dataset_root = workspace_paths().root / "datasets"
    candidates = sorted(
        dataset_root.glob("crab_green_yolo_*/data.yaml"),
        key=lambda path: path.parent.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_data_yaml(value: str | Path | None) -> Path:
    """Resolve a data.yaml path from an explicit file, dataset folder, or latest dataset."""
    if value is None:
        latest = latest_synthetic_dataset()
        if latest is None:
            raise SystemExit("No dataset provided and no Workspace/datasets/crab_green_yolo_*/data.yaml was found.")
        return latest.resolve()

    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "data.yaml"
    if not path.exists():
        raise SystemExit(f"Could not find YOLO data.yaml: {path}")
    return path.resolve()


def _torch_device_is_supported() -> bool:
    try:
        import torch
    except Exception:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
        except Exception:
            return False
    return arch in supported_arches


def choose_training_device(explicit: str | None) -> str:
    """Choose a conservative training device for the local PyTorch build."""
    if explicit:
        return explicit
    return "0" if _torch_device_is_supported() else "cpu"


def default_project_dir() -> Path:
    return workspace_paths(create=True).root / "models" / "crab_yolo"


def resolve_base_model(value: str | Path) -> str:
    """Prefer a local workspace cache for the default pretrained model."""
    raw_value = str(value)
    explicit_path = Path(raw_value).expanduser()
    if explicit_path.exists():
        return str(explicit_path.resolve())
    if raw_value == DEFAULT_BASE_MODEL:
        cached = workspace_paths(create=True).root / "models" / "base" / DEFAULT_BASE_MODEL
        if cached.exists():
            return str(cached.resolve())
    return raw_value


def default_run_name() -> str:
    return f"green_crab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an Ultralytics YOLO model for European green crab boxes.")
    parser.add_argument(
        "data",
        nargs="?",
        help="Path to data.yaml or a YOLO dataset folder. Defaults to the latest Workspace/datasets/crab_green_yolo_*.",
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base YOLO weights or config.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Use -1 for Ultralytics auto batch.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example cpu, 0, or 0,1. Defaults to GPU only when PyTorch supports it.",
    )
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. 0 is safest on Windows.")
    parser.add_argument("--project", type=Path, default=None, help="Output project folder.")
    parser.add_argument("--name", default=None, help="Run name below the project folder.")
    parser.add_argument("--seed", type=int, default=20260531, help="Training seed.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument("--cache", action="store_true", help="Cache images during training.")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics training plots.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Ultralytics run.")
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N model layers during fine tuning.",
    )
    parser.add_argument("--optimizer", default=None, help="Ultralytics optimizer, for example SGD, AdamW, or auto.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate fraction.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate scheduling.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optimizer weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs.")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, help="Warmup bias learning rate.")
    parser.add_argument("--warmup-momentum", type=float, default=None, help="Warmup momentum.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic this many epochs before the end.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability.")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste augmentation probability.")
    parser.add_argument("--erasing", type=float, default=None, help="Random erasing probability.")
    parser.add_argument("--degrees", type=float, default=None, help="Random rotation degrees.")
    parser.add_argument("--translate", type=float, default=None, help="Random translation fraction.")
    parser.add_argument("--scale", type=float, default=None, help="Random scale gain.")
    parser.add_argument("--shear", type=float, default=None, help="Random shear degrees.")
    parser.add_argument("--perspective", type=float, default=None, help="Random perspective fraction.")
    parser.add_argument("--flipud", type=float, default=None, help="Vertical flip probability.")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability.")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation gain.")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation gain.")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation gain.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_yaml = resolve_data_yaml(args.data)
    device = choose_training_device(args.device)
    project = (args.project or default_project_dir()).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    name = args.name or default_run_name()

    print(f"Training data: {data_yaml}")
    base_model = resolve_base_model(args.model)
    print(f"Base model: {base_model}")
    print(f"Device: {device}")
    print(f"Output: {project / name}")
    if device == "cpu":
        print("Using CPU because no compatible CUDA device is available to this PyTorch build.")

    YOLO = _import_yolo()
    model = YOLO(base_model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "project": str(project),
        "name": name,
        "seed": args.seed,
        "patience": args.patience,
        "fraction": args.fraction,
        "cache": args.cache,
        "plots": args.plots,
        "resume": args.resume,
        "val": True,
    }
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    optional_train_args = {
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": True if args.cos_lr else None,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "warmup_momentum": args.warmup_momentum,
        "close_mosaic": args.close_mosaic,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "erasing": args.erasing,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
    }
    train_kwargs.update({key: value for key, value in optional_train_args.items() if value is not None})
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", project / name)).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    print(f"Run directory: {save_dir}")
    print(f"Best weights: {best if best.exists() else 'not written'}")
    print(f"Last weights: {last if last.exists() else 'not written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
