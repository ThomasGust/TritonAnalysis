"""Evaluate a crab YOLO checkpoint on synthetic and real-world sanity sets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["european_green_crab", "native_rock_crab", "jonah_crab"]


@dataclass(frozen=True)
class Box:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    conf: float | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.xc - self.w / 2.0,
            self.yc - self.h / 2.0,
            self.xc + self.w / 2.0,
            self.yc + self.h / 2.0,
        )


@dataclass(frozen=True)
class PredictTarget:
    key: str
    source: Path
    manual_labels: Path | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="YOLO checkpoint to evaluate.")
    parser.add_argument("--data", required=True, type=Path, help="Dataset data.yaml for synthetic validation.")
    parser.add_argument("--project", default=Path("Workspace/runs/crab_yolo"), type=Path, help="YOLO output project folder.")
    parser.add_argument("--name-prefix", default=None, help="Prefix for generated YOLO eval run folders.")
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Parse existing output folders instead of rerunning YOLO.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    project = (repo / args.project).resolve() if not args.project.is_absolute() else args.project.resolve()
    weights = (repo / args.weights).resolve() if not args.weights.is_absolute() else args.weights.resolve()
    data_yaml = (repo / args.data).resolve() if not args.data.is_absolute() else args.data.resolve()
    name_prefix = args.name_prefix or weights.parents[1].name
    yolo = repo / ".venv" / "Scripts" / "yolo.exe"
    if not yolo.exists():
        yolo = Path("yolo")

    if not weights.exists():
        raise FileNotFoundError(weights)
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    project.mkdir(parents=True, exist_ok=True)
    summary_dir = project / f"{name_prefix}_evaluation"
    summary_dir.mkdir(parents=True, exist_ok=True)

    val_name = f"{name_prefix}_val"
    if not args.skip_val and not args.reuse:
        run(
            [
                yolo,
                "detect",
                "val",
                f"model={weights}",
                f"data={data_yaml}",
                f"imgsz={args.imgsz}",
                f"batch={args.batch}",
                f"device={args.device}",
                f"workers={args.workers}",
                f"project={project}",
                f"name={val_name}",
                "exist_ok=True",
                "plots=True",
            ]
        )

    predict_targets = discover_predict_targets(repo)
    predict_results: dict[str, dict[str, object]] = {}
    if not args.skip_predict:
        for target in predict_targets:
            out_name = f"{name_prefix}_{target.key}_conf{int(args.conf * 1000):03d}"
            out_dir = project / out_name
            if not args.reuse:
                run(
                    [
                        yolo,
                        "detect",
                        "predict",
                        f"model={weights}",
                        f"source={target.source}",
                        f"imgsz={args.imgsz}",
                        f"conf={args.conf}",
                        f"iou={args.iou}",
                        f"device={args.device}",
                        "save=True",
                        "save_txt=True",
                        "save_conf=True",
                        f"project={project}",
                        f"name={out_name}",
                        "exist_ok=True",
                    ]
                )
            result = summarize_prediction_dir(out_dir)
            if target.manual_labels and target.manual_labels.exists():
                manual = read_boxes(target.manual_labels, has_conf=False)
                predicted = read_prediction_boxes_for_source(out_dir, target.source)
                result["manual_truth"] = class_counts(manual)
                result["manual_iou_025"] = match_boxes(manual, predicted, threshold=0.25)
                result["manual_iou_050"] = match_boxes(manual, predicted, threshold=0.50)
            predict_results[target.key] = result

    val_run = project / val_name
    summary = {
        "weights": str(weights),
        "data": str(data_yaml),
        "project": str(project),
        "val_run": str(val_run) if val_run.exists() else None,
        "predict_results": predict_results,
    }
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (summary_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {summary_dir / 'summary.md'}")
    return 0


def discover_predict_targets(repo: Path) -> list[PredictTarget]:
    crab_runs = repo / "Workspace" / "runs" / "crab_yolo"
    pilot = repo.parent / "TritonPilot"
    candidates = [
        PredictTarget("mate_samples", repo / "Workspace" / "crab_mate_samples"),
        PredictTarget(
            "stereo_pair000018_full",
            pilot
            / "recordings_archive"
            / "recordings"
            / "stereo_sessions"
            / "20260525-083113"
            / "right"
            / "pair_000018_right.png",
            crab_runs / "manual_annotations" / "pair_000018_right_manual_boxes_yolo_full.txt",
        ),
        PredictTarget(
            "stereo_pair000018_boardcrop",
            crab_runs / "stereo_pair000018_right_diagnostics" / "pair_000018_right_board_crop.png",
            crab_runs / "manual_annotations" / "pair_000018_right_manual_boxes_yolo_crop.txt",
        ),
        PredictTarget(
            "murky_pool_boardcrop",
            crab_runs / "murky_pool_diagnostics" / "Arm Camera_20260503-190906_board_crop.png",
        ),
        PredictTarget(
            "murky_pool_boardcrop_wbclahe",
            crab_runs / "murky_pool_diagnostics" / "Arm Camera_20260503-190906_board_crop_wb_clahe.png",
        ),
    ]
    return [target for target in candidates if target.source.exists()]


def run(command: list[Path | str]) -> None:
    printable = " ".join(str(part) for part in command)
    print(printable, flush=True)
    subprocess.run([str(part) for part in command], check=True)


def summarize_prediction_dir(out_dir: Path) -> dict[str, object]:
    labels_dir = out_dir / "labels"
    rows: list[dict[str, object]] = []
    counts = {name: 0 for name in CLASS_NAMES}
    if labels_dir.exists():
        for path in sorted(labels_dir.glob("*.txt")):
            boxes = read_boxes(path, has_conf=True)
            file_counts = class_counts(boxes)
            for name, count in file_counts.items():
                counts[name] += count
            rows.append(
                {
                    "file": path.name,
                    "total": len(boxes),
                    "counts": file_counts,
                    "mean_conf": round(mean([box.conf or 0.0 for box in boxes]), 4) if boxes else None,
                }
            )
    return {"run_dir": str(out_dir), "total": sum(counts.values()), "counts": counts, "files": rows}


def read_prediction_boxes_for_source(out_dir: Path, source: Path) -> list[Box]:
    label_path = out_dir / "labels" / f"{source.stem}.txt"
    if not label_path.exists():
        return []
    return read_boxes(label_path, has_conf=True)


def read_boxes(path: Path, has_conf: bool) -> list[Box]:
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        conf = float(parts[5]) if has_conf and len(parts) >= 6 else None
        boxes.append(Box(int(float(parts[0])), *(float(value) for value in parts[1:5]), conf=conf))
    return boxes


def class_counts(boxes: list[Box]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for box in boxes:
        if 0 <= box.class_id < len(CLASS_NAMES):
            counts[CLASS_NAMES[box.class_id]] += 1
    return counts


def match_boxes(truth: list[Box], predicted: list[Box], threshold: float) -> dict[str, object]:
    matched_truth: set[int] = set()
    matches: list[dict[str, object]] = []
    false_positive = 0
    ordered_predictions = sorted(enumerate(predicted), key=lambda item: item[1].conf or 0.0, reverse=True)
    for _, pred in ordered_predictions:
        best_iou = 0.0
        best_index = None
        for truth_index, gt in enumerate(truth):
            if truth_index in matched_truth or gt.class_id != pred.class_id:
                continue
            overlap = iou(gt, pred)
            if overlap > best_iou:
                best_iou = overlap
                best_index = truth_index
        if best_index is None or best_iou < threshold:
            false_positive += 1
            continue
        matched_truth.add(best_index)
        matches.append(
            {
                "class": CLASS_NAMES[pred.class_id] if 0 <= pred.class_id < len(CLASS_NAMES) else str(pred.class_id),
                "iou": round(best_iou, 4),
                "conf": round(pred.conf or 0.0, 4),
            }
        )
    return {
        "threshold": threshold,
        "truth": len(truth),
        "predicted": len(predicted),
        "matched": len(matches),
        "missed": len(truth) - len(matched_truth),
        "false_positive": false_positive,
        "matches": matches,
    }


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - intersection
    return intersection / denom if denom > 0.0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Crab YOLO Evaluation",
        "",
        f"- weights: `{summary['weights']}`",
        f"- data: `{summary['data']}`",
        f"- synthetic val run: `{summary['val_run']}`" if summary.get("val_run") else "- synthetic val run: skipped",
        "",
        "## Prediction Counts",
        "",
    ]
    predict_results = summary.get("predict_results", {})
    if isinstance(predict_results, dict):
        for key, result in predict_results.items():
            if not isinstance(result, dict):
                continue
            counts = result.get("counts", {})
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"- run: `{result.get('run_dir')}`")
            lines.append(f"- total: {result.get('total', 0)}")
            if isinstance(counts, dict):
                for name in CLASS_NAMES:
                    lines.append(f"- {name}: {counts.get(name, 0)}")
            manual = result.get("manual_iou_025")
            if isinstance(manual, dict):
                lines.append(
                    "- manual IoU 0.25: "
                    f"{manual.get('matched', 0)}/{manual.get('truth', 0)} matched, "
                    f"{manual.get('missed', 0)} missed, {manual.get('false_positive', 0)} false positives"
                )
            manual_50 = result.get("manual_iou_050")
            if isinstance(manual_50, dict):
                lines.append(
                    "- manual IoU 0.50: "
                    f"{manual_50.get('matched', 0)}/{manual_50.get('truth', 0)} matched, "
                    f"{manual_50.get('missed', 0)} missed, {manual_50.get('false_positive', 0)} false positives"
                )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
