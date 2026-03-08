#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path


EPOCH_LINE_RE = re.compile(
    r"\[(?P<phase>[A-Z_]+)\]\s+Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+\|\s+(?P<body>.+)"
)
KV_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>-?\d+(?:\.\d+)?)")


@dataclass(slots=True)
class MetricPoint:
    phase: str
    epoch: int
    total_epochs: int
    train_loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None
    seq_len: int | None = None


def _phase_order(phase: str) -> tuple[int, str]:
    order = {
        "PRETRAIN": 0,
        "DROPE": 1,
        "FINETUNE": 2,
        "TRAIN": 3,
    }
    return (order.get(phase, 99), phase)


def parse_training_log(path: Path) -> list[MetricPoint]:
    rows: dict[tuple[str, int, int], MetricPoint] = {}
    if not path.exists():
        return []

    with path.open() as f:
        for raw_line in f:
            match = EPOCH_LINE_RE.search(raw_line)
            if not match:
                continue

            phase = match.group("phase")
            epoch = int(match.group("epoch"))
            total_epochs = int(match.group("total"))
            key = (phase, epoch, total_epochs)
            point = rows.setdefault(key, MetricPoint(phase=phase, epoch=epoch, total_epochs=total_epochs))

            for metric_match in KV_RE.finditer(match.group("body")):
                metric_name = metric_match.group("key")
                raw_value = metric_match.group("value")
                if metric_name == "loss":
                    point.train_loss = float(raw_value)
                elif metric_name == "train_loss":
                    point.train_loss = float(raw_value)
                elif metric_name == "val_loss":
                    point.val_loss = float(raw_value)
                elif metric_name == "lr":
                    point.lr = float(raw_value)
                elif metric_name == "seq_len":
                    point.seq_len = int(raw_value)

    return sorted(rows.values(), key=lambda row: (_phase_order(row.phase), row.epoch))


def _moving_average(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values

    result: list[float | None] = []
    for idx in range(len(values)):
        window_values = [v for v in values[max(0, idx - window + 1): idx + 1] if v is not None]
        result.append(sum(window_values) / len(window_values) if window_values else None)
    return result


def save_plot(
    points: list[MetricPoint],
    output_path: Path,
    *,
    title: str | None = None,
    smooth_window: int = 1,
) -> None:
    if not points:
        raise ValueError("No training metrics were found in the log.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the active environment first."
        ) from exc

    phases = sorted({point.phase for point in points}, key=_phase_order)
    fig, axes = plt.subplots(len(phases), 1, figsize=(12, 4 * len(phases)), sharex=False)
    if len(phases) == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        phase_points = [point for point in points if point.phase == phase]
        epochs = [point.epoch for point in phase_points]
        train_losses = _moving_average([point.train_loss for point in phase_points], smooth_window)
        val_epochs = [point.epoch for point in phase_points if point.val_loss is not None]
        val_losses = [point.val_loss for point in phase_points if point.val_loss is not None]

        ax.plot(epochs, train_losses, label="train_loss", color="#1f77b4", linewidth=2)
        if val_epochs:
            ax.plot(val_epochs, val_losses, label="val_loss", color="#d62728", marker="o", linewidth=1.5)

        last_seq_len: int | None = None
        for point in phase_points:
            if point.seq_len is None or point.seq_len == last_seq_len:
                continue
            if last_seq_len is not None:
                ax.axvline(point.epoch, color="#888888", linestyle="--", alpha=0.35)
            ax.text(
                point.epoch,
                ax.get_ylim()[1],
                f"seq={point.seq_len}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#666666",
            )
            last_seq_len = point.seq_len

        ax.set_title(phase.title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(title or f"Training loss from {output_path.stem}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _print_latest(points: list[MetricPoint]) -> None:
    phase_groups: dict[str, list[MetricPoint]] = {}
    for point in points:
        phase_groups.setdefault(point.phase, []).append(point)

    for phase in sorted(phase_groups, key=_phase_order):
        latest = phase_groups[phase][-1]
        train_text = f"{latest.train_loss:.4f}" if latest.train_loss is not None else "-"
        val_text = f"{latest.val_loss:.4f}" if latest.val_loss is not None else "-"
        seq_text = f", seq_len={latest.seq_len}" if latest.seq_len is not None else ""
        print(
            f"{phase}: epoch {latest.epoch}/{latest.total_epochs}{seq_text}, "
            f"train_loss={train_text}, val_loss={val_text}"
        )


def _default_output_path(log_path: Path) -> Path:
    if log_path.suffix:
        return log_path.with_suffix(".png")
    return log_path.parent / f"{log_path.name}.png"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot train/validation loss from a bach-gen training log."
    )
    parser.add_argument("log_path", type=Path, help="Path to the training log file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to the log path with a .png suffix.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Re-read the log periodically and refresh the plot until interrupted.",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=30.0,
        help="Refresh interval when --watch is set (default: 30).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Apply a trailing moving average to train loss (default: 1 = no smoothing).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title override for the plot.",
    )
    parser.add_argument(
        "--print-latest",
        action="store_true",
        help="Print the latest parsed metrics after each refresh.",
    )
    args = parser.parse_args()

    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be >= 1")

    output_path = args.output or _default_output_path(args.log_path)

    if not args.watch:
        points = parse_training_log(args.log_path)
        save_plot(points, output_path, title=args.title, smooth_window=args.smooth_window)
        if args.print_latest:
            _print_latest(points)
        print(f"Wrote plot: {output_path}")
        return 0

    last_signature: tuple[int, int] | None = None
    try:
        while True:
            if args.log_path.exists():
                stat = args.log_path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                if signature != last_signature:
                    points = parse_training_log(args.log_path)
                    if points:
                        save_plot(points, output_path, title=args.title, smooth_window=args.smooth_window)
                        print(f"Updated plot: {output_path}")
                        if args.print_latest:
                            _print_latest(points)
                    else:
                        print(f"No epoch metrics found yet in {args.log_path}")
                    last_signature = signature
            else:
                print(f"Waiting for log: {args.log_path}")
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
