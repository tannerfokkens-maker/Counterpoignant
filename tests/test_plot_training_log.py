from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "plot_training_log.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_training_log", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_training_log_dedupes_progress_and_info_lines(tmp_path):
    module = _load_module()
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "\n".join(
            [
                "[03/08/26 00:00:00] INFO     [PRETRAIN] Epoch 5/80 | train_loss=2.3456 | seq_len=4096 | train_cat[pitch=1.5000, octave=1.2000] | lr=0.000300",
                "[PRETRAIN] Epoch 10/80 | loss=2.1000 | val_loss=2.0000",
                "[03/08/26 00:10:00] INFO     [PRETRAIN] Epoch 10/80 | train_loss=2.1000 | seq_len=4096 | train_cat[pitch=1.4000, octave=1.1000] | val_loss=2.0000 | val_cat[pitch=1.3500, octave=1.0500] | lr=0.000298",
                "[03/08/26 00:20:00] INFO     [DROPE] Epoch 1/20 | train_loss=1.9000 | train_cat[pitch=1.2500] | val_loss=1.8500 | val_cat[pitch=1.2000] | lr=0.001000",
            ]
        )
        + "\n"
    )

    points = module.parse_training_log(log_path)

    assert [(point.phase, point.epoch) for point in points] == [
        ("PRETRAIN", 5),
        ("PRETRAIN", 10),
        ("DROPE", 1),
    ]
    assert points[1].train_loss == 2.1
    assert points[1].val_loss == 2.0
    assert points[0].train_pitch_loss == 1.5
    assert points[1].train_pitch_loss == 1.4
    assert points[1].val_pitch_loss == 1.35
    assert points[1].seq_len == 4096
    assert points[2].train_pitch_loss == 1.25
    assert points[2].val_pitch_loss == 1.2
    assert points[2].lr == 0.001


def test_save_plots_adds_pitch_companion_file(tmp_path):
    module = _load_module()
    output_path = tmp_path / "loss.png"
    calls = []

    def fake_save_metric_plot(points, output_path, **kwargs):
        calls.append((output_path, kwargs["train_attr"], kwargs["val_attr"]))

    module._save_metric_plot = fake_save_metric_plot
    points = [
        module.MetricPoint(
            phase="PRETRAIN",
            epoch=1,
            total_epochs=10,
            train_loss=1.0,
            val_loss=0.9,
            train_pitch_loss=1.2,
            val_pitch_loss=1.1,
        )
    ]

    output_paths = module.save_plots(points, output_path)

    assert output_paths == [output_path, tmp_path / "loss_pitch.png"]
    assert calls == [
        (output_path, "train_loss", "val_loss"),
        (tmp_path / "loss_pitch.png", "train_pitch_loss", "val_pitch_loss"),
    ]
