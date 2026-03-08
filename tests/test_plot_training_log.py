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
                "[03/08/26 00:00:00] INFO     [PRETRAIN] Epoch 5/80 | train_loss=2.3456 | seq_len=4096 | lr=0.000300",
                "[PRETRAIN] Epoch 10/80 | loss=2.1000 | val_loss=2.0000",
                "[03/08/26 00:10:00] INFO     [PRETRAIN] Epoch 10/80 | train_loss=2.1000 | seq_len=4096 | val_loss=2.0000 | lr=0.000298",
                "[03/08/26 00:20:00] INFO     [DROPE] Epoch 1/20 | train_loss=1.9000 | val_loss=1.8500 | lr=0.001000",
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
    assert points[1].seq_len == 4096
    assert points[2].lr == 0.001
