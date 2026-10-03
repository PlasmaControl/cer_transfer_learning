"""Epoch-level metrics logging.

CSVLogger: dependency-free, append-mode (requeue-safe — a resumed job
continues the same file), one row per epoch, header written once.

TensorBoardLogger: optional; imports torch.utils.tensorboard lazily so the
'tensorboard' package is only required when actually used.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional


class CSVLogger:
    """Append one row per epoch to a CSV file (header from the first row).

    Parameters
    ----------
    path : Path
        CSV file; created on the first ``log`` call.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: Optional[list[str]] = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path) as f:
                self._fieldnames = next(csv.reader(f))

    def log(self, row: dict) -> None:
        """Write one row.

        Parameters
        ----------
        row : dict
            Column name -> value; the first call fixes the columns.
        """
        row = {**row, "wall_time": f"{time.time():.0f}"}
        mode = "a"
        if self._fieldnames is None:
            self._fieldnames = list(row)
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                w.writeheader()
                w.writerow(row)
            return
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in self._fieldnames})

    def close(self) -> None:
        """Close the file."""
        pass


class TensorBoardLogger:
    """Write scalar metrics to a TensorBoard run directory.

    Parameters
    ----------
    logdir : Path
    """

    def __init__(self, logdir: Path):
        from torch.utils.tensorboard import SummaryWriter  # lazy import

        self.writer = SummaryWriter(log_dir=str(logdir))

    def log(self, row: dict) -> None:
        """Log all numeric entries of a row against its ``epoch``.

        Parameters
        ----------
        row : dict
        """
        step = int(row.get("epoch", 0))
        for k, v in row.items():
            if k == "epoch":
                continue
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.flush()
        self.writer.close()
