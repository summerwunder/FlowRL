import os
import datetime
import numpy as np
import pandas as pd
from termcolor import colored


CONSOLE_FORMAT = [
    ("step", "I", "int"),
    ("episode", "E", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "train": "blue",
    "eval": "green",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


class Logger:
    """Lightweight logger for FlowRL, styled after TDMPC2."""

    def __init__(self, log_dir, save_csv=True):
        self._log_dir = make_dir(log_dir)
        self._save_csv = save_csv
        self._csv = {}
        self._model_dir = make_dir(os.path.join(self._log_dir, "checkpoint"))

    @property
    def model_dir(self):
        return self._model_dir

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.1f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category_colored = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category_colored:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        self._write_csv(d, category)
        self._print(d, category)

    def _csv_value(self, value):
        """Convert common tensor/array values into CSV-friendly scalars."""
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return value.numpy().tolist()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def _write_csv(self, d, category):
        if isinstance(self._save_csv, dict):
            if not self._save_csv.get(category, True):
                return
        elif not self._save_csv:
            return
        row = {k: self._csv_value(v) for k, v in d.items()}
        self._csv.setdefault(category, []).append(row)
        pd.DataFrame(self._csv[category]).to_csv(
            os.path.join(self._log_dir, f"{category}.csv"), index=None
        )

    def finish(self):
        pass
