import os
from pathlib import Path


def kagglehub_dir():
    dir = os.environ.get("KAGGLEHUB_CACHE_DIR")
    if not dir:
        dir = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub")
    return Path(dir) / "datasets"
