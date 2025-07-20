import os
from pathlib import Path


def kagglehub_dir():
    dir = os.environ.get("KAGGLEHUB_CACHE_DIR")
    if not dir:
        dir = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub")
    return Path(dir) / "datasets"


def _convert_tcp_flag_to_int(flag: str) -> int:
    if flag[:2] == "0x":
        return int(flag, 16)
    numflag = 0
    for i, bit in enumerate(["F", "S", "R", "P", "A"]):
        if bit in flag:
            numflag += 1 << i
    return numflag
