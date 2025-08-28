import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


def remove_na_or_inf(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    指定されたDataFrameの列から、NaNまたは無限大を含む行を削除します。

    Args:
        df (pd.DataFrame): 処理対象のDataFrame。
        col_name (str): NaNまたは無限大をチェックする列の名前。

    Returns:
        pd.DataFrame: NaNまたは無限大の行が削除された新しいDataFrame。
                      元のDataFrameは変更されません。
    """
    try:
        df[col_name] = df[col_name].astype(float)
    except ValueError as e:
        logging.error(f"Column '{col_name}' contains non-numeric values that cannot be converted to float: {e}")
        return df  # あるいはエラーを再スロー
    is_inf = np.isinf(df[col_name])  # 2. 無限大 (inf, -inf) のチェック
    is_na = df[col_name].isna()  # 3. 欠損値 (NaN) のチェック
    rows_to_drop = is_inf | is_na  # 4. 削除対象の行を特定
    return df[~rows_to_drop]  # 6. 該当する行を削除して新しいDataFrameを返す


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


def to_datetime(value: pd.Timestamp | np.datetime64) -> datetime.datetime:
    if isinstance(value, pd.Timestamp):  # Pandas Timestamp の場合
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):  # NumPy datetime64 の場合
        return pd.Timestamp(value).to_pydatetime()  # Pandas経由が最もロバスト
    raise Exception


def datetime_base(timestamps: np.ndarray, base_time: pd.Timestamp, freq: str, time_scale: float = 1.0):
    diff = (timestamps - np.array(base_time)).astype("timedelta64[us]")
    return time_scale * (diff / freq_to_timedelta64(freq))


def datetime_diff(timestamps: np.ndarray, freq: str, time_scale: float = 1.0):
    diff = np.diff(timestamps).astype("timedelta64[us]")
    convert_diff = time_scale * (diff / freq_to_timedelta64(freq))
    return convert_diff


def freq_to_timedelta64(freq_str):
    """pandasのfreqからnp.timedelta64に変換します"""
    offset = pd.tseries.frequencies.to_offset(freq_str)
    return np.timedelta64(offset.delta)
