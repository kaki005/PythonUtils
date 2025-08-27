from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_sparse_matrix(
    path: Path, row_name: str, col_name: str, return_labels: bool = False
) -> tuple[sparse.coo_matrix | np.ndarray, np.ndarray | None]:
    """
    save_sparse_matrixで保存されたCSVファイルから疎行列データを読み込み、
    元の密行列またはCOO形式の疎行列を再構築します。

    Args:
        path (Path): 読み込むCSVファイルのパス。
        row_name (str): 行インデックスの列名。
        col_name (str): 列インデックスの列名。
        return_labels (bool): row_labelsが保存されている場合、それも一緒に返すかどうか。

    Returns:
        tuple:
            - scipy.sparse.coo_matrix または np.ndarray: 再構築された疎行列。
                                                          NumPy配列として返す場合は .toarray() を呼び出す。
            - np.ndarray or None: row_labels が保存されており、return_labels=True の場合に
                                  対応する row_labels の配列。それ以外は None。
    """
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    # CSVファイルを読み込む
    df = pd.read_csv(path)

    # 必要な列が存在するか確認
    required_cols = [row_name, col_name, "value"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSVファイルに必須の列 '{col}' が見つかりません。")

    # 行インデックス、列インデックス、値を取得
    rows = df[row_name].values
    cols = df[col_name].values
    data = df["value"].values

    # 行列の形状を推測する (最大インデックス + 1)
    # これが元の行列の形状を正確に反映していない場合もあるので注意
    # 正確な形状が必要な場合は、別途保存するか、事前に知っている必要があります。
    num_rows = rows.max() + 1
    num_cols = cols.max() + 1

    # COO形式の疎行列を再構築
    reconstructed_sparse_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    row_labels_data = None
    if return_labels and f"{row_name}_labels" in df.columns:
        # row_labels_data は非ゼロ要素に対応するラベルのみ
        row_labels_data, first_indices = np.unique(df[f"{row_name}_labels"].values, return_index=True)
        row_labels_data = row_labels_data[np.argsort(first_indices)]
    return reconstructed_sparse_matrix, row_labels_data


def save_sparse_matrix(
    path: Path, matrix: np.ndarray, row_name: str, col_name: str, row_labels: list[str] | None = None
):
    sparse_matrix = sparse.coo_matrix(matrix)  # 座標形式に変換
    if row_labels is None:
        df = pd.DataFrame(
            {
                row_name: sparse_matrix.row,
                col_name: sparse_matrix.col,
                "value": sparse_matrix.data,
            }
        )
    else:
        df = pd.DataFrame(
            {
                row_name: sparse_matrix.row,
                col_name: sparse_matrix.col,
                f"{row_name}_labels": np.array(row_labels)[sparse_matrix.row],
                "value": sparse_matrix.data,
            }
        )
    df.to_csv(path, index=False)
