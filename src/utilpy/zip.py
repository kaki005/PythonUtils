import zipfile
from pathlib import Path


def create_zip_archive(outputdir: Path, zip_path: Path):
    """
    指定されたディレクトリの内容をZIPアーカイブに圧縮します。

    Args:
        outputdir (Path): 圧縮したいディレクトリのパス。
        zip_path (Path): 生成するZIPファイルのパス。
    """
    if not outputdir.is_dir():
        raise ValueError(f"'{outputdir}' is not a directory.")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:  # zipfile.ZIP_DEFLATEDで圧縮
        for file in outputdir.rglob("*"):  # outputdir以下のすべてのファイルとディレクトリを再帰的に取得
            if file.is_file():  # ファイルのみを対象とする
                # ZIPファイル内でのパス名 (outputdirからの相対パス)
                arcname = file.relative_to(outputdir)
                # 第1引数に実際のファイルのパス (file) を渡す
                # arcname にはZIP内でのパス (arcname) を渡す
                zipf.write(file, arcname=arcname)
                print(f"  append : {arcname}")

    print(f"'{zip_path}' was created successfully")
