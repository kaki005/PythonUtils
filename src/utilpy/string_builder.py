from io import StringIO

from multipledispatch import dispatch


class StringBuilder:
    def __init__(self):
        self._file_str: StringIO = StringIO()

    def Append(self, str: str):
        self._file_str.write(str)

    @dispatch()
    def AppendLine(self) -> None:  # type: ignore
        self._file_str.write("\n")

    @dispatch(str)  # type: ignore[no-redef]
    def AppendLine(self, message: str) -> None:  # noqa: F811
        self._file_str.write(f"{message}\n")

    def __str__(self):
        return self._file_str.getvalue()
