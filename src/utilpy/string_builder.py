from io import StringIO


class StringBuilder:
    def __init__(self):
        self._file_str: StringIO = StringIO()

    def Append(self, str):
        self._file_str.write(str)

    def __str__(self):
        return self._file_str.getvalue()
