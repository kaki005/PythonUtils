import os
import pathlib
import sys

import rich
from utilpy import LapTime, StringBuilder, format_directory


def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def main():
    with LapTime("hello"):
        fibonacci(20)

    sb = StringBuilder()
    sb.Append("aaa")
    sb.AppendLine()
    sb.AppendLine("bbb")
    print(sb)

    try:
        directory = os.path.abspath(sys.argv[1])
    except IndexError:
        print("[b]Usage:[/] python tree.py <DIRECTORY>")
    else:
        tree = format_directory(pathlib.Path(directory))
        rich.print(tree)


if __name__ == "__main__":
    main()
