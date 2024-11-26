import logging

import rich
import rich.highlighter
import rich.pretty as pretty
from rich.logging import RichHandler


class OriginalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        return pretty.Pretty(record.msg)


def log_init(level=logging.INFO):
    handler = RichHandler(markup=True, rich_tracebacks=True)
    # handler.setFormatter(OriginalFormatter())
    logging.basicConfig(
        level=level,
        # # format="[%(filename)s:%(lineno)d] %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
