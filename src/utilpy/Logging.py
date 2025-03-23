import logging

import rich.pretty as pretty
from catppuccin.extras.rich_ctp import frappe, latte, macchiato, mocha
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


class OriginalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        return pretty.Pretty(record.msg)


def log_init(theme: Theme | None = mocha, level=logging.INFO):
    console = Console(theme=theme) if theme is not None else None
    handler = RichHandler(console=console, markup=True, rich_tracebacks=True)
    # handler.setFormatter(OriginalFormatter())
    logging.basicConfig(
        level=level,
        # # format="[%(filename)s:%(lineno)d] %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
