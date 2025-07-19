import logging
import time
from collections.abc import Callable
from functools import wraps
from logging import Logger
from typing import Any


def lap_time(timer: Callable = time.perf_counter) -> Callable:
    """wrapper for measuring executed time of function"""

    def laptime_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kargs) -> Any:
            with LapTime(func.__name__, time_func=timer):
                result = func(*args, **kargs)
            return result

        return wrapper

    return laptime_wrapper


class LapTime:
    def __init__(
        self,
        message: str = "Elapsed time",
        time_func: Callable = time.perf_counter,
        disable: bool = False,
    ) -> None:
        """

        Args:
            message (str, optional): log message. Defaults to "Elapsed time".
            time_func (Callable, optional): meas time function. Defaults to time.perf_counter.
            disable (bool, optional): whether to disable log on console. Defaults to Dlaw.

        """
        self.timer: Callable = time_func
        self.message = message
        self.disable = disable
        self.logger: Logger = logging.getLogger()

    def __enter__(self) -> None:
        self.start = self.timer()

    def __exit__(self, type, value, traceback) -> None:
        end = self.timer()
        msg = f"{self.message}: {(end - self.start):.6f} [sec]"
        if self.disable:
            return
        self.logger.info(msg, stacklevel=2)
