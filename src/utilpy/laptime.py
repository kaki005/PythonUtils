import time
from functools import wraps
from logging import Logger
from typing import Any, Callable


def lap_time(timer: Callable = time.process_time) -> Callable:
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
        write_log: bool = True,
        time_func: Callable = time.process_time,
        logger: Logger | None = None,
    ) -> None:
        """

        Args:
            message (str, optional): log message. Defaults to "Elapsed time".
            write_log (bool, optional): whether to output log on console. Defaults to True.
            time_func (Callable, optional): meas time function. Defaults to time.process_time.
            logger: logging class
        """
        self.timer: Callable = time_func
        self.message = message
        self.write_log = write_log
        self.logger: Logger | None = logger

    def __enter__(self) -> None:
        self.start = self.timer()

    def __exit__(self, type, value, traceback) -> None:
        end = self.timer()
        msg = f"{self.message}: {(end - self.start):.6f} [sec]"
        if not self.write_log:
            return
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
