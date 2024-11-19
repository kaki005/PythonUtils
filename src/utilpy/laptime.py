import time
from functools import wraps
from re import A
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
        self, message: str = "Elapsed time", write_log: bool = True, time_func: Callable = time.process_time
    ) -> None:
        """

        Args:
            message (str, optional): log message. Defaults to "Elapsed time".
            write_log (bool, optional): whether to output log on console. Defaults to True.
            time_func (Callable, optional): _description_. Defaults to time.process_time.
        """
        self.timer = time_func
        self.message = message
        self.write_log = write_log

    def __enter__(self) -> None:
        self.start = self.timer()

    def __exit__(self, type, value, traceback) -> None:
        end = self.timer()
        if not self.write_log:
            return
        print(f"{self.message}: {(end - self.start):.6f} [sec]")
