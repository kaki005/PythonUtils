import time
from functools import wraps
from operator import call
from re import A
from typing import Any, Callable


def lap_time(timer: Callable = time.process_time) -> Callable:
    """wrapper for measuring executed time of function"""

    def laptime_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kargs) -> Any:
            with LapTime(func.__name__, timer):
                result = func(*args, **kargs)
            return result

        return wrapper

    return laptime_wrapper


class LapTime:
    def __init__(self, message: str = "", time_func: Callable = time.process_time) -> None:
        self.timer = time_func
        self.message = message

    def __enter__(self) -> None:
        self.start = self.timer()

    def __exit__(self, type, value, traceback) -> None:
        end = self.timer()
        if self.message != "":
            print(f"{self.message}:  Elapsed time: {(end - self.start):.6f} [sec]")
        else:
            print(f"Elapsed time: {(end - self.start):.6f} [sec]")
