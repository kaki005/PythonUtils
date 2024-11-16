from typing import Callable
import time
from functools import wraps



def lap_time(timer:Callable = time.process_time) :
    """wrapper for measuring executed time of function"""
    def laptime_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kargs) :
            with LapTime(func.__name__, timer):
                result = func(*args,**kargs)
            return result
        return wrapper
    return laptime_wrapper

class LapTime:
    def __init__(self, message:str = "", time_func:Callable = time.process_time):
        self.timer = time_func
        self.message =message

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, type, value, traceback):
            end = self.timer()
            if self.message != "":
                print(f"{self.message}:  Elapsed time: {(end - self.start):.6f} [sec]")
            else:
                print(f"Elapsed time: {(end - self.start):.6f} [sec]")
