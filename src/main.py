from utilpy import LapTime, lap_time


def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def main():
    with LapTime("hello"):
        fibonacci(20)


if __name__ == "__main__":
    main()
