from functools import wraps


def memoize(func):
    cache = {}

    @wraps(func)
    def memoized_func(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func
