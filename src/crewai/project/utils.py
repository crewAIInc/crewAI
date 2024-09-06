def memoize(func):
    cache = {}

    def memoized_func(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    memoized_func.__dict__.update(func.__dict__)
    return memoized_func
