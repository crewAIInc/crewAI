import weakref
from functools import wraps


def memoize(func):
    """
    Memoization decorator that uses weak references to prevent memory leaks.

    This implementation uses weak references for the first argument (typically 'self')
    to allow proper garbage collection of instances while still providing caching
    benefits for method calls.
    """
    # Use WeakKeyDictionary for instance-based caching
    instance_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    @wraps(func)
    def memoized_func(*args, **kwargs):
        if not args:
            # No arguments, use simple caching
            key = tuple(kwargs.items())
            if not hasattr(memoized_func, "_simple_cache"):
                memoized_func._simple_cache = {}
            cache = memoized_func._simple_cache
        else:
            # First argument exists, check if it's an instance
            first_arg = args[0]

            # Try to use the first argument as a weak reference key
            try:
                # Get or create cache for this instance
                if first_arg not in instance_caches:
                    instance_caches[first_arg] = {}
                cache = instance_caches[first_arg]

                # Create key from remaining args and kwargs
                key = (args[1:], tuple(kwargs.items()))

            except TypeError:
                # First argument is not hashable or cannot be weakly referenced
                # Fall back to regular caching (for primitive types, etc.)
                if not hasattr(memoized_func, "_fallback_cache"):
                    memoized_func._fallback_cache = {}
                cache = memoized_func._fallback_cache
                key = (args, tuple(kwargs.items()))

        # Check cache and return or compute result
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    # Add method to clear cache for debugging/testing
    def clear_cache():
        instance_caches.clear()
        if hasattr(memoized_func, "_simple_cache"):
            memoized_func._simple_cache.clear()
        if hasattr(memoized_func, "_fallback_cache"):
            memoized_func._fallback_cache.clear()

    memoized_func.clear_cache = clear_cache
    return memoized_func
