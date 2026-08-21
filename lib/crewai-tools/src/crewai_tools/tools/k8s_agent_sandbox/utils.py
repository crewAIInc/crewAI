import importlib
import types

# Global dictionary acting as the cache collection
_MODULE_CACHE: dict[str, types.ModuleType] = {}


def lazy_import_k8s_agent_sandbox(target: str) -> types.ModuleType:
    """
    Lazily imports a `k8s_agent_sandbox` submodule by name, caches it globally,
    and returns it. Subsequent calls return the module directly from the cache.

    :param target: Submodule name (e.g., 'sandbox_client')
    :return: The imported module object
    """

    full_target = f"k8s_agent_sandbox.{target}"

    module = _MODULE_CACHE.get(full_target)
    if module is not None:
        return module

    try:
        obj = importlib.import_module(full_target)
    except ModuleNotFoundError as e:
        # An installed SDK can raise this too when one of its own dependencies
        # is missing, and reporting that as an absent SDK sends people down the
        # wrong path.
        if e.name != "k8s_agent_sandbox":
            raise
        raise ImportError(
            "The 'k8s-agent-sandbox' package is required for K8s Agent Sandbox tools. "
        ) from e

    _MODULE_CACHE[full_target] = obj

    return obj
