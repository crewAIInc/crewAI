import importlib
from typing import Any

# Global dictionary acting as the cache collection
_MODULE_CACHE: dict[str, Any] = {}

def lazy_import_k8s_agent_sandbox(target: str):
    """
    Lazily imports a module or attribute by string name, caches it globally,
    and returns it. Subsequent calls return the object directly from the cache.

    :param target: Module name (e.g., 'json') or object path (e.g., 'math.sqrt')
    :return: The imported module or attribute object
    """

    full_target = f"k8s_agent_sandbox.{target}"

    module = _MODULE_CACHE.get(full_target)
    if module is not None:
        return module

    try:
        obj = importlib.import_module(full_target)
    except ModuleNotFoundError as e:
        raise ImportError(
            "The 'k8s-agent-sandbox' package is required for K8s Agent Sandbox tools. "
        ) from e

    # 3. Store in global cache collection
    _MODULE_CACHE[full_target] = obj

    # 4. Inject into the module's global space (using short name)
    short_name = full_target.split(".")[-1]
    globals()[short_name] = obj

    return obj
