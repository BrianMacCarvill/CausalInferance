from numpy.typing import NDArray

def _get_from_registry(name_or_callable, registry: dict[str, Callable], kind: str, **kwargs):
    """Helper to resolve a callable or string key from a registry."""
    if callable(name_or_callable):
        return name_or_callable
    
    return registry[name_or_callable](**kwargs) if kind == "model" else registry[name_or_callable]

def _unit_range(x: NDArray, min=0, max=1):
    """
    get unit range for numpy function
    """
    
    return (x - x.min()) / (x.max() - x.min())
