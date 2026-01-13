import inspect
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import PlotAlgorithm

# Algorithm registry - populated by auto-discovery
_ALGORITHM_REGISTRY: Dict[str, Type[PlotAlgorithm]] = {}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _register_algorithm_class(cls: Type[PlotAlgorithm]) -> None:
    """Instantiate a class to determine its registry name, then register it."""
    instance = cls()
    algo_name = _normalize_name(instance.get_algorithm_name())
    if algo_name:
        _ALGORITHM_REGISTRY[algo_name] = cls

def discover_algorithms():
    """
    Automatically discover all algorithm classes in this package.
    
    Scans all .py files in the algorithms directory and finds classes
    that inherit from PlotAlgorithm.
    """
    global _ALGORITHM_REGISTRY
    
    if _ALGORITHM_REGISTRY:
        # Already discovered
        return
    
    # Get the algorithms directory path
    algorithms_dir = Path(__file__).resolve().parent
    
    # Scan all Python files except __init__.py and base.py
    for file_path in algorithms_dir.glob("*.py"):
        if file_path.stem in ('__init__', 'base'):
            continue
        
        module_name = f"algorithms.{file_path.stem}"
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj is PlotAlgorithm:
                    continue
                # Check if it's a subclass of PlotAlgorithm
                if issubclass(obj, PlotAlgorithm):
                    try:
                        _register_algorithm_class(obj)
                    except Exception as e:
                        print(f"Warning: Could not instantiate {name} from {module_name}: {e}")
        
        except Exception as e:
            print(f"Warning: Could not import {module_name}: {e}")


def get_algorithm(algorithm_name: str, config: Optional[Dict[str, Any]] = None) -> PlotAlgorithm:
    """
    Get an algorithm instance by name.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., 'waves', 'circles')
        config: Optional configuration dictionary
        
    Returns:
        Algorithm instance
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    # Ensure algorithms are discovered
    discover_algorithms()
    
    algorithm_key = _normalize_name(algorithm_name)
    if algorithm_key not in _ALGORITHM_REGISTRY:
        available = ', '.join(_ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm: '{algorithm_name}'. "
            f"Available algorithms: {available}"
        )
    
    algorithm_class = _ALGORITHM_REGISTRY[algorithm_key]
    return algorithm_class(config)


def list_algorithms() -> List[Dict[str, Any]]:
    """
    List all available algorithms.
    
    Returns:
        List of dicts with 'name' and 'description' keys
    """
    # Ensure algorithms are discovered
    discover_algorithms()
    
    algorithms: List[Dict[str, Any]] = []
    for name, cls in sorted(_ALGORITHM_REGISTRY.items(), key=lambda kv: kv[0]):
        try:
            instance = cls()
            algorithms.append({
                'name': name,
                'description': instance.get_algorithm_description(),
                'procedural': instance.is_procedural()
            })
        except Exception:
            algorithms.append({
                'name': name,
                'description': 'Description not available',
                'procedural': False
            })
    
    return algorithms


def get_algorithm_names() -> List[str]:
    """
    Get list of all available algorithm names.
    
    Returns:
        List of algorithm names
    """
    discover_algorithms()
    return sorted(_ALGORITHM_REGISTRY.keys())


# Auto-discover algorithms on import
discover_algorithms()


# Export main interface
__all__ = [
    'PlotAlgorithm',
    'get_algorithm',
    'list_algorithms',
    'get_algorithm_names',
    'discover_algorithms'
]
