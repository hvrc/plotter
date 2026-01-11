"""algorithms.base - shared interface for plot algorithms.

Algorithms implement a common interface so the CLI can:
- generate paths (from an image or procedurally)
- generate SVG output
- generate a preview image
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image


Paths = List[List[Tuple[float, float]]]


class PlotAlgorithm(ABC):
    """Base class for all plot generation algorithms."""

    DEFAULT_CONFIG: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)

    def set_config(self, config: Dict[str, Any]):
        self.config.update(config)

    @abstractmethod
    def get_algorithm_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_algorithm_description(self) -> str:
        raise NotImplementedError

    def is_procedural(self) -> bool:
        """Whether this algorithm can run without an input image."""
        return False

    @abstractmethod
    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Return (paths, metadata). Paths may be multi-layer (e.g. dict per color)."""
        raise NotImplementedError

    @abstractmethod
    def generate_svg(self, paths: Any, metadata: Dict[str, Any]) -> Any:
        """Return SVG string (or multi-layer structure)."""
        raise NotImplementedError

    @abstractmethod
    def generate_preview(self, paths: Any, metadata: Dict[str, Any]) -> Image.Image:
        """Return a PIL preview image."""
        raise NotImplementedError
