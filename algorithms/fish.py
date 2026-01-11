"""
fish.py - Single circle plot generation

Generates a single centered circle (procedural).
"""

import math
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw

from .base import PlotAlgorithm


class FishGenerator(PlotAlgorithm):
    """Generate a single centered circle on the canvas (procedural, no image needed)."""

    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.125,
        'calc_dpi': 300,
        'radius_fraction': 0.9,  # relative to half of drawable min dimension
        'points_per_circle': 720,
        'stroke_width': 0.5
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]):
        self.config.update(config)

    def get_algorithm_name(self) -> str:
        return "fish"

    def get_algorithm_description(self) -> str:
        return "Single centered circle (procedural)"

    def is_procedural(self) -> bool:
        return True

    def generate_paths(self, image_path: str = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        calc_dpi = self.config['calc_dpi']
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        radius_fraction = float(self.config.get('radius_fraction', 0.9))
        points_per_circle = int(self.config.get('points_per_circle', 720))

        total_width_px = int(output_width_inches * calc_dpi)
        total_height_px = int(output_height_inches * calc_dpi)
        margin_px = int(margin_inches * calc_dpi)

        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        if drawable_width_px <= 0 or drawable_height_px <= 0:
            raise ValueError("Margins too large for output size")

        radius_fraction = max(0.0, min(1.0, radius_fraction))

        center_x = drawable_width_px / 2 + margin_px
        center_y = drawable_height_px / 2 + margin_px
        radius = (min(drawable_width_px, drawable_height_px) / 2) * radius_fraction

        path: List[Tuple[float, float]] = []
        for i in range(points_per_circle + 1):
            angle = (i / points_per_circle) * 2 * math.pi
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            path.append((x, y))

        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': output_width_inches,
            'output_height_inches': output_height_inches,
            'algorithm': self.get_algorithm_name(),
            'stroke_width': float(self.config.get('stroke_width', 0.5))
        }

        return [path], metadata

    def generate_svg(self, paths: List[List[Tuple[float, float]]], metadata: Dict[str, Any]) -> str:
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        output_width_inches = metadata['output_width_inches']
        output_height_inches = metadata['output_height_inches']
        stroke_width = float(metadata.get('stroke_width', 0.5))

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{output_width_inches}in" '
            f'height="{output_height_inches}in" '
            f'viewBox="0 0 {total_width_px} {total_height_px}">\n'
        ]

        for path in paths:
            if len(path) < 2:
                continue
            d_str = f"M {path[0][0]:.2f},{path[0][1]:.2f}"
            for x, y in path[1:]:
                d_str += f" L {x:.2f},{y:.2f}"
            svg_lines.append(
                f'<path d="{d_str}" stroke="black" fill="none" stroke-width="{stroke_width}"/>\n'
            )

        svg_lines.append('</svg>')
        return ''.join(svg_lines)

    def generate_preview(self, paths: List[List[Tuple[float, float]]], metadata: Dict[str, Any]) -> Image.Image:
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']

        preview = Image.new('RGB', (total_width_px, total_height_px), 'white')
        draw = ImageDraw.Draw(preview)
        for path in paths:
            if len(path) > 1:
                draw.line(path, fill='black', width=1)
        return preview
