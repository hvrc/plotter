"""sphere.py - 3D sphere plot generation.

A lightweight procedural sphere visualizer.
"""

import math
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw

from .base import PlotAlgorithm


class SphereGenerator(PlotAlgorithm):
    """Procedurally generate a sphere using latitude/longitude curves."""

    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.125,
        'calc_dpi': 300,
        'radius_fraction': 0.9,
        'num_latitudes': 36,
        'num_longitudes': 18,
        'stroke_width': 0.5,
    }

    def get_algorithm_name(self) -> str:
        return 'sphere'

    def get_algorithm_description(self) -> str:
        return '3D sphere with latitude/longitude flow lines (procedural)'

    def is_procedural(self) -> bool:
        return True

    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        calc_dpi = int(self.config['calc_dpi'])
        output_width_inches = float(self.config['output_width_inches'])
        output_height_inches = float(self.config['output_height_inches'])
        margin_inches = float(self.config['margin_inches'])
        radius_fraction = float(self.config.get('radius_fraction', 0.9))
        num_latitudes = int(self.config.get('num_latitudes', 36))
        num_longitudes = int(self.config.get('num_longitudes', 18))

        total_width_px = int(output_width_inches * calc_dpi)
        total_height_px = int(output_height_inches * calc_dpi)
        margin_px = int(margin_inches * calc_dpi)

        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        if drawable_width_px <= 0 or drawable_height_px <= 0:
            raise ValueError('Margins too large for output size')

        radius_fraction = max(0.0, min(1.0, radius_fraction))
        center_x = drawable_width_px / 2 + margin_px
        center_y = drawable_height_px / 2 + margin_px
        radius = (min(drawable_width_px, drawable_height_px) / 2) * radius_fraction

        paths: List[List[Tuple[float, float]]] = []

        # Latitudes: horizontal ellipses, compressed near top/bottom.
        num_latitudes = max(2, num_latitudes)
        for i in range(1, num_latitudes):
            t = (i / num_latitudes) * math.pi  # 0..pi
            y_off = math.cos(t) * radius
            r_lat = math.sin(t) * radius
            if r_lat <= 1e-6:
                continue

            # Foreshortening: make latitudes appear more 3D by scaling x radius.
            x_scale = 0.85
            x_r = r_lat * x_scale

            pts: List[Tuple[float, float]] = []
            steps = 360
            for s in range(steps + 1):
                a = (s / steps) * 2 * math.pi
                x = center_x + x_r * math.cos(a)
                y = center_y + y_off + (r_lat * 0.35) * math.sin(a)
                pts.append((x, y))
            paths.append(pts)

        # Longitudes: vertical ellipses.
        num_longitudes = max(4, num_longitudes)
        for i in range(num_longitudes):
            phi = (i / num_longitudes) * math.pi
            pts: List[Tuple[float, float]] = []
            steps = 360
            for s in range(steps + 1):
                t = (s / steps) * 2 * math.pi
                x = center_x + radius * math.sin(t) * math.cos(phi) * 0.85
                y = center_y + radius * math.cos(t)
                pts.append((x, y))
            paths.append(pts)

        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': output_width_inches,
            'output_height_inches': output_height_inches,
            'algorithm': self.get_algorithm_name(),
            'stroke_width': float(self.config.get('stroke_width', 0.5)),
        }
        return paths, metadata

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
            d = f'M {path[0][0]:.2f},{path[0][1]:.2f}'
            for x, y in path[1:]:
                d += f' L {x:.2f},{y:.2f}'
            svg_lines.append(
                f'<path d="{d}" stroke="black" fill="none" stroke-width="{stroke_width}"/>\n'
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
