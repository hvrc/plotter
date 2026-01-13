"""algorithms.features

Random features plot generation.

Currently supports procedural circles distributed across the canvas.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .base import PlotAlgorithm

class FeaturesGenerator(PlotAlgorithm):
    """
    Generates random circles of fixed radius across the canvas.
    
    This is a procedural algorithm that creates patterns with random
    circle placements. Sub-settings allow different feature types.
    """
    
    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.5,
        'calc_dpi': 300,
        'feature_type': 'circles',  # Sub-setting: 'circles'
        'num_features': 100,  # Number of circles to draw
        'circle_radius': 0.2,  # Fixed radius in inches
        'points_per_circle': 360,  # Resolution of each circle
        'stroke_width': 0.5
    }
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "features"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Random circles of fixed radius distributed across the canvas"

    def is_procedural(self) -> bool:
        return True

    @staticmethod
    def _circle_path(cx: float, cy: float, radius: float, points: int) -> List[Tuple[float, float]]:
        """Return a closed circle polyline in inches."""
        circle: List[Tuple[float, float]] = []
        for i in range(points + 1):
            angle = (i / points) * 2 * math.pi
            circle.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        return circle

    @staticmethod
    def _to_px_path(path: List[Tuple[float, float]], dpi: float) -> List[Tuple[float, float]]:
        return [(x * dpi, y * dpi) for x, y in path]
    
    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate random circle paths.
        
        Args:
            image_path: Not used for this procedural algorithm
            
        Returns:
            Tuple of (paths, metadata)
        """
        config = self.config
        output_width_inches = float(config['output_width_inches'])
        output_height_inches = float(config['output_height_inches'])
        margin_inches = float(config['margin_inches'])

        feature_type = str(config.get('feature_type', 'circles') or 'circles').lower()
        num_features = int(config.get('num_features', 100) or 0)
        circle_radius = float(config.get('circle_radius', 0.2) or 0.0)
        points_per_circle = int(config.get('points_per_circle', 360) or 0)

        drawable_width = output_width_inches - 2 * margin_inches
        drawable_height = output_height_inches - 2 * margin_inches
        if drawable_width <= 0 or drawable_height <= 0:
            raise ValueError("Margins too large for output size")

        paths: List[List[Tuple[float, float]]] = []
        if feature_type == 'circles' and num_features > 0 and points_per_circle >= 3:
            paths = self._generate_random_circles(
                drawable_width=drawable_width,
                drawable_height=drawable_height,
                margin=margin_inches,
                num_circles=num_features,
                radius=circle_radius,
                points=points_per_circle,
            )
        
        # Create metadata for SVG generation
        metadata = {
            'width_inches': output_width_inches,
            'height_inches': output_height_inches,
            'margin_inches': margin_inches,
            'algorithm': 'features',
            'feature_type': feature_type,
            'num_features': num_features
        }
        
        return paths, metadata
    
    def _generate_random_circles(
        self,
        *,
        drawable_width: float,
        drawable_height: float,
        margin: float,
        num_circles: int,
        radius: float,
        points: int,
    ) -> List[List[Tuple[float, float]]]:
        """
        Generate random circles with fixed radius.
        
        Args:
            drawable_width: Width of drawable area in inches
            drawable_height: Height of drawable area in inches
            margin: Margin in inches
            num_circles: Number of circles to generate
            radius: Fixed radius of circles in inches
            points: Number of points per circle
            
        Returns:
            List of circle paths
        """
        paths: List[List[Tuple[float, float]]] = []

        max_x = drawable_width - 2 * radius
        max_y = drawable_height - 2 * radius
        if max_x <= 0 or max_y <= 0:
            return paths

        for _ in range(num_circles):
            cx = margin + radius + random.random() * max_x
            cy = margin + radius + random.random() * max_y
            paths.append(self._circle_path(cx, cy, radius, points))

        return paths
    
    def generate_svg(self, paths: List[List[Tuple[float, float]]], metadata: Dict[str, Any]) -> str:
        """
        Generate SVG content from paths.
        
        Args:
            paths: List of path segments
            metadata: Metadata dict with dimensions
            
        Returns:
            SVG string
        """
        width_inches = float(metadata['width_inches'])
        height_inches = float(metadata['height_inches'])
        stroke_width = self.config.get('stroke_width', 0.5)

        # Convert inches to pixels for viewBox (using 96 DPI standard for SVG)
        svg_dpi = 96.0
        width_px = width_inches * svg_dpi
        height_px = height_inches * svg_dpi
        
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width_inches}in" '
            f'height="{height_inches}in" '
            f'viewBox="0 0 {width_px} {height_px}">\n'
        ]
        
        # Draw all paths
        for path in paths:
            if len(path) < 2:
                continue

            pixel_path = self._to_px_path(path, svg_dpi)
            d_str = f"M {pixel_path[0][0]:.2f},{pixel_path[0][1]:.2f}" + ''.join(
                f" L {x:.2f},{y:.2f}" for x, y in pixel_path[1:]
            )
            
            svg_lines.append(
                f'<path d="{d_str}" stroke="black" fill="none" stroke-width="{stroke_width}"/>\n'
            )
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def generate_preview(self, paths: List[List[Tuple[float, float]]], metadata: Dict[str, Any]) -> Image.Image:
        """
        Generate a preview image of the plot.
        
        Args:
            paths: List of path segments
            metadata: Metadata dict with dimensions
            
        Returns:
            PIL Image
        """
        width_inches = float(metadata['width_inches'])
        height_inches = float(metadata['height_inches'])

        # Create image at 150 DPI for preview
        preview_dpi = 150.0
        width_px = int(width_inches * preview_dpi)
        height_px = int(height_inches * preview_dpi)
        
        # Create white background
        preview = Image.new('RGB', (width_px, height_px), 'white')
        draw = ImageDraw.Draw(preview)
        
        # Draw all paths
        for path in paths:
            if len(path) < 2:
                continue

            pixel_path = self._to_px_path(path, preview_dpi)
            
            draw.line(pixel_path, fill='black', width=1)
        
        return preview
