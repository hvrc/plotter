"""
features.py - Random features plot generation

Generates random circles of fixed radius distributed across canvas.
"""

import math
import random
from typing import Any, Dict, List, Tuple, Optional

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
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with optional configuration override."""
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Update configuration."""
        self.config.update(config)
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "features"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Random circles of fixed radius distributed across the canvas"

    def is_procedural(self) -> bool:
        return True
    
    def generate_paths(self, image_path: str = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate random circle paths.
        
        Args:
            image_path: Not used for this procedural algorithm
            
        Returns:
            Tuple of (paths, metadata)
        """
        # Extract configuration
        calc_dpi = self.config['calc_dpi']
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        feature_type = self.config.get('feature_type', 'circles')
        num_features = self.config.get('num_features', 100)
        circle_radius = self.config.get('circle_radius', 0.2)
        points_per_circle = self.config.get('points_per_circle', 360)
        
        # Calculate drawable area
        drawable_width = output_width_inches - 2 * margin_inches
        drawable_height = output_height_inches - 2 * margin_inches
        
        # Generate paths based on feature type
        paths = []
        
        if feature_type == 'circles':
            paths = self._generate_random_circles(
                drawable_width, drawable_height, margin_inches,
                num_features, circle_radius, points_per_circle
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
    
    def _generate_random_circles(self, drawable_width: float, drawable_height: float,
                                  margin: float, num_circles: int, radius: float,
                                  points: int) -> List[List[Tuple[float, float]]]:
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
        paths = []
        
        for _ in range(num_circles):
            # Random center position within drawable area, accounting for radius
            # to keep circles within bounds
            cx = margin + radius + random.random() * (drawable_width - 2 * radius)
            cy = margin + radius + random.random() * (drawable_height - 2 * radius)
            
            # Generate circle path
            circle_path = []
            for i in range(points + 1):  # +1 to close the circle
                angle = (i / points) * 2 * math.pi
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                circle_path.append((x, y))
            
            paths.append(circle_path)
        
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
        width_inches = metadata['width_inches']
        height_inches = metadata['height_inches']
        stroke_width = self.config.get('stroke_width', 0.5)
        
        # Convert inches to pixels for viewBox (using 96 DPI standard for SVG)
        svg_dpi = 96
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
            
            # Convert from inches to pixels
            pixel_path = [(x * svg_dpi, y * svg_dpi) for x, y in path]
            
            # Build path data
            d_str = f"M {pixel_path[0][0]:.2f},{pixel_path[0][1]:.2f}"
            for x, y in pixel_path[1:]:
                d_str += f" L {x:.2f},{y:.2f}"
            
            svg_lines.append(
                f'<path d="{d_str}" stroke="black" fill="none" stroke-width="{stroke_width}"/>\n'
            )
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def generate_preview(self, paths, metadata):
        """
        Generate a preview image of the plot.
        
        Args:
            paths: List of path segments
            metadata: Metadata dict with dimensions
            
        Returns:
            PIL Image
        """
        from PIL import Image, ImageDraw
        
        # Get dimensions
        width_inches = metadata['width_inches']
        height_inches = metadata['height_inches']
        
        # Create image at 150 DPI for preview
        preview_dpi = 150
        width_px = int(width_inches * preview_dpi)
        height_px = int(height_inches * preview_dpi)
        
        # Create white background
        preview = Image.new('RGB', (width_px, height_px), 'white')
        draw = ImageDraw.Draw(preview)
        
        # Draw all paths
        for path in paths:
            if len(path) < 2:
                continue
            
            # Convert from inches to pixels
            pixel_path = [
                (x * preview_dpi, y * preview_dpi)
                for x, y in path
            ]
            
            draw.line(pixel_path, fill='black', width=1)
        
        return preview
