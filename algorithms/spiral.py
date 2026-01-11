"""
spiral.py - Spiral attractor plot generation

Generates flowing spirals around random attractor points.
"""

import math
import random
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw

from .base import PlotAlgorithm

class SpiralAttractorGenerator(PlotAlgorithm):
    """
    Generates flowing spiral patterns around multiple random attractor points.
    
    Creates organic, flowing curves that spiral around randomly placed attractor
    points on the canvas, similar to magnetic field lines or flowing water.
    """
    
    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.125,
        'num_attractors': 5,           # Number of attractor points
        'num_particles': 80,            # Number of starting particles
        'max_steps': 300,               # Maximum steps per particle path
        'step_size': 3.0,               # Size of each step in pixels
        'rotation_speed': 0.15,         # How fast spirals rotate around attractors
        'attraction_strength': 0.5,     # How strongly attractors pull particles
        'noise_influence': 0.3,         # Amount of random noise/variation
        'spiral_tightness': 0.02,       # How tight the spirals are (higher = tighter)
        'stroke_width': 0.5,            # SVG stroke width
        'calc_dpi': 300,                # DPI for calculations
        'color_gradient': False,        # Whether to use color gradients (red to orange)
        'min_distance': 1.0,            # Minimum distance to move in a step
        'connect_paths': True,          # Whether to connect end of one path to start of next
        'min_attractor_distance': 100,  # Minimum distance between attractor centers (in pixels)
        'min_path_points': 5,           # Minimum number of points for a path to be included
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
        return "spiral"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Flowing spirals around random attractor points"

    def is_procedural(self) -> bool:
        return True
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _angle_to(self, from_point: Tuple[float, float], to_point: Tuple[float, float]) -> float:
        """Calculate angle from one point to another."""
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        return math.atan2(dy, dx)
    
    def _find_nearest_attractor(self, point: Tuple[float, float], 
                                attractors: List[Tuple[float, float]]) -> Tuple[int, float]:
        """Find the nearest attractor to a point."""
        min_dist = float('inf')
        nearest_idx = 0
        for i, attractor in enumerate(attractors):
            dist = self._distance(point, attractor)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist
    
    def _calculate_force(self, point: Tuple[float, float], 
                        attractors: List[Tuple[float, float]],
                        current_angle: float) -> Tuple[float, float]:
        """
        Calculate the direction of movement for a particle.
        
        Combines attraction to nearest attractor with spiral rotation.
        """
        nearest_idx, dist = self._find_nearest_attractor(point, attractors)
        nearest = attractors[nearest_idx]
        
        # Angle towards attractor
        attraction_angle = self._angle_to(point, nearest)
        
        # Add spiral component (perpendicular to attraction + slight inward)
        spiral_angle = attraction_angle + math.pi / 2  # 90 degrees perpendicular
        
        # Blend between straight attraction and spiral
        # As we get closer to attractor, increase spiral component
        max_spiral_dist = 400  # Distance at which spiral is minimal
        spiral_factor = 1.0 - min(dist / max_spiral_dist, 1.0)
        
        # Combine angles with weights
        attraction_weight = self.config['attraction_strength']
        spiral_weight = self.config['rotation_speed'] * spiral_factor
        
        # Add noise for organic variation
        noise_angle = random.uniform(-1, 1) * self.config['noise_influence'] * math.pi
        
        # Calculate weighted direction
        dx = (math.cos(attraction_angle) * attraction_weight + 
              math.cos(spiral_angle) * spiral_weight +
              math.cos(noise_angle) * 0.1)
        dy = (math.sin(attraction_angle) * attraction_weight + 
              math.sin(spiral_angle) * spiral_weight +
              math.sin(noise_angle) * 0.1)
        
        # Normalize
        magnitude = math.sqrt(dx*dx + dy*dy)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        return dx, dy
    
    def _generate_particle_path(self, start_x: float, start_y: float,
                                attractors: List[Tuple[float, float]],
                                width: float, height: float) -> List[Tuple[float, float]]:
        """Generate a single flowing particle path."""
        path = [(start_x, start_y)]
        x, y = start_x, start_y
        current_angle = random.uniform(0, 2 * math.pi)
        
        max_steps = self.config['max_steps']
        step_size = self.config['step_size']
        min_distance = self.config['min_distance']
        
        for step in range(max_steps):
            # Calculate movement direction
            dx, dy = self._calculate_force((x, y), attractors, current_angle)
            
            # Move particle
            x += dx * step_size
            y += dy * step_size
            
            # Check bounds
            if x < 0 or x > width or y < 0 or y > height:
                break
            
            # Check if we're too close to the nearest attractor (termination condition)
            _, dist = self._find_nearest_attractor((x, y), attractors)
            if dist < 10:  # Stop if very close to an attractor
                break
            
            # Add point to path if it moved enough
            if len(path) == 1 or self._distance(path[-1], (x, y)) >= min_distance:
                path.append((x, y))
        
        return path
    
    def generate_paths(self, image_path: str = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate spiral attractor paths.
        
        Args:
            image_path: Not used for this procedural algorithm (optional)
            
        Returns:
            Tuple of (paths, metadata)
        """
        # Calculate dimensions
        calc_dpi = self.config['calc_dpi']
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        
        total_width_px = int(output_width_inches * calc_dpi)
        total_height_px = int(output_height_inches * calc_dpi)
        margin_px = int(margin_inches * calc_dpi)
        
        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        
        # Generate random attractor points
        num_attractors = self.config['num_attractors']
        min_attractor_dist = self.config.get('min_attractor_distance', 100)
        attractors = []
        max_attempts = 1000  # Prevent infinite loop
        
        for _ in range(num_attractors):
            attempts = 0
            while attempts < max_attempts:
                x = margin_px + random.uniform(0.2, 0.8) * drawable_width_px
                y = margin_px + random.uniform(0.2, 0.8) * drawable_height_px
                
                # Check if this point is far enough from existing attractors
                valid = True
                for existing in attractors:
                    if self._distance((x, y), existing) < min_attractor_dist:
                        valid = False
                        break
                
                if valid or len(attractors) == 0:
                    attractors.append((x, y))
                    break
                
                attempts += 1
            
            # If we couldn't find a valid spot, just add it anyway
            if attempts >= max_attempts and len(attractors) < num_attractors:
                x = margin_px + random.uniform(0.2, 0.8) * drawable_width_px
                y = margin_px + random.uniform(0.2, 0.8) * drawable_height_px
                attractors.append((x, y))
        
        # Generate particle starting positions (random around canvas)
        num_particles = self.config['num_particles']
        all_paths = []
        
        for _ in range(num_particles):
            # Start particles from random positions
            start_x = margin_px + random.uniform(0, 1) * drawable_width_px
            start_y = margin_px + random.uniform(0, 1) * drawable_height_px
            
            path = self._generate_particle_path(
                start_x, start_y, attractors,
                total_width_px - margin_px,
                total_height_px - margin_px
            )
            
            min_path_points = self.config.get('min_path_points', 5)
            if len(path) > min_path_points:  # Only add paths with enough points
                all_paths.append(path)
        
        # Connect all paths into one continuous path if connect_paths is enabled
        if self.config.get('connect_paths', True) and len(all_paths) > 1:
            connected_path = []
            for path in all_paths:
                connected_path.extend(path)
            all_paths = [connected_path]
        
        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': output_width_inches,
            'output_height_inches': output_height_inches,
            'algorithm': self.get_algorithm_name(),
            'stroke_width': float(self.config.get('stroke_width', 0.5)),
            'color_gradient': self.config.get('color_gradient', True),
            'num_paths': len(all_paths)
        }
        
        return all_paths, metadata
    
    def generate_svg(self, paths: List[List[Tuple[float, float]]], metadata: Dict[str, Any]) -> str:
        """Generate SVG from paths."""
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        output_width_inches = metadata['output_width_inches']
        output_height_inches = metadata['output_height_inches']
        stroke_width = float(metadata.get('stroke_width', 0.5))
        use_gradient = metadata.get('color_gradient', True)
        
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{output_width_inches}in" '
            f'height="{output_height_inches}in" '
            f'viewBox="0 0 {total_width_px} {total_height_px}">\n'
        ]
        
        # Generate paths with optional color gradient
        num_paths = len(paths)
        for i, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            # Calculate color (gradient from red to orange)
            if use_gradient and num_paths > 1:
                # Red (180, 0, 0) to Orange (255, 140, 0)
                t = i / (num_paths - 1)
                r = int(180 + (255 - 180) * t)
                g = int(0 + (140 - 0) * t)
                b = 0
                color = f"rgb({r},{g},{b})"
            else:
                color = "black"
            
            d_str = f"M {path[0][0]:.2f},{path[0][1]:.2f}"
            for x, y in path[1:]:
                d_str += f" L {x:.2f},{y:.2f}"
            svg_lines.append(
                f'<path d="{d_str}" stroke="{color}" fill="none" stroke-width="{stroke_width}"/>\n'
            )
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def generate_preview(self, paths: List[List[Tuple[float, float]]], metadata: Dict[str, Any]) -> Image.Image:
        """Generate a preview image."""
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        use_gradient = metadata.get('color_gradient', True)
        
        preview = Image.new('RGB', (total_width_px, total_height_px), 'white')
        draw = ImageDraw.Draw(preview)
        
        num_paths = len(paths)
        for i, path in enumerate(paths):
            if len(path) > 1:
                # Calculate color
                if use_gradient and num_paths > 1:
                    t = i / (num_paths - 1)
                    r = int(180 + (255 - 180) * t)
                    g = int(0 + (140 - 0) * t)
                    b = 0
                    color = (r, g, b)
                else:
                    color = 'black'
                
                draw.line(path, fill=color, width=2)
        
        return preview
