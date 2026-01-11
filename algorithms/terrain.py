"""
terrain.py - 3D terrain plot generation

Generates 3D topographic terrain visualizations with wireframe lines.
"""

import math
import random
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw

from .base import PlotAlgorithm


class TerrainGenerator(PlotAlgorithm):
    """
    Generates 3D topographic terrain visualizations.
    
    Creates a 3D wireframe terrain map using Perlin noise or image-based
    elevation data. The output resembles a cut-out cube showing terrain contours.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'output_width_inches': 6.0,
        'output_height_inches': 6.0,
        'margin_inches': 0.5,
        'calc_dpi': 300,
        'grid_resolution_x': 60,  # Number of lines along X axis
        'grid_resolution_z': 60,  # Number of lines along Z axis (depth)
        'cube_height': 0.3,  # Height of the cube base (in normalized units)
        'terrain_amplitude': 1.0,  # Height scale (in inches)
        'noise_scale': 0.08,  # Perlin noise frequency
        'noise_octaves': 4,  # Perlin noise detail
        'elevation_mode': 'perlin',  # 'perlin' or 'image'
        'perspective_strength': 0.6,  # 3D perspective effect (0-1)
        'rotation_x': 25,  # Rotation around X axis (degrees)
        'rotation_z': 35,  # Rotation around Z axis (degrees)
        'cutout_front': True,  # Show front face cutout
        'cutout_side': True,  # Show side face cutout
        'line_density': 1.0,  # Line spacing multiplier
        'smooth_terrain': True,  # Apply smoothing to terrain
        'terrain_center_x': 0.5,  # Center of terrain feature (0-1)
        'terrain_center_z': 0.5,  # Center of terrain feature (0-1)
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
        return "terrain"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "3D topographic terrain visualization with wireframe lines"

    def is_procedural(self) -> bool:
        return True
    
    def generate_paths(self, image_path: str = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate 3D terrain paths.
        
        Args:
            image_path: Optional image for elevation data (if elevation_mode='image')
            
        Returns:
            Tuple of (paths, metadata)
        """
        import numpy as np
        
        # Extract configuration
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        grid_res_x = int(self.config['grid_resolution_x'])
        grid_res_z = int(self.config['grid_resolution_z'])
        cube_height = self.config['cube_height']
        terrain_amplitude = self.config['terrain_amplitude']
        noise_scale = self.config['noise_scale']
        noise_octaves = self.config['noise_octaves']
        elevation_mode = self.config['elevation_mode']
        perspective = self.config['perspective_strength']
        rot_x = math.radians(self.config['rotation_x'])
        rot_z = math.radians(self.config['rotation_z'])
        cutout_front = self.config['cutout_front']
        cutout_side = self.config['cutout_side']
        smooth = self.config['smooth_terrain']
        center_x = self.config['terrain_center_x']
        center_z = self.config['terrain_center_z']
        
        # Calculate drawable area
        drawable_width = output_width_inches - 2 * margin_inches
        drawable_height = output_height_inches - 2 * margin_inches
        
        # Generate random offset for terrain variation
        random_offset = random.random() * 10000
        
        # Generate elevation map
        if elevation_mode == 'image' and image_path:
            elevation_map = self._get_elevation_from_image(image_path, grid_res_x, grid_res_z)
        else:
            elevation_map = self._generate_perlin_terrain(
                grid_res_x, grid_res_z, noise_scale, noise_octaves, center_x, center_z, random_offset
            )
        
        # Apply smoothing if enabled
        if smooth:
            elevation_map = self._smooth_terrain(elevation_map)
        
        # Scale elevation
        elevation_map = elevation_map * terrain_amplitude
        
        # Generate 3D terrain grid
        paths = []
        
        # Draw terrain lines along Z axis (depth lines)
        for x in range(grid_res_x):
            line_path = []
            for z in range(grid_res_z):
                # Normalized coordinates (-0.5 to 0.5)
                x_norm = (x / (grid_res_x - 1)) - 0.5
                z_norm = (z / (grid_res_z - 1)) - 0.5
                # Terrain sits on top of cube: y = cube_height + terrain elevation
                y_norm = cube_height + elevation_map[z, x]
                
                # Apply 3D rotation and perspective projection
                x_2d, y_2d = self._project_3d_to_2d(
                    x_norm, y_norm, z_norm, rot_x, rot_z, perspective
                )
                
                # Scale to drawable area and offset by margin
                x_final = (x_2d + 0.5) * drawable_width + margin_inches
                y_final = (y_2d + 0.5) * drawable_height + margin_inches
                
                line_path.append((x_final, y_final))
            
            if len(line_path) > 1:
                paths.append(line_path)
        
        # Draw terrain lines along X axis (width lines)
        for z in range(grid_res_z):
            line_path = []
            for x in range(grid_res_x):
                x_norm = (x / (grid_res_x - 1)) - 0.5
                z_norm = (z / (grid_res_z - 1)) - 0.5
                # Terrain sits on top of cube: y = cube_height + terrain elevation
                y_norm = cube_height + elevation_map[z, x]
                
                x_2d, y_2d = self._project_3d_to_2d(
                    x_norm, y_norm, z_norm, rot_x, rot_z, perspective
                )
                
                x_final = (x_2d + 0.5) * drawable_width + margin_inches
                y_final = (y_2d + 0.5) * drawable_height + margin_inches
                
                line_path.append((x_final, y_final))
            
            if len(line_path) > 1:
                paths.append(line_path)
        
        # Add cube walls and base
        paths.extend(self._generate_cube_walls(
            grid_res_x, grid_res_z, elevation_map, cube_height, rot_x, rot_z, perspective,
            drawable_width, drawable_height, margin_inches
        ))
        
        # Create metadata
        metadata = {
            'width_inches': output_width_inches,
            'height_inches': output_height_inches,
            'margin_inches': margin_inches,
            'algorithm': 'terrain',
            'grid_resolution_x': grid_res_x,
            'grid_resolution_z': grid_res_z,
            'elevation_mode': elevation_mode
        }
        
        return paths, metadata
    
    def _generate_perlin_terrain(self, width: int, depth: int, scale: float, 
                                   octaves: int, center_x: float, center_z: float, offset: float = 0) -> 'np.ndarray':
        """Generate terrain elevation using Perlin noise."""
        import numpy as np
        
        if not NOISE_AVAILABLE:
            # Fallback: simple sine wave pattern with randomization
            elevation = np.zeros((depth, width))
            for z in range(depth):
                for x in range(width):
                    x_norm = x / width
                    z_norm = z / depth
                    elevation[z, x] = (
                        math.sin((x_norm - center_x + offset * 0.01) * 10 * scale) * 
                        math.cos((z_norm - center_z + offset * 0.013) * 10 * scale) * 0.5
                    )
        else:
            # Use Perlin noise for realistic terrain with random offset
            elevation = np.zeros((depth, width))
            for z in range(depth):
                for x in range(width):
                    x_pos = (x / width - center_x) * scale * 10 + offset
                    z_pos = (z / depth - center_z) * scale * 10 + offset * 1.3
                    
                    value = noise.pnoise2(
                        x_pos, z_pos,
                        octaves=octaves,
                        persistence=0.5,
                        lacunarity=2.0,
                        repeatx=1024,
                        repeaty=1024,
                        base=0
                    )
                    elevation[z, x] = value
        
        # Normalize to sea level: shift so minimum is 0 (all terrain above base)
        min_elevation = np.min(elevation)
        elevation = elevation - min_elevation
        
        return elevation
    
    def _get_elevation_from_image(self, image_path: str, width: int, depth: int) -> 'np.ndarray':
        """Extract elevation data from image brightness."""
        import numpy as np
        
        # Load and process image
        img = Image.open(image_path)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((width, depth), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to 0 to 1 (sea level to max elevation)
        elevation = np.array(img, dtype=float)
        elevation = elevation / 255.0
        
        return elevation
    
    def _smooth_terrain(self, elevation: 'np.ndarray') -> 'np.ndarray':
        """Apply Gaussian smoothing to terrain."""
        import numpy as np
        
        # Simple box blur for smoothing
        kernel_size = 3
        padded = np.pad(elevation, kernel_size // 2, mode='edge')
        smoothed = np.zeros_like(elevation)
        
        for i in range(elevation.shape[0]):
            for j in range(elevation.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                smoothed[i, j] = np.mean(region)
        
        return smoothed
    
    def _project_3d_to_2d(self, x: float, y: float, z: float, 
                           rot_x: float, rot_z: float, perspective: float) -> Tuple[float, float]:
        """
        Project 3D coordinates to 2D using rotation and perspective.
        
        Args:
            x, y, z: 3D coordinates (normalized -0.5 to 0.5)
            rot_x: Rotation around X axis (radians)
            rot_z: Rotation around Z axis (radians)
            perspective: Perspective strength (0-1)
            
        Returns:
            Tuple of (x_2d, y_2d) coordinates
        """
        # Rotate around X axis
        y_rot = y * math.cos(rot_x) - z * math.sin(rot_x)
        z_rot = y * math.sin(rot_x) + z * math.cos(rot_x)
        
        # Rotate around Z axis (vertical)
        x_rot = x * math.cos(rot_z) - z_rot * math.sin(rot_z)
        z_final = x * math.sin(rot_z) + z_rot * math.cos(rot_z)
        
        # Apply perspective projection
        if perspective > 0:
            distance = 2.0  # Camera distance
            scale = distance / (distance + z_final * perspective)
            x_2d = x_rot * scale
            y_2d = y_rot * scale
        else:
            x_2d = x_rot
            y_2d = y_rot
        
        return x_2d, y_2d
    
    def _generate_cube_walls(self, grid_res_x: int, grid_res_z: int, elevation: 'np.ndarray',
                             cube_height: float, rot_x: float, rot_z: float, perspective: float,
                             drawable_width: float, drawable_height: float,
                             margin: float) -> List[List[Tuple[float, float]]]:
        """Generate vertical lines from the 4 corners of terrain to base."""
        paths = []
        
        # Helper function to project and convert to final coordinates
        def project(x_norm, y_norm, z_norm):
            x_2d, y_2d = self._project_3d_to_2d(x_norm, y_norm, z_norm, rot_x, rot_z, perspective)
            x_final = (x_2d + 0.5) * drawable_width + margin
            y_final = (y_2d + 0.5) * drawable_height + margin
            return (x_final, y_final)
        
        # Get the 4 corners of the terrain
        # Front-left corner
        x_norm = -0.5
        z_norm = -0.5
        y_terrain = cube_height + elevation[0, 0]
        terrain_pt = project(x_norm, y_terrain, z_norm)
        base_pt = project(x_norm, 0, z_norm)
        paths.append([terrain_pt, base_pt])
        corner_base_pts = [base_pt]
        
        # Front-right corner
        x_norm = 0.5
        z_norm = -0.5
        y_terrain = cube_height + elevation[0, -1]
        terrain_pt = project(x_norm, y_terrain, z_norm)
        base_pt = project(x_norm, 0, z_norm)
        paths.append([terrain_pt, base_pt])
        corner_base_pts.append(base_pt)
        
        # Back-right corner
        x_norm = 0.5
        z_norm = 0.5
        y_terrain = cube_height + elevation[-1, -1]
        terrain_pt = project(x_norm, y_terrain, z_norm)
        base_pt = project(x_norm, 0, z_norm)
        paths.append([terrain_pt, base_pt])
        corner_base_pts.append(base_pt)
        
        # Back-left corner
        x_norm = -0.5
        z_norm = 0.5
        y_terrain = cube_height + elevation[-1, 0]
        terrain_pt = project(x_norm, y_terrain, z_norm)
        base_pt = project(x_norm, 0, z_norm)
        paths.append([terrain_pt, base_pt])
        corner_base_pts.append(base_pt)
        
        # Draw base outline connecting the 4 corners
        base_outline = corner_base_pts + [corner_base_pts[0]]  # Close the loop
        paths.append(base_outline)
        
        return paths
    
    def generate_svg(self, paths: List[List[Tuple[float, float]]], 
                     metadata: Dict[str, Any]) -> str:
        """Generate SVG from paths."""
        width_inches = metadata['width_inches']
        height_inches = metadata['height_inches']
        
        # SVG header
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            f'<svg width="{width_inches}in" height="{height_inches}in" ',
            f'viewBox="0 0 {width_inches} {height_inches}" ',
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width_inches}" height="{height_inches}" fill="white"/>',
            f'<g stroke="black" stroke-width="0.01" fill="none">'
        ]
        
        # Add paths
        for path in paths:
            if len(path) < 2:
                continue
            
            path_d = f'M {path[0][0]:.4f},{path[0][1]:.4f}'
            for x, y in path[1:]:
                path_d += f' L {x:.4f},{y:.4f}'
            
            svg_parts.append(f'<path d="{path_d}"/>')
        
        svg_parts.append('</g>')
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def generate_preview(self, paths: List[List[Tuple[float, float]]], 
                         metadata: Dict[str, Any], preview_dpi: int = 100) -> Image.Image:
        """Generate a preview image."""
        width_inches = metadata['width_inches']
        height_inches = metadata['height_inches']
        
        width_px = int(width_inches * preview_dpi)
        height_px = int(height_inches * preview_dpi)
        
        preview = Image.new('RGB', (width_px, height_px), 'white')
        draw = ImageDraw.Draw(preview)
        
        for path in paths:
            if len(path) < 2:
                continue
            
            pixel_path = [
                (x * preview_dpi, y * preview_dpi)
                for x, y in path
            ]
            
            draw.line(pixel_path, fill='black', width=1)
        
        return preview
