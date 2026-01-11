"""
fabric.py - Fabric texture plot generation

Generates fabric-like patterns using Perlin noise or ripple displacement.
Can be procedural or image-based depending on displacement_mode.
"""

import math
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw

from .base import PlotAlgorithm

class FabricGenerator(PlotAlgorithm):
    """
    Generates fabric texture plots using Perlin noise displacement.
    
    Unlike waves and circles algorithms which convert input images to plots,
    this algorithm generates original fabric patterns from scratch using
    procedural noise algorithms.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.125,
        'calc_dpi': 300,
        'grid_cols': 150,
        'grid_rows': 200,
        'displacement_mode': 'random',  # 'random', 'ripples', or 'image'
        'noise_scale_x': 0.002,
        'noise_scale_y': 0.002,
        'noise_octaves': 4,
        'noise_persistence': 0.5,
        'noise_lacunarity': 2.0,
        'displacement_magnitude': 500,
        'ripple_frequency': 0.015,  # Frequency of ripples (smaller = wider ripples)
        'ripple_amplitude': 400,   # Amplitude of ripple displacement
        'ripple_centers': 1,  # Number of ripple centers (>= 1)
        'ripple_centers_locations': [],  # Optional explicit centers: [[x, y], ...] (0..1 normalized or px)
        'image_displacement_scale': 500,  # Scale factor for image-based displacement
        'enable_weave': True,  # Enable the over/under weave effect
        'line_direction': 'horizontal',  # 'horizontal', 'vertical', or 'both'
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
        return "fabric"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Procedurally generated fabric texture using Perlin noise"

    def is_procedural(self) -> bool:
        return True
    
    def generate_paths(self, image_path: str = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate fabric texture paths.
        
        Args:
            image_path: Not used (fabric generates patterns, doesn't convert images)
            
        Returns:
            Tuple of (paths, metadata)
        """
        # Import noise library for Perlin noise
        try:
            import noise
        except ImportError:
            raise ImportError("The 'noise' library is required for fabric generation. Install with: pip install noise")
        
        import numpy as np
        import random
        
        # Extract config
        calc_dpi = self.config['calc_dpi']
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        grid_cols = self.config['grid_cols']
        grid_rows = self.config['grid_rows']
        displacement_mode = self.config.get('displacement_mode', 'random')
        noise_scale_x = self.config['noise_scale_x']
        noise_scale_y = self.config['noise_scale_y']
        noise_octaves = self.config['noise_octaves']
        noise_persistence = self.config['noise_persistence']
        noise_lacunarity = self.config['noise_lacunarity']
        displacement_magnitude = self.config['displacement_magnitude']
        ripple_frequency = self.config.get('ripple_frequency', 0.02)
        ripple_amplitude = self.config.get('ripple_amplitude', 300)
        ripple_centers = int(self.config.get('ripple_centers', 1) or 1)
        ripple_centers_locations = self.config.get('ripple_centers_locations', [])
        image_displacement_scale = self.config.get('image_displacement_scale', 500)
        enable_weave = self.config['enable_weave']
        line_direction = self.config.get('line_direction', 'horizontal')  # 'horizontal', 'vertical', or 'both'
        
        # Calculate dimensions
        total_width_px = int(output_width_inches * calc_dpi)
        total_height_px = int(output_height_inches * calc_dpi)
        margin_px = int(margin_inches * calc_dpi)
        
        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        
        if drawable_width_px <= 0 or drawable_height_px <= 0:
            raise ValueError("Margins too large for output size")
        
        # Create base grid - no padding to respect margins
        x = np.linspace(0, drawable_width_px, grid_cols)
        y = np.linspace(0, drawable_height_px, grid_rows)
        X, Y = np.meshgrid(x, y)
        
        # Generate displacement based on mode
        noised_X = np.zeros_like(X)
        noised_Y = np.zeros_like(Y)
        
        if displacement_mode == 'ripples':
            if ripple_centers < 1:
                ripple_centers = 1

            # Determine ripple centers.
            # If `ripple_centers_locations` is provided, use it (normalized 0..1 or px).
            # If fewer than `ripple_centers`, fill the remainder randomly.
            # If more than `ripple_centers`, only the first N are used.
            centers = []
            if isinstance(ripple_centers_locations, list) and len(ripple_centers_locations) > 0:
                for item in ripple_centers_locations:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    try:
                        cx = float(item[0])
                        cy = float(item[1])
                    except (TypeError, ValueError):
                        continue

                    # If the point looks normalized (both components in 0..1), scale to pixels.
                    if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                        cx *= float(drawable_width_px)
                        cy *= float(drawable_height_px)

                    # Clamp into drawable bounds.
                    cx = max(0.0, min(float(drawable_width_px), cx))
                    cy = max(0.0, min(float(drawable_height_px), cy))
                    centers.append((cx, cy))

                centers = centers[:ripple_centers]

            while len(centers) < ripple_centers:
                centers.append(
                    (
                        random.uniform(0.2 * drawable_width_px, 0.8 * drawable_width_px),
                        random.uniform(0.2 * drawable_height_px, 0.8 * drawable_height_px),
                    )
                )

            # Use Perlin noise to warp ripple spacing/phase so rings aren't perfectly periodic.
            # This keeps the overall radial "push/pull" feel, but breaks up the fixed intervals.
            # Default warp is a fraction of the ripple wavelength (≈ 2π / ripple_frequency)
            # so you still get visible irregularity even if ripple_amplitude is small.
            default_ripple_noise_warp = (0.7 * math.pi) / max(float(ripple_frequency), 1e-9)
            ripple_noise_warp = self.config.get('ripple_noise_warp', default_ripple_noise_warp)
            ripple_frequency_jitter = self.config.get('ripple_frequency_jitter', 0.35)
            ripple_amplitude_jitter = self.config.get('ripple_amplitude_jitter', 0.25)
            ripple_seed = random.randint(0, 1000)
            
            # Calculate distance from center for each grid point
            for i in range(grid_rows):
                for j in range(grid_cols):
                    # Perlin noise value in [-1, 1] for this point.
                    # Reuse the configured noise parameters to control "organic" variation.
                    n = noise.pnoise2(
                        X[i, j] * noise_scale_x,
                        Y[i, j] * noise_scale_y,
                        octaves=noise_octaves,
                        persistence=noise_persistence,
                        lacunarity=noise_lacunarity,
                        repeatx=drawable_width_px,
                        repeaty=drawable_height_px,
                        base=ripple_seed,
                    )

                    # Warp the ripple phase and slightly vary local frequency/amplitude.
                    local_frequency = ripple_frequency * (1.0 + n * ripple_frequency_jitter)
                    local_amplitude = 1.0 + n * ripple_amplitude_jitter

                    # Combine contributions from multiple ripple centers.
                    # Average so overall amplitude stays comparable when increasing centers.
                    accum_x = 0.0
                    accum_y = 0.0
                    for center_x, center_y in centers:
                        dx = X[i, j] - center_x
                        dy = Y[i, j] - center_y
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= 1e-9:
                            continue

                        warped_distance = distance + (n * ripple_noise_warp)
                        ripple = np.sin(warped_distance * local_frequency) * local_amplitude

                        dx_norm = dx / distance
                        dy_norm = dy / distance
                        accum_x += dx_norm * ripple
                        accum_y += dy_norm * ripple

                    divisor = float(len(centers)) if len(centers) > 0 else 1.0
                    noised_X[i, j] = accum_x / divisor
                    noised_Y[i, j] = accum_y / divisor
            
            # Scale by ripple amplitude
            X_distorted = X + noised_X * ripple_amplitude
            Y_distorted = Y + noised_Y * ripple_amplitude
        
        elif displacement_mode == 'image':
            # Image-based displacement mode
            if image_path is None:
                raise ValueError("Image path is required for 'image' displacement mode")
            
            # Load and process the image
            from PIL import Image, ImageOps
            
            img = Image.open(image_path)
            img = ImageOps.grayscale(img)
            
            # Resize image to match grid dimensions for sampling
            img_resized = img.resize((grid_cols, grid_rows), Image.LANCZOS)
            
            # Convert to numpy array (values 0-255)
            img_array = np.array(img_resized)
            
            # Normalize to -1 to 1 range (darker = negative displacement, lighter = positive)
            img_normalized = (img_array / 255.0) * 2 - 1
            
            # Create displacement based on image brightness
            # We'll use the brightness to determine displacement magnitude
            for i in range(grid_rows):
                for j in range(grid_cols):
                    # Get brightness value at this grid point
                    brightness = img_normalized[i, j]
                    
                    # Calculate gradient-based displacement direction
                    # This creates a 3D-like effect where bright areas push out
                    # and dark areas push in
                    
                    # Calculate local gradient (if not at edge)
                    grad_x = 0
                    grad_y = 0
                    
                    if j > 0 and j < grid_cols - 1:
                        grad_x = (img_normalized[i, min(j+1, grid_cols-1)] - 
                                 img_normalized[i, max(j-1, 0)]) / 2
                    
                    if i > 0 and i < grid_rows - 1:
                        grad_y = (img_normalized[min(i+1, grid_rows-1), j] - 
                                 img_normalized[max(i-1, 0), j]) / 2
                    
                    # Use both brightness and gradient for displacement
                    # Brightness creates the overall depth
                    # Gradient creates the directional flow
                    noised_X[i, j] = grad_x + brightness * 0.3
                    noised_Y[i, j] = grad_y + brightness * 0.3
            
            # Scale by image displacement magnitude
            X_distorted = X + noised_X * image_displacement_scale
            Y_distorted = Y + noised_Y * image_displacement_scale
        
        else:  # displacement_mode == 'random' (default)
            # Generate Perlin noise for displacement
            seed_x = random.randint(0, 1000)
            seed_y = random.randint(0, 1000)
            
            for i in range(grid_rows):
                for j in range(grid_cols):
                    nx = noise.pnoise2(
                        X[i, j] * noise_scale_x, 
                        Y[i, j] * noise_scale_y,
                        octaves=noise_octaves,
                        persistence=noise_persistence,
                        lacunarity=noise_lacunarity,
                        repeatx=drawable_width_px, 
                        repeaty=drawable_height_px,
                        base=seed_x
                    )
                    ny = noise.pnoise2(
                        X[i, j] * noise_scale_x, 
                        Y[i, j] * noise_scale_y,
                        octaves=noise_octaves,
                        persistence=noise_persistence,
                        lacunarity=noise_lacunarity,
                        repeatx=drawable_width_px, 
                        repeaty=drawable_height_px,
                        base=seed_y
                    )
                    
                    noised_X[i, j] = nx
                    noised_Y[i, j] = ny
            
            # Apply displacement
            X_distorted = X + noised_X * displacement_magnitude
            Y_distorted = Y + noised_Y * displacement_magnitude
        
        # Generate paths based on line_direction
        horizontal_paths = []
        vertical_paths = []
        
        # Helper function to add horizontal lines
        def add_horizontal_lines():
            for i in range(grid_rows):
                if enable_weave:
                    # Generate weave pattern with gaps
                    for j in range(grid_cols - 1):
                        p1_x = X_distorted[i, j] + margin_px
                        p1_y = Y_distorted[i, j] + margin_px
                        p2_x = X_distorted[i, j+1] + margin_px
                        p2_y = Y_distorted[i, j+1] + margin_px
                        
                        # Checkerboard pattern for over/under
                        if (i + j) % 2 == 0:
                            # "Over" - full segment
                            horizontal_paths.append([(p1_x, p1_y), (p2_x, p2_y)])
                        else:
                            # "Under" - segmented with gap
                            mid1_x = p1_x + 0.25 * (p2_x - p1_x)
                            mid1_y = p1_y + 0.25 * (p2_y - p1_y)
                            mid2_x = p1_x + 0.75 * (p2_x - p1_x)
                            mid2_y = p1_y + 0.75 * (p2_y - p1_y)
                            
                            horizontal_paths.append([(p1_x, p1_y), (mid1_x, mid1_y)])
                            horizontal_paths.append([(mid2_x, mid2_y), (p2_x, p2_y)])
                else:
                    # Simple continuous lines
                    path = []
                    for j in range(grid_cols):
                        x = X_distorted[i, j] + margin_px
                        y = Y_distorted[i, j] + margin_px
                        path.append((x, y))
                    
                    if path:
                        horizontal_paths.append(path)
        
        # Helper function to add vertical lines
        def add_vertical_lines():
            for j in range(grid_cols):
                if enable_weave:
                    # Generate weave pattern with gaps
                    for i in range(grid_rows - 1):
                        p1_x = X_distorted[i, j] + margin_px
                        p1_y = Y_distorted[i, j] + margin_px
                        p2_x = X_distorted[i+1, j] + margin_px
                        p2_y = Y_distorted[i+1, j] + margin_px
                        
                        # Checkerboard pattern for over/under (reversed for vertical)
                        if (i + j) % 2 == 1:
                            # "Over" - full segment
                            vertical_paths.append([(p1_x, p1_y), (p2_x, p2_y)])
                        else:
                            # "Under" - segmented with gap
                            mid1_x = p1_x + 0.25 * (p2_x - p1_x)
                            mid1_y = p1_y + 0.25 * (p2_y - p1_y)
                            mid2_x = p1_x + 0.75 * (p2_x - p1_x)
                            mid2_y = p1_y + 0.75 * (p2_y - p1_y)
                            
                            vertical_paths.append([(p1_x, p1_y), (mid1_x, mid1_y)])
                            vertical_paths.append([(mid2_x, mid2_y), (p2_x, p2_y)])
                else:
                    # Simple continuous lines
                    path = []
                    for i in range(grid_rows):
                        x = X_distorted[i, j] + margin_px
                        y = Y_distorted[i, j] + margin_px
                        path.append((x, y))
                    
                    if path:
                        vertical_paths.append(path)
        
        # Generate lines based on direction setting
        if line_direction == 'horizontal':
            add_horizontal_lines()
            all_paths = horizontal_paths
        elif line_direction == 'vertical':
            add_vertical_lines()
            all_paths = vertical_paths
        elif line_direction == 'both':
            add_horizontal_lines()
            add_vertical_lines()
            # Return as dict for separate SVG handling
            all_paths = {
                'horizontal': horizontal_paths,
                'vertical': vertical_paths
            }
        else:
            # Default to horizontal if invalid option
            add_horizontal_lines()
            all_paths = horizontal_paths
        
        # Create metadata
        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': output_width_inches,
            'output_height_inches': output_height_inches,
            'algorithm': self.get_algorithm_name(),
            'line_direction': line_direction
        }
        
        return all_paths, metadata
    
    def generate_svg(self, paths, metadata: Dict[str, Any]):
        """
        Generate SVG content from paths.
        
        Args:
            paths: List of path segments OR dict with 'horizontal' and 'vertical' keys
            metadata: Metadata dict from generate_paths
            
        Returns:
            SVG content as string OR dict of {direction: svg_content} for both directions
        """
        # Check if paths is a dict (both directions)
        if isinstance(paths, dict) and 'horizontal' in paths and 'vertical' in paths:
            # Generate separate SVGs for each direction
            return {
                'horizontal': self._generate_svg_from_paths(paths['horizontal'], metadata),
                'vertical': self._generate_svg_from_paths(paths['vertical'], metadata)
            }
        else:
            # Single direction - generate one SVG
            return self._generate_svg_from_paths(paths, metadata)
    
    def _generate_svg_from_paths(self, paths: List[List[Tuple[float, float]]], 
                                  metadata: Dict[str, Any]) -> str:
        """
        Internal method to generate SVG from a list of paths.
        """
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        output_width_inches = metadata['output_width_inches']
        output_height_inches = metadata['output_height_inches']
        
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{output_width_inches}in" '
            f'height="{output_height_inches}in" '
            f'viewBox="0 0 {total_width_px} {total_height_px}">\n'
        ]
        
        for path in paths:
            if len(path) < 2:
                continue
            
            d_str = f"M {path[0][0]:.2f} {path[0][1]:.2f} "
            for p in path[1:]:
                d_str += f"L {p[0]:.2f} {p[1]:.2f} "
            
            svg_lines.append(
                f'<path d="{d_str}" fill="none" stroke="black" stroke-width="0.5" />\n'
            )
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def generate_preview(self, paths, metadata: Dict[str, Any]) -> Image.Image:
        """
        Generate a preview PNG from paths.
        
        Args:
            paths: List of path segments OR dict with 'horizontal' and 'vertical' keys
            metadata: Metadata dict from generate_paths
            
        Returns:
            PIL Image object
        """
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        
        preview_img = Image.new('RGB', (total_width_px, total_height_px), 'white')
        drawer = ImageDraw.Draw(preview_img)
        
        # Handle both dict and list paths
        if isinstance(paths, dict) and 'horizontal' in paths and 'vertical' in paths:
            # Draw both horizontal and vertical lines
            for path in paths['horizontal']:
                if len(path) > 1:
                    drawer.line(path, fill='black', width=1, joint='curve')
            for path in paths['vertical']:
                if len(path) > 1:
                    drawer.line(path, fill='black', width=1, joint='curve')
        else:
            # Single direction
            for path in paths:
                if len(path) > 1:
                    drawer.line(path, fill='black', width=1, joint='curve')
        
        return preview_img
