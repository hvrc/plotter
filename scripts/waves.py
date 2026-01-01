"""
waves.py - Wave-based plot generation algorithm

This module implements a wave-based algorithm for converting images into plottable SVG paths.
It's designed to be one of potentially many plot generation algorithms.
"""

import math
from PIL import Image, ImageOps, ImageDraw
from typing import List, Tuple, Dict, Any


class WavesGenerator:
    """
    Generates wave-based plots from images.
    
    This class implements the algorithm interface that all plot generators should follow:
    - __init__ with config dict
    - generate_paths(image_path) -> List of path segments
    - get_config() -> current configuration
    - set_config(config) -> update configuration
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.125,
        'n_rows': 100,
        'contrast_power': 1.5,
        'amplitude_scale': 1,
        'frequency_scale': 1.75,
        'white_threshold': 250,
        'use_serpentine': False,
        'calc_dpi': 300,
        'draw_direction': 'horizontal',  # 'horizontal' or 'vertical'
        'numColors': 1  # Number of dominant colors (1 = single black layer)
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
        return "waves"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Wave-based plot generation using sine wave modulation"
    
    def _quantize_colors(self, image: Image.Image, num_colors: int) -> Tuple[Image.Image, List[Tuple[int, int, int]]]:
        """
        Reduce image to N dominant colors using k-means clustering.
        
        Args:
            image: RGB PIL Image
            num_colors: Number of dominant colors to extract
            
        Returns:
            Tuple of (quantized_image, list_of_dominant_colors)
        """
        import numpy as np
        from collections import Counter
        
        # Convert image to RGB if not already
        rgb_image = image.convert('RGB')
        
        # Use PIL's built-in quantization for efficiency
        quantized = rgb_image.quantize(colors=num_colors, method=2)  # method=2 = median cut
        quantized_rgb = quantized.convert('RGB')
        
        # Extract the actual dominant colors from the quantized image
        pixels = list(quantized_rgb.getdata())
        color_counts = Counter(pixels)
        dominant_colors = [color for color, count in color_counts.most_common(num_colors)]
        
        return quantized_rgb, dominant_colors
    
    def _create_color_mask(self, image: Image.Image, target_color: Tuple[int, int, int], tolerance: int = 10) -> Image.Image:
        """
        Create a binary mask for pixels matching the target color.
        
        Args:
            image: RGB PIL Image
            target_color: RGB tuple to match
            tolerance: Color matching tolerance
            
        Returns:
            Grayscale mask image (255 = match, 0 = no match)
        """
        import numpy as np
        
        img_array = np.array(image)
        target = np.array(target_color)
        
        # Calculate color distance
        distance = np.sqrt(np.sum((img_array - target) ** 2, axis=2))
        
        # Create binary mask
        mask = (distance <= tolerance).astype(np.uint8) * 255
        
        return Image.fromarray(mask, mode='L')
    
    def generate_paths(self, image_path: str) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate plot paths from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (paths, metadata)
            - paths: List of path segments OR dict of {color: paths} if numColors > 1
            - metadata: Dict with dimensions and other info for SVG generation
        """
        try:
            original_img = Image.open(image_path)
        except IOError as e:
            raise ValueError(f"Could not open image: {image_path}") from e
        
        # Extract config
        calc_dpi = self.config['calc_dpi']
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        n_rows = self.config['n_rows']
        contrast_power = self.config['contrast_power']
        amplitude_scale = self.config['amplitude_scale']
        frequency_scale = self.config['frequency_scale']
        white_threshold = self.config['white_threshold']
        use_serpentine = self.config['use_serpentine']
        draw_direction = self.config.get('draw_direction', 'horizontal')
        num_colors = self.config.get('numColors', 1)
        
        # Calculate dimensions
        total_width_px = int(output_width_inches * calc_dpi)
        total_height_px = int(output_height_inches * calc_dpi)
        margin_px = int(margin_inches * calc_dpi)
        
        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        
        if drawable_width_px <= 0 or drawable_height_px <= 0:
            raise ValueError("Margins are larger than paper size")
        
        # Handle color-aware generation
        if num_colors > 1:
            # Quantize colors
            rgb_img = original_img.convert('RGB')
            fitted_rgb = ImageOps.fit(
                rgb_img,
                (drawable_width_px, drawable_height_px),
                Image.Resampling.LANCZOS
            )
            quantized_img, dominant_colors = self._quantize_colors(fitted_rgb, num_colors)
            
            # Generate paths for each color
            color_paths = {}
            for i, color in enumerate(dominant_colors):
                # Create mask for this color
                mask = self._create_color_mask(quantized_img, color, tolerance=15)
                
                # Convert mask to what the algorithm expects (inverted: dark=wave, light=skip)
                mask_inverted = ImageOps.invert(mask)
                
                # Generate paths using the mask
                if draw_direction == 'vertical':
                    paths = self._generate_vertical_paths(
                        mask_inverted, n_rows, drawable_width_px, drawable_height_px,
                        margin_px, contrast_power, amplitude_scale, frequency_scale,
                        white_threshold, use_serpentine
                    )
                else:
                    paths = self._generate_horizontal_paths(
                        mask_inverted, n_rows, drawable_width_px, drawable_height_px,
                        margin_px, contrast_power, amplitude_scale, frequency_scale,
                        white_threshold, use_serpentine
                    )
                
                color_paths[color] = paths
            
            all_paths = color_paths
            dominant_colors_rgb = [(int(c[0]), int(c[1]), int(c[2])) for c in dominant_colors]
        else:
            # Single color (grayscale) generation - original behavior
            gray_img = original_img.convert('L')
            processed_img = ImageOps.fit(
                gray_img, 
                (drawable_width_px, drawable_height_px), 
                Image.Resampling.LANCZOS
            )
            processed_img = ImageOps.autocontrast(processed_img)
            
            # Choose drawing direction
            if draw_direction == 'vertical':
                all_paths = self._generate_vertical_paths(
                    processed_img, n_rows, drawable_width_px, drawable_height_px,
                    margin_px, contrast_power, amplitude_scale, frequency_scale,
                    white_threshold, use_serpentine
                )
            else:  # horizontal (default)
                all_paths = self._generate_horizontal_paths(
                    processed_img, n_rows, drawable_width_px, drawable_height_px,
                    margin_px, contrast_power, amplitude_scale, frequency_scale,
                    white_threshold, use_serpentine
                )
            dominant_colors_rgb = None
        
        # Create metadata for SVG generation
        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': output_width_inches,
            'output_height_inches': output_height_inches,
            'algorithm': self.get_algorithm_name(),
            'numColors': num_colors,
            'dominant_colors': dominant_colors_rgb
        }
        
        return all_paths, metadata
    
    def generate_svg(self, paths: List[List[Tuple[float, float]]], 
                     metadata: Dict[str, Any]) -> str:
        """
        Generate SVG content from paths.
        
        Args:
            paths: List of path segments OR dict of {color: paths} for multi-color
            metadata: Metadata dict from generate_paths
            
        Returns:
            SVG content as string OR dict of {color: svg_content} for multi-color
        """
        num_colors = metadata.get('numColors', 1)
        
        if num_colors > 1 and isinstance(paths, dict):
            # Multi-color mode: return dict of SVGs
            svg_dict = {}
            for color, color_paths in paths.items():
                svg_content = self._generate_single_svg(color_paths, metadata, color)
                svg_dict[color] = svg_content
            return svg_dict
        else:
            # Single color mode: return single SVG
            return self._generate_single_svg(paths, metadata, (0, 0, 0))
    
    def _generate_single_svg(self, paths: List[List[Tuple[float, float]]], 
                            metadata: Dict[str, Any], 
                            stroke_color: Tuple[int, int, int]) -> str:
        """
        Generate a single SVG file.
        
        Args:
            paths: List of path segments
            metadata: Metadata dict
            stroke_color: RGB color tuple for stroke
            
        Returns:
            SVG content as string
        """
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        output_width_inches = metadata['output_width_inches']
        output_height_inches = metadata['output_height_inches']
        
        # Convert RGB to hex
        color_hex = f"#{stroke_color[0]:02x}{stroke_color[1]:02x}{stroke_color[2]:02x}"
        
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
                f'<path d="{d_str}" fill="none" stroke="{color_hex}" stroke-width="2" />\n'
            )
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def _generate_horizontal_paths(self, processed_img, n_rows, drawable_width_px, 
                                   drawable_height_px, margin_px, contrast_power, 
                                   amplitude_scale, frequency_scale, white_threshold, 
                                   use_serpentine) -> List[List[Tuple[float, float]]]:
        """Generate paths drawing from left to right (horizontal)."""
        sample_img = processed_img.resize(
            (drawable_width_px, n_rows), 
            Image.Resampling.LANCZOS
        )
        pixels = sample_img.load()
        
        row_height = drawable_height_px / n_rows
        max_amplitude = row_height * amplitude_scale
        
        all_paths = []
        
        for r in range(n_rows):
            row_segments = []
            current_segment = []
            
            baseline_y_local = (r * row_height) + (row_height / 2)
            theta = 0.0
            
            for x_local in range(drawable_width_px):
                brightness = pixels[x_local, r]
                val = (255 - brightness) / 255.0
                
                val_contrast = val ** contrast_power
                step = (0.05 + (1.0 * val_contrast)) * frequency_scale
                theta += step
                
                is_blank = brightness > white_threshold
                
                if not is_blank:
                    current_amp = max_amplitude * val_contrast
                    y_offset = current_amp * math.sin(theta)
                    y_local = baseline_y_local + y_offset
                    
                    final_x = x_local + margin_px
                    final_y = y_local + margin_px
                    
                    current_segment.append((final_x, final_y))
                else:
                    if len(current_segment) > 1:
                        row_segments.append(current_segment)
                    current_segment = []
            
            if len(current_segment) > 1:
                row_segments.append(current_segment)
            
            if use_serpentine and (r % 2 == 1):
                row_segments.reverse()
                for seg in row_segments:
                    seg.reverse()
            
            all_paths.extend(row_segments)
        
        return all_paths
    
    def _generate_vertical_paths(self, processed_img, n_rows, drawable_width_px, 
                                drawable_height_px, margin_px, contrast_power, 
                                amplitude_scale, frequency_scale, white_threshold, 
                                use_serpentine) -> List[List[Tuple[float, float]]]:
        """Generate paths drawing from top to bottom (vertical)."""
        sample_img = processed_img.resize(
            (n_rows, drawable_height_px), 
            Image.Resampling.LANCZOS
        )
        pixels = sample_img.load()
        
        col_width = drawable_width_px / n_rows
        max_amplitude = col_width * amplitude_scale
        
        all_paths = []
        
        for c in range(n_rows):
            col_segments = []
            current_segment = []
            
            baseline_x_local = (c * col_width) + (col_width / 2)
            theta = 0.0
            
            for y_local in range(drawable_height_px):
                brightness = pixels[c, y_local]
                val = (255 - brightness) / 255.0
                
                val_contrast = val ** contrast_power
                step = (0.05 + (1.0 * val_contrast)) * frequency_scale
                theta += step
                
                is_blank = brightness > white_threshold
                
                if not is_blank:
                    current_amp = max_amplitude * val_contrast
                    x_offset = current_amp * math.sin(theta)
                    x_local = baseline_x_local + x_offset
                    
                    final_x = x_local + margin_px
                    final_y = y_local + margin_px
                    
                    current_segment.append((final_x, final_y))
                else:
                    if len(current_segment) > 1:
                        col_segments.append(current_segment)
                    current_segment = []
            
            if len(current_segment) > 1:
                col_segments.append(current_segment)
            
            if use_serpentine and (c % 2 == 1):
                col_segments.reverse()
                for seg in col_segments:
                    seg.reverse()
            
            all_paths.extend(col_segments)
        
        return all_paths
    
    def generate_preview(self, paths: List[List[Tuple[float, float]]], 
                        metadata: Dict[str, Any]) -> Image.Image:
        """
        Generate a preview PNG from paths.
        
        Args:
            paths: List of path segments OR dict of {color: paths} for multi-color
            metadata: Metadata dict from generate_paths
            
        Returns:
            PIL Image object
        """
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        num_colors = metadata.get('numColors', 1)
        
        preview_img = Image.new('RGB', (total_width_px, total_height_px), 'white')
        drawer = ImageDraw.Draw(preview_img)
        
        if num_colors > 1 and isinstance(paths, dict):
            # Multi-color mode: draw each color separately
            for color, color_paths in paths.items():
                color_rgb = tuple(int(c) for c in color)
                for path in color_paths:
                    if len(path) > 1:
                        drawer.line(path, fill=color_rgb, width=2, joint='curve')
        else:
            # Single color mode
            for path in paths:
                if len(path) > 1:
                    drawer.line(path, fill='black', width=2, joint='curve')
        
        return preview_img


class FabricGenerator:
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
        'displacement_mode': 'random',  # 'random' or 'ripples'
        'noise_scale_x': 0.002,
        'noise_scale_y': 0.002,
        'noise_octaves': 4,
        'noise_persistence': 0.5,
        'noise_lacunarity': 2.0,
        'displacement_magnitude': 500,
        'ripple_frequency': 0.015,  # Frequency of ripples (smaller = wider ripples)
        'ripple_amplitude': 400,   # Amplitude of ripple displacement
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
            # Pick a random center point for the ripples
            center_x = random.uniform(0.2 * drawable_width_px, 0.8 * drawable_width_px)
            center_y = random.uniform(0.2 * drawable_height_px, 0.8 * drawable_height_px)
            
            # Calculate distance from center for each grid point
            for i in range(grid_rows):
                for j in range(grid_cols):
                    # Distance from center (this creates the spherical/circular pattern)
                    dx = X[i, j] - center_x
                    dy = Y[i, j] - center_y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Create ripple effect using sine wave based on distance
                    # The sine creates the concentric wave pattern
                    ripple = np.sin(distance * ripple_frequency)
                    
                    # Apply radial displacement - pushing outward/inward from center
                    # This creates spherical ripples emanating from the center point
                    if distance > 0:
                        # Normalize direction vector (radial direction)
                        dx_norm = dx / distance
                        dy_norm = dy / distance
                        
                        # Apply ripple displacement in radial direction
                        noised_X[i, j] = dx_norm * ripple
                        noised_Y[i, j] = dy_norm * ripple
            
            # Scale by ripple amplitude
            X_distorted = X + noised_X * ripple_amplitude
            Y_distorted = Y + noised_Y * ripple_amplitude
        
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


class CirclesGenerator:
    """
    Generates concentric circle-based plots from images.
    
    This algorithm draws circles from the center outward, modulating the radius
    with squiggles based on image brightness to capture detail.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'output_width_inches': 4.5,
        'output_height_inches': 6.0,
        'margin_inches': 0.125,
        'n_circles': 80,
        'contrast_power': 1.5,
        'amplitude_scale': 1,
        'frequency_scale': 2.0,
        'white_threshold': 250,
        'calc_dpi': 300,
        'points_per_circle': 360  # Number of sample points per circle
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
        return "circles"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Concentric circles with squiggles based on image brightness"
    
    def generate_paths(self, image_path: str) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate plot paths from an image using concentric circles.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (paths, metadata)
        """
        try:
            original_img = Image.open(image_path)
        except IOError as e:
            raise IOError(f"Cannot open image: {e}")
        
        # Extract config
        calc_dpi = self.config['calc_dpi']
        output_width_inches = self.config['output_width_inches']
        output_height_inches = self.config['output_height_inches']
        margin_inches = self.config['margin_inches']
        n_circles = self.config['n_circles']
        contrast_power = self.config['contrast_power']
        amplitude_scale = self.config['amplitude_scale']
        frequency_scale = self.config['frequency_scale']
        white_threshold = self.config['white_threshold']
        points_per_circle = self.config['points_per_circle']
        
        # Calculate dimensions
        total_width_px = int(output_width_inches * calc_dpi)
        total_height_px = int(output_height_inches * calc_dpi)
        margin_px = int(margin_inches * calc_dpi)
        
        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        
        if drawable_width_px <= 0 or drawable_height_px <= 0:
            raise ValueError("Margins too large for output size")
        
        # Process image
        gray_img = original_img.convert('L')
        processed_img = ImageOps.fit(
            gray_img, 
            (drawable_width_px, drawable_height_px), 
            Image.Resampling.LANCZOS
        )
        processed_img = ImageOps.autocontrast(processed_img)
        
        # Generate circular paths
        paths = self._generate_circular_paths(
            processed_img, n_circles, drawable_width_px, drawable_height_px,
            margin_px, contrast_power, amplitude_scale, frequency_scale,
            white_threshold, points_per_circle
        )
        
        # Create metadata for SVG generation
        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': output_width_inches,
            'output_height_inches': output_height_inches,
            'algorithm': self.get_algorithm_name()
        }
        
        return paths, metadata
    
    def _generate_circular_paths(self, processed_img, n_circles, drawable_width_px,
                                 drawable_height_px, margin_px, contrast_power,
                                 amplitude_scale, frequency_scale, white_threshold,
                                 points_per_circle) -> List[List[Tuple[float, float]]]:
        """Generate concentric circular paths from center outward."""
        import math
        
        pixels = processed_img.load()
        img_width, img_height = processed_img.size
        
        # Center of the drawable area
        center_x = drawable_width_px / 2 + margin_px
        center_y = drawable_height_px / 2 + margin_px
        
        # Maximum radius (to edge of drawable area)
        max_radius = min(drawable_width_px, drawable_height_px) / 2
        
        all_paths = []
        
        for circle_idx in range(n_circles):
            # Base radius for this circle
            base_radius = (circle_idx + 1) * (max_radius / n_circles)
            
            # Maximum amplitude for squiggles (proportional to spacing between circles)
            max_amplitude = (max_radius / n_circles) * amplitude_scale * 0.8
            
            path = []
            
            for point_idx in range(points_per_circle + 1):  # +1 to close the circle
                # Angle around the circle
                angle = (point_idx / points_per_circle) * 2 * math.pi
                
                # Position in image space for sampling
                img_x = int((base_radius * math.cos(angle) + drawable_width_px / 2))
                img_y = int((base_radius * math.sin(angle) + drawable_height_px / 2))
                
                # Clamp to image bounds
                img_x = max(0, min(img_width - 1, img_x))
                img_y = max(0, min(img_height - 1, img_y))
                
                # Sample brightness
                brightness = pixels[img_x, img_y]
                
                # Skip if too bright (white)
                if brightness >= white_threshold:
                    # For very bright areas, stay at base radius (no squiggle)
                    amplitude = 0
                else:
                    # Normalize brightness (darker = higher value)
                    normalized = (255 - brightness) / 255.0
                    normalized = normalized ** contrast_power
                    
                    # Calculate squiggle amplitude
                    amplitude = normalized * max_amplitude
                    
                    # Add high-frequency detail using sine wave
                    detail_angle = point_idx * frequency_scale
                    detail_offset = math.sin(detail_angle) * amplitude * 0.3
                    amplitude += detail_offset
                
                # Calculate actual radius with squiggle
                actual_radius = base_radius + amplitude
                
                # Convert to canvas coordinates
                x = center_x + actual_radius * math.cos(angle)
                y = center_y + actual_radius * math.sin(angle)
                
                path.append((x, y))
            
            if path:
                all_paths.append(path)
        
        return all_paths
    
    def generate_svg(self, paths: List[List[Tuple[float, float]]], 
                     metadata: Dict[str, Any]) -> str:
        """
        Generate SVG content from paths.
        
        Args:
            paths: List of path segments
            metadata: Metadata dict from generate_paths
            
        Returns:
            SVG content as string
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
            if not path:
                continue
            
            path_data = f'M {path[0][0]:.2f},{path[0][1]:.2f}'
            for x, y in path[1:]:
                path_data += f' L {x:.2f},{y:.2f}'
            
            svg_lines.append(f'<path d="{path_data}" stroke="black" fill="none" stroke-width="0.5"/>\n')
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def generate_preview(self, paths: List[List[Tuple[float, float]]], 
                        metadata: Dict[str, Any]) -> Image.Image:
        """
        Generate a preview image of the paths.
        
        Args:
            paths: List of path segments
            metadata: Metadata dict from generate_paths
            
        Returns:
            PIL Image object
        """
        total_width_px = metadata['total_width_px']
        total_height_px = metadata['total_height_px']
        
        preview = Image.new('RGB', (total_width_px, total_height_px), 'white')
        draw = ImageDraw.Draw(preview)
        
        for path in paths:
            if len(path) > 1:
                draw.line(path, fill='black', width=1)
        
        return preview


# Factory function for getting algorithm instances
def get_algorithm(algorithm_name: str, config: Dict[str, Any] = None):
    """
    Factory function to get plot generation algorithm instances.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., 'waves')
        config: Optional configuration dict
        
    Returns:
        Algorithm instance
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    algorithms = {
        'waves': WavesGenerator,
        'circles': CirclesGenerator,
        'fabric': FabricGenerator
    }
    
    if algorithm_name.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return algorithms[algorithm_name.lower()](config)


def list_algorithms() -> List[Dict[str, str]]:
    """
    List all available plot generation algorithms.
    
    Returns:
        List of dicts with 'name' and 'description' keys
    """
    return [
        {
            'name': 'waves',
            'description': 'Wave-based plot generation using sine wave modulation'
        },
        {
            'name': 'circles',
            'description': 'Concentric circles with squiggles based on image brightness'
        },
        {
            'name': 'fabric',
            'description': 'Procedurally generated fabric texture using noise or ripples'
        }
    ]