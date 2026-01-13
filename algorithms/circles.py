"""algorithms.circles

Concentric circles plot generation.

Generates concentric circles with squiggles based on image brightness.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps, ImageDraw

from .base import PlotAlgorithm


class CirclesGenerator(PlotAlgorithm):
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
        'points_per_circle': 360,  # Number of sample points per circle
    }
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "circles"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Concentric circles with squiggles based on image brightness"

    def _compute_canvas(self) -> tuple[int, int, int, int, int]:
        calc_dpi = self.config['calc_dpi']
        total_width_px = int(self.config['output_width_inches'] * calc_dpi)
        total_height_px = int(self.config['output_height_inches'] * calc_dpi)
        margin_px = int(self.config['margin_inches'] * calc_dpi)
        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        return total_width_px, total_height_px, margin_px, drawable_width_px, drawable_height_px
    
    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
        """
        Generate plot paths from an image using concentric circles.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (paths, metadata)
        """
        if not image_path:
            raise ValueError("image_path is required for circles algorithm")

        try:
            original_img = Image.open(image_path)
        except OSError as e:
            raise OSError(f"Cannot open image: {e}")

        total_width_px, total_height_px, margin_px, drawable_width_px, drawable_height_px = self._compute_canvas()
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
            processed_img,
            drawable_width_px=drawable_width_px,
            drawable_height_px=drawable_height_px,
            margin_px=margin_px,
        )

        # Create metadata for SVG generation
        metadata = {
            'total_width_px': total_width_px,
            'total_height_px': total_height_px,
            'output_width_inches': self.config['output_width_inches'],
            'output_height_inches': self.config['output_height_inches'],
            'algorithm': self.get_algorithm_name()
        }
        
        return paths, metadata
    
    def _generate_circular_paths(
        self,
        processed_img: Image.Image,
        *,
        drawable_width_px: int,
        drawable_height_px: int,
        margin_px: int,
    ) -> List[List[Tuple[float, float]]]:
        """Generate concentric circular paths from center outward."""
        pixels = processed_img.load()
        img_width, img_height = processed_img.size

        n_circles = int(self.config['n_circles'])
        contrast_power = float(self.config['contrast_power'])
        amplitude_scale = float(self.config['amplitude_scale'])
        frequency_scale = float(self.config['frequency_scale'])
        white_threshold = int(self.config['white_threshold'])
        points_per_circle = int(self.config['points_per_circle'])
        
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
                img_x = int(base_radius * math.cos(angle) + drawable_width_px / 2)
                img_y = int(base_radius * math.sin(angle) + drawable_height_px / 2)
                
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

        stroke_width = self.config.get('stroke_width', 0.5)
        
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
            
            svg_lines.append(
                f'<path d="{path_data}" stroke="black" fill="none" stroke-width="{stroke_width}"/>\n'
            )
        
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
