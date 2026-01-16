"""algorithms.waves

Wave-based plot generation algorithm.

Converts images into plottable SVG paths using sine wave modulation.
Brightness controls wave amplitude and frequency.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageOps

from .base import PlotAlgorithm


class WavesGenerator(PlotAlgorithm):
    """Generates wave-based plots from images."""

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
        'numColors': 1,  # Number of dominant colors (1 = single black layer)
        'color_mask': []
    }
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "waves"
    
    def get_algorithm_description(self) -> str:
        """Return a description of this algorithm."""
        return "Wave-based plot generation using sine wave modulation"

    def is_procedural(self) -> bool:
        return False

    @staticmethod
    def _compute_canvas_px(
        *,
        output_width_inches: float,
        output_height_inches: float,
        margin_inches: float,
        dpi: float,
    ) -> Tuple[int, int, int, int, int]:
        total_width_px = int(output_width_inches * dpi)
        total_height_px = int(output_height_inches * dpi)
        margin_px = int(margin_inches * dpi)

        drawable_width_px = total_width_px - (2 * margin_px)
        drawable_height_px = total_height_px - (2 * margin_px)
        if drawable_width_px <= 0 or drawable_height_px <= 0:
            raise ValueError("Margins are larger than paper size")

        return total_width_px, total_height_px, margin_px, drawable_width_px, drawable_height_px

    @staticmethod
    def _apply_serpentine(segments: List[List[Tuple[float, float]]], index: int, use_serpentine: bool) -> None:
        if use_serpentine and (index % 2 == 1):
            segments.reverse()
            for seg in segments:
                seg.reverse()

    @staticmethod
    def _path_to_svg_d(path: List[Tuple[float, float]]) -> str:
        return f"M {path[0][0]:.2f} {path[0][1]:.2f} " + ''.join(
            f"L {x:.2f} {y:.2f} " for x, y in path[1:]
        )
    
    def _quantize_colors(self, image: Image.Image, num_colors: int) -> Tuple[Image.Image, List[Tuple[int, int, int]]]:
        """
        Reduce image to N dominant colors using k-means clustering.
        
        Args:
            image: RGB PIL Image
            num_colors: Number of dominant colors to extract
            
        Returns:
            Tuple of (quantized_image, list_of_dominant_colors)
        """
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
        # Pure-Python implementation (avoids hard dependency on numpy).
        # This is slower than numpy but keeps the module importable so the
        # algorithm can be discovered/registered even in minimal environments.
        if tolerance < 0:
            tolerance = 0

        tr, tg, tb = (int(target_color[0]), int(target_color[1]), int(target_color[2]))
        tol2 = int(tolerance) * int(tolerance)

        rgb = image.convert('RGB')
        mask = Image.new('L', rgb.size, 0)

        out = []
        for r, g, b in rgb.getdata():
            dr = int(r) - tr
            dg = int(g) - tg
            db = int(b) - tb
            dist2 = dr * dr + dg * dg + db * db
            out.append(255 if dist2 <= tol2 else 0)

        mask.putdata(out)
        return mask
    
    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate plot paths from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (paths, metadata)
            - paths: List of path segments OR dict of {color: paths} if numColors > 1
            - metadata: Dict with dimensions and other info for SVG generation
        """
        if not image_path:
            raise ValueError("image_path is required")
        try:
            original_img = Image.open(image_path)
        except IOError as e:
            raise ValueError(f"Could not open image: {image_path}") from e

        config = self.config
        dpi = float(config['calc_dpi'])
        output_width_inches = float(config['output_width_inches'])
        output_height_inches = float(config['output_height_inches'])
        margin_inches = float(config['margin_inches'])
        n_rows = int(config['n_rows'])
        contrast_power = float(config['contrast_power'])
        amplitude_scale = float(config['amplitude_scale'])
        frequency_scale = float(config['frequency_scale'])
        white_threshold = int(config['white_threshold'])
        use_serpentine = bool(config['use_serpentine'])
        draw_direction = str(config.get('draw_direction', 'horizontal') or 'horizontal').lower()
        num_colors = int(config.get('numColors', 1) or 1)
        color_mask = config.get('color_mask', [])

        total_width_px, total_height_px, margin_px, drawable_width_px, drawable_height_px = self._compute_canvas_px(
            output_width_inches=output_width_inches,
            output_height_inches=output_height_inches,
            margin_inches=margin_inches,
            dpi=dpi,
        )
        
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
                # Check execution mask
                if color_mask and i < len(color_mask) and not int(color_mask[i]):
                    continue

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
    
    def generate_svg(self, paths: Any, metadata: Dict[str, Any]) -> Any:
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
            return {
                color: self._generate_single_svg(color_paths, metadata, color)
                for color, color_paths in paths.items()
            }
        else:
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
            d_str = self._path_to_svg_d(path)
            svg_lines.append(
                f'<path d="{d_str}" fill="none" stroke="{color_hex}" stroke-width="2" />\n'
            )
        
        svg_lines.append('</svg>')
        return ''.join(svg_lines)
    
    def _generate_horizontal_paths(
        self,
        processed_img: Image.Image,
        n_rows: int,
        drawable_width_px: int,
        drawable_height_px: int,
        margin_px: int,
        contrast_power: float,
        amplitude_scale: float,
        frequency_scale: float,
        white_threshold: int,
        use_serpentine: bool,
    ) -> List[List[Tuple[float, float]]]:
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

            self._apply_serpentine(row_segments, r, use_serpentine)
            
            all_paths.extend(row_segments)
        
        return all_paths
    
    def _generate_vertical_paths(
        self,
        processed_img: Image.Image,
        n_rows: int,
        drawable_width_px: int,
        drawable_height_px: int,
        margin_px: int,
        contrast_power: float,
        amplitude_scale: float,
        frequency_scale: float,
        white_threshold: int,
        use_serpentine: bool,
    ) -> List[List[Tuple[float, float]]]:
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

            self._apply_serpentine(col_segments, c, use_serpentine)
            
            all_paths.extend(col_segments)
        
        return all_paths
    
    def generate_preview(self, paths: Any, metadata: Dict[str, Any]) -> Image.Image:
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
