from typing import Any, Dict, List, Tuple, Optional
from PIL import Image
from .base import PlotAlgorithm

class PixelsAlgorithm(PlotAlgorithm):
    DEFAULT_CONFIG = {
        'dpcm': 10  # Default to 10 pixels per cm (1mm pen)
    }

    def get_algorithm_name(self) -> str:
        return "pixels"

    def get_algorithm_description(self) -> str:
        return "Direct pixel-to-dot mapping. 1 pixel = 1 dot."

    def is_procedural(self) -> bool:
        return False

    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Generates paths where each black pixel becomes a single point path.
        strictly black and white input expected.
        """
        if not image_path:
            raise ValueError("Pixels algorithm requires an input image.")
        
        # Open the image.
        # User requirement: "No scaling, stretching, resampling".
        # We load strictly.
        with Image.open(image_path) as img:
            # We assume the image is black/white as per requirements.
            # Converting to 'L' to safely check values (0=Black, 255=White typically)
            # or user might provide '1'.
            if img.mode != '1' and img.mode != 'L':
                gray = img.convert('L')
            else:
                gray = img
            
            width, height = gray.size
            pixels = gray.load()
            
            paths = []
            
            # Iterate over all pixels
            for y in range(height):
                for x in range(width):
                    # Get pixel value
                    # In 'L' or '1', 0 is usually black.
                    val = pixels[x, y]
                    
                    # Assuming strictly black and white
                    # Treat low values (black) as pen-down
                    if val < 128:
                        # Create a point path
                        # Using (x, y) and (x, y) creates a zero-length path
                        paths.append([(x, y), (x, y)])

            metadata = {
                'width_px': width,
                'height_px': height,
                'dpcm': self.config.get('dpcm', 10),
                'count': len(paths)
            }
            
            return paths, metadata

    def generate_svg(self, paths: Any, metadata: Dict[str, Any]) -> Any:
        width_px = metadata['width_px']
        height_px = metadata['height_px']
        dpcm = metadata['dpcm']
        
        # Calculate physical dimensions in centimeters
        width_cm = width_px / dpcm
        height_cm = height_px / dpcm
        
        # Create SVG header
        # Using physical units for width/height, but pixels for viewBox
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" ',
            f'width="{width_cm:.2f}cm" height="{height_cm:.2f}cm" ',
            f'viewBox="0 0 {width_px} {height_px}">'
        ]
        
        # Add paths
        for path in paths:
            if not path:
                continue
            
            x, y = path[0]
            # Create a dot using a zero-length line with round caps
            # This ensures a "pen-down" action at the exact location
            svg_lines.append(
                f'<path d="M {x} {y} L {x} {y}" '
                f'stroke="black" stroke-width="1" stroke-linecap="round" fill="none" />'
            )
            
        svg_lines.append('</svg>')
        return "".join(svg_lines)

    def generate_preview(self, paths: Any, metadata: Dict[str, Any]) -> Image.Image:
        width = metadata['width_px']
        height = metadata['height_px']
        
        # Create a new white image
        img = Image.new('L', (width, height), 255)
        pixels = img.load()
        
        # Draw the dots
        for path in paths:
            if not path:
                continue
            x, y = path[0]
            # Ensure coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = 0  # Set to black
                
        return img
