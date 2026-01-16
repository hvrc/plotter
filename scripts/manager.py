import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

class DatabaseManager:
    def __init__(self, base_path: str = "database"):
        """
        Initialize the database manager.
        
        Args:
            base_path: Base directory for database (default: 'database')
        """
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.plots_path = self.base_path / "plots"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create database directories if they don't exist."""
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)

    def _iter_files(self, directory: Path, extensions: Optional[Iterable[str]] = None) -> List[Path]:
        """Return files in directory optionally filtered by extension (case-insensitive)."""
        if not directory.exists():
            return []

        normalized_exts = None
        if extensions is not None:
            normalized_exts = {
                (ext if ext.startswith('.') else f'.{ext}').lower()
                for ext in extensions
            }

        results: List[Path] = []
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if normalized_exts is not None and path.suffix.lower() not in normalized_exts:
                continue
            results.append(path)

        return results
    
    # ==========================================
    # IMAGE OPERATIONS
    # ==========================================
    
    def list_images(self, extensions: List[str] = None) -> List[str]:
        """
        List all image files in the images directory.
        
        Args:
            extensions: List of extensions to filter (default: common image formats)
            
        Returns:
            Sorted list of image filenames
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

        image_files = self._iter_files(self.images_path, extensions)
        return sorted(p.name for p in image_files)
    
    def get_image_path(self, filename: str) -> str:
        """
        Get full path to an image file.
        
        Args:
            filename: Image filename
            
        Returns:
            Full path to image
        """
        return str(self.images_path / filename)
    
    def image_exists(self, filename: str) -> bool:
        """Check if an image file exists."""
        return (self.images_path / filename).exists()
    
    def add_image(self, source_path: str, destination_name: Optional[str] = None) -> str:
        """
        Copy an image into the database.
        
        Args:
            source_path: Path to source image
            destination_name: Optional name for destination (default: use source name)
            
        Returns:
            Destination filename
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        
        dest_name = destination_name or source.name
        dest_path = self.images_path / dest_name
        
        shutil.copy2(source, dest_path)
        return dest_name
    
    # ==========================================
    # PLOT OPERATIONS
    # ==========================================
    
    def list_plots(self) -> List[str]:
        """
        List all plot directories.
        
        Returns:
            Sorted list of plot directory names
        """
        if not self.plots_path.exists():
            return []

        plot_dirs = [p.name for p in self.plots_path.iterdir() if p.is_dir()]
        return sorted(plot_dirs)
    
    def get_plot_path(self, plot_name: str) -> str:
        """
        Get full path to a plot directory.
        
        Args:
            plot_name: Plot directory name
            
        Returns:
            Full path to plot directory
        """
        return str(self.plots_path / plot_name)
    
    def plot_exists(self, plot_name: str) -> bool:
        """Check if a plot directory exists."""
        return (self.plots_path / plot_name).is_dir()
    
    def create_plot_directory(self, plot_name: str) -> str:
        """
        Create a new plot directory.
        
        Args:
            plot_name: Name for the plot directory
            
        Returns:
            Full path to created directory
        """
        plot_dir = self.plots_path / plot_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        return str(plot_dir)
    
    def get_plot_info(self, plot_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plot.
        
        Args:
            plot_name: Plot directory name
            
        Returns:
            Dict with plot information or None if not found
        """
        plot_dir = self.plots_path / plot_name
        if not plot_dir.is_dir():
            return None
        
        info: Dict[str, Any] = {
            'name': plot_name,
            'path': str(plot_dir),
            'files': []
        }

        for file in self._iter_files(plot_dir):
            info['files'].append({
                'name': file.name,
                'type': file.suffix.lower(),
                'path': str(file)
            })
        
        return info
    
    def get_plot_svg(self, plot_name: str) -> Optional[str]:
        """
        Get the SVG file path for a plot.
        
        Args:
            plot_name: Plot directory name
            
        Returns:
            Path to SVG file or None if not found
        """
        plot_dir = self.plots_path / plot_name
        if not plot_dir.is_dir():
            return None
        
        svg_files = self.get_plot_svgs(plot_name)
        return svg_files[0] if svg_files else None
    
    def get_plot_svgs(self, plot_name: str) -> List[str]:
        """
        Get all SVG file paths for a plot.
        
        Args:
            plot_name: Plot directory name
            
        Returns:
            List of paths to SVG files (empty list if none found)
        """
        plot_dir = self.plots_path / plot_name
        if not plot_dir.is_dir():
            return []
        
        svg_paths = self._iter_files(plot_dir, extensions=['.svg'])
        return [str(p) for p in sorted(svg_paths, key=lambda p: p.name.lower())]
    
    def delete_plot(self, plot_name: str) -> bool:
        """
        Delete a plot directory and all its contents.
        
        Args:
            plot_name: Plot directory name
            
        Returns:
            True if deleted, False if not found
        """
        plot_dir = self.plots_path / plot_name
        if not plot_dir.is_dir():
            return False
        
        shutil.rmtree(plot_dir)
        return True
    
    # ==========================================
    # PLOT CREATION HELPER
    # ==========================================
    
    def save_plot(self, plot_name: str, svg_content, 
                  preview_image=None, original_image_path: str = None,
                  metadata: Dict = None) -> Dict[str, str]:
        """
        Save a complete plot with SVG, preview, and optional original image.
        
        Args:
            plot_name: Name for the plot
            svg_content: SVG content as string OR dict of {color: svg_content}
            preview_image: PIL Image object for preview (optional)
            original_image_path: Path to original image to copy (optional)
            metadata: Optional metadata dict to save as JSON
            
        Returns:
            Dict with paths to saved files
        """
        # Create plot directory
        plot_dir = self.create_plot_directory(plot_name)
        plot_path = Path(plot_dir)
        
        saved_files = {}
        
        # Save SVG(s)
        if isinstance(svg_content, dict):
            # Check if it's fabric with horizontal/vertical directions
            if 'horizontal' in svg_content or 'vertical' in svg_content:
                # Fabric directions mode: save one SVG per direction
                for direction, svg in svg_content.items():
                    svg_path = plot_path / f"{plot_name}_{direction}.svg"
                    with open(svg_path, 'w', encoding='utf-8') as f:
                        f.write(svg)
                    saved_files[f'svg_{direction}'] = str(svg_path)
            else:
                # Multi-color mode: save one SVG per color
                for i, (color, svg) in enumerate(svg_content.items()):
                    color_hex = f"{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    svg_path = plot_path / f"{plot_name}_color{i+1}_{color_hex}.svg"
                    with open(svg_path, 'w', encoding='utf-8') as f:
                        f.write(svg)
                    saved_files[f'svg_color{i+1}'] = str(svg_path)
        else:
            # Single color mode
            svg_path = plot_path / f"{plot_name}.svg"
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            saved_files['svg'] = str(svg_path)
        
        # Save preview if provided
        if preview_image:
            preview_path = plot_path / f"{plot_name}_preview.png"
            preview_image.save(preview_path)
            saved_files['preview'] = str(preview_path)
        
        # Copy original image if provided
        if original_image_path:
            original_name = Path(original_image_path).name
            original_dest = plot_path / original_name
            shutil.copy2(original_image_path, original_dest)
            saved_files['original'] = str(original_dest)
        
        # Save metadata if provided
        if metadata:
            import json
            metadata_path = plot_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_path)
        
        return saved_files
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dict with counts and sizes
        """
        return {
            'images_count': len(self.list_images()),
            'plots_count': len(self.list_plots()),
            'images_path': str(self.images_path),
            'plots_path': str(self.plots_path)
        }
    
    def cleanup_empty_plots(self) -> int:
        """
        Remove empty plot directories.
        
        Returns:
            Number of directories removed
        """
        count = 0
        for plot_dir in self.plots_path.iterdir():
            if plot_dir.is_dir() and not any(plot_dir.iterdir()):
                shutil.rmtree(plot_dir)
                count += 1
        return count

    def merge_svgs(self, plot_name: str, input_filenames: List[str], output_filename: str) -> bool:
        """
        Merge multiple SVG files into a single SVG file.
        
        Args:
            plot_name: Name of the plot directory
            input_filenames: List of filenames to merge
            output_filename: Name of the output file
            
        Returns:
            True if successful, False otherwise
        """
        if not input_filenames:
            return False
            
        plot_dir = self.plots_path / plot_name
        files = [plot_dir / f for f in input_filenames]
        
        # Check all exist
        if not all(f.exists() for f in files):
            return False
            
        try:
            # Register namespace to ensure 'svg' tag doesn't get strict namespace prefix
            svg_ns = "http://www.w3.org/2000/svg"
            ET.register_namespace('', svg_ns)
            
            # Parse first file to use as base
            tree = ET.parse(files[0])
            root = tree.getroot()
            
            # Append contents of other files
            for f_path in files[1:]:
                f_tree = ET.parse(f_path)
                f_root = f_tree.getroot()
                # Append all children of the root (groups, paths, defs)
                for child in f_root:
                    root.append(child)
            
            output_path = plot_dir / output_filename
            tree.write(output_path, encoding='unicode', xml_declaration=False)
            return True
        except Exception as e:
            print(f"Error merging SVGs: {e}")
            return False