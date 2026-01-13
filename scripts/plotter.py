"""
plotter.py - AxiDraw plotter control and configuration

This module handles all AxiDraw hardware interactions including:
- Plotting SVG files
- Manual plotter control commands
- Hardware configuration and settings
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Try to import AxiDraw
try:
    from pyaxidraw import axidraw
    AXIDRAW_AVAILABLE = True
except ImportError:
    AXIDRAW_AVAILABLE = False


class PlotterController:
    """
    Controller for AxiDraw plotter hardware.
    
    Handles plotting operations and manual control commands.
    """
    
    # Default hardware configuration
    DEFAULT_CONFIG = {
        'pen_pos_up': 100,
        'pen_pos_down': 30,
        'auto_rotate': False,
        'speed_pendown': 25,
        'speed_penup': 75,
        'accel': 75
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize plotter controller.
        
        Args:
            config: Optional hardware configuration override
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        if not AXIDRAW_AVAILABLE:
            print("Warning: pyaxidraw not available. Plotter functions disabled.")
    
    def is_available(self) -> bool:
        """Check if AxiDraw library is available."""
        return AXIDRAW_AVAILABLE
    
    def get_config(self) -> Dict[str, Any]:
        """Get current plotter configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Update plotter configuration."""
        self.config.update(config)
    
    def _configure_axidraw(self, ad) -> object:
        """
        Apply configuration to an AxiDraw instance.
        
        Args:
            ad: AxiDraw instance
            
        Returns:
            Configured AxiDraw instance
        """
        ad.options.pen_pos_up = self.config.get('pen_pos_up', self.DEFAULT_CONFIG['pen_pos_up'])
        ad.options.pen_pos_down = self.config.get('pen_pos_down', self.DEFAULT_CONFIG['pen_pos_down'])
        ad.options.auto_rotate = self.config.get('auto_rotate', self.DEFAULT_CONFIG['auto_rotate'])

        # Optional speed settings
        if 'speed_pendown' in self.config:
            ad.options.speed_pendown = self.config['speed_pendown']
        if 'speed_penup' in self.config:
            ad.options.speed_penup = self.config['speed_penup']
        if 'accel' in self.config:
            ad.options.accel = self.config['accel']
        
        return ad

    def _run_axidraw(
        self,
        *,
        mode: str,
        svg_path: Optional[str] = None,
        manual_cmd: Optional[str] = None,
        init_message: str,
        run_message: str,
    ) -> bool:
        if not AXIDRAW_AVAILABLE:
            print("Error: pyaxidraw not available")
            return False

        try:
            print(f"\n{init_message}...")
            ad = axidraw.AxiDraw()

            if svg_path is None:
                ad.plot_setup()
            else:
                ad.plot_setup(svg_path)

            ad = self._configure_axidraw(ad)

            ad.options.mode = mode
            if manual_cmd is not None:
                ad.options.manual_cmd = manual_cmd

            print(run_message)
            ad.plot_run()
            print("\nOperation complete")
            return True
        except Exception as e:
            print(f"Error during plotting: {e}")
            return False
    
    def plot_file(self, svg_path: str, preview: bool = False) -> bool:
        """
        Plot an SVG file.
        
        Args:
            svg_path: Path to SVG file
            preview: If True, run in preview mode (no actual plotting)
            
        Returns:
            True if successful, False otherwise
        """
        return self._run_axidraw(
            mode="preview" if preview else "plot",
            svg_path=svg_path,
            init_message=f"Initializing AxiDraw for {'preview' if preview else 'plotting'}",
            run_message="Running preview..." if preview else "Starting plot...",
        )
    
    def execute_command(self, command: str) -> bool:
        """
        Execute a manual plotter command.
        
        Available commands:
        - 'disable_xy': Turn off stepper motors
        - 'walk_home': Return plotter to home position
        - 'raise_pen': Raise the pen
        - 'lower_pen': Lower the pen
        
        Args:
            command: Command name
            
        Returns:
            True if successful, False otherwise
        """
        valid_commands = ['disable_xy', 'walk_home', 'raise_pen', 'lower_pen']
        if command not in valid_commands:
            print(f"Error: Invalid command '{command}'")
            print(f"Valid commands: {', '.join(valid_commands)}")
            return False

        ok = self._run_axidraw(
            mode="manual",
            svg_path=None,
            manual_cmd=command,
            init_message=f"Executing command: {command}",
            run_message="Running manual command...",
        )
        if ok:
            print("Command complete")
        return ok
    
    def get_available_commands(self) -> Dict[str, str]:
        """
        Get list of available manual commands with descriptions.
        
        Returns:
            Dict mapping command names to descriptions
        """
        return {
            'disable_xy': 'Turn off stepper motors (allows manual movement)',
            'walk_home': 'Return plotter to home position (0, 0)',
            'raise_pen': 'Raise the pen to up position',
            'lower_pen': 'Lower the pen to down position'
        }
    
    def test_connection(self) -> bool:
        """
        Test connection to AxiDraw plotter.
        
        Returns:
            True if plotter is connected, False otherwise
        """
        if not AXIDRAW_AVAILABLE:
            return False
        
        try:
            ad = axidraw.AxiDraw()
            ad.interactive()
            connected = ad.connect()
            if connected:
                ad.disconnect()
            return connected
        except Exception:
            return False


class PlotterSettings:
    """
    Manages persistent plotter settings.
    
    This class handles saving and loading plotter configuration
    to enable settings persistence between sessions.
    """
    
    def __init__(self, settings_file: str = "plotter_settings.json"):
        """
        Initialize settings manager.
        
        Args:
            settings_file: Path to settings file
        """
        self.settings_file = settings_file
        self.settings = PlotterController.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load settings from file.
        
        Returns:
            Loaded settings dict
        """
        path = Path(self.settings_file)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.settings.update(loaded)
            except Exception as e:
                print(f"Warning: Could not load settings: {e}")

        return self.settings.copy()
    
    def save(self) -> bool:
        """
        Save current settings to file.
        
        Returns:
            True if successful, False otherwise
        """
        path = Path(self.settings_file)
        try:
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get(self, key: str, default=None):
        """Get a setting value."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a setting value."""
        self.settings[key] = value
    
    def update(self, settings: Dict[str, Any]):
        """Update multiple settings."""
        self.settings.update(settings)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings."""
        return self.settings.copy()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = PlotterController.DEFAULT_CONFIG.copy()