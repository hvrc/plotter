"""
settings.py - Centralized settings management

This module handles all application settings including:
- Plotter hardware configuration
- Algorithm parameters
- Application preferences
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class SettingsManager:
    """
    Manages all application settings with automatic persistence.
    
    Settings are stored in settings/settings.json
    """
    
    # Default settings for all components
    DEFAULT_SETTINGS = {
        'plotter': {
            'pen_pos_up': 100,
            'pen_pos_down': 30,
            'auto_rotate': False,
            'speed_pendown': 25,
            'speed_penup': 75,
            'accel': 75
        },
        'algorithm': {
            'current_algorithm': 'waves',
            'waves': {
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
                'draw_direction': 'horizontal',
                'numColors': 1
            },
            'circles': {
                'output_width_inches': 4.5,
                'output_height_inches': 6.0,
                'margin_inches': 0.125,
                'n_circles': 80,
                'contrast_power': 1.5,
                'amplitude_scale': 1,
                'frequency_scale': 2.0,
                'white_threshold': 250,
                'calc_dpi': 300,
                'points_per_circle': 360
            },
            'fabric': {
                'output_width_inches': 4.5,
                'output_height_inches': 6.0,
                'margin_inches': 0.125,
                'calc_dpi': 300,
                'grid_cols': 150,
                'grid_rows': 200,
                'displacement_mode': 'random',
                'noise_scale_x': 0.002,
                'noise_scale_y': 0.002,
                'noise_octaves': 4,
                'noise_persistence': 0.5,
                'noise_lacunarity': 6.0,
                'displacement_magnitude': 500,
                'ripple_frequency': 0.015,
                'ripple_amplitude': 400,
                'ripple_centers': 1,
                'ripple_centers_locations': [],
                'enable_weave': True,
                'line_direction': 'horizontal'
            },
            'fish': {
                'output_width_inches': 4.5,
                'output_height_inches': 6.0,
                'margin_inches': 0.125,
                'calc_dpi': 300,
                'radius_fraction': 0.9,
                'points_per_circle': 720,
                'stroke_width': 0.5
            },
            'sphere': {
                'output_width_inches': 6.0,
                'output_height_inches': 6.0,
                'margin_inches': 0.5,
                'calc_dpi': 300,
                'sphere_radius': 0.8,
                'rotation_x': 20,
                'rotation_y': 20,
                'pattern': 'flow',
                'num_latitude_lines': 12,
                'num_longitude_lines': 16,
                'num_flow_lines': 20,
                'spiral_revolutions': 8,
                'points_per_curve': 200,
                'stroke_width': 0.5,
                'show_hidden_lines': False,
                'num_pole_circles': 5,
                'pole_circle_spacing': 10
            },
            'terrain': {
                'output_width_inches': 6.0,
                'output_height_inches': 6.0,
                'margin_inches': 0.5,
                'calc_dpi': 300,
                'grid_resolution_x': 60,
                'grid_resolution_z': 60,
                'cube_height': 0.3,
                'terrain_amplitude': 1.0,
                'noise_scale': 0.08,
                'noise_octaves': 4,
                'elevation_mode': 'perlin',
                'perspective_strength': 0.6,
                'rotation_x': 25,
                'rotation_z': 35,
                'cutout_front': True,
                'cutout_side': True,
                'line_density': 1.0,
                'smooth_terrain': True,
                'terrain_center_x': 0.5,
                'terrain_center_z': 0.5
            }
        }
    }
    
    def __init__(self, settings_path: str = "settings/settings.json"):
        """
        Initialize settings manager.
        
        Args:
            settings_path: Path to settings file
        """
        self.settings_path = Path(settings_path)
        self.settings = self._deep_copy(self.DEFAULT_SETTINGS)
        
        # Ensure settings directory exists
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing settings
        self.load()
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a nested dict structure."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def load(self) -> Dict[str, Any]:
        """
        Load settings from file.
        
        Returns:
            Loaded settings dict
        """
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'r') as f:
                    loaded = json.load(f)
                    # Merge loaded settings with defaults (in case new settings were added)
                    self._merge_settings(self.settings, loaded)
                    print(f"Settings loaded from {self.settings_path}")
            except Exception as e:
                print(f"Warning: Could not load settings from {self.settings_path}: {e}")
                print("Using default settings")
        else:
            print(f"No settings file found. Creating default settings at {self.settings_path}")
            self.save()

        # Normalize/validate algorithm selection (handles typos like 'cirlces').
        self._validate_current_algorithm(auto_save=True)
        
        return self._deep_copy(self.settings)

    def _validate_current_algorithm(self, auto_save: bool = True):
        """Ensure current_algorithm is valid; fix common typos and fall back safely."""
        algo_settings = self.settings.get('algorithm', {})
        current = str(algo_settings.get('current_algorithm', '') or '').strip().lower()

        # Common aliases/typos
        aliases = {
            'cirlces': 'circles',
            'circle': 'circles',
            'spirals': 'spiral',
        }
        if current in aliases:
            current = aliases[current]

        try:
            from algorithms import get_algorithm_names

            available = set(name.lower() for name in get_algorithm_names())
            if current not in available:
                # Prefer waves if available; otherwise choose first available.
                if 'waves' in available:
                    fixed = 'waves'
                elif 'circles' in available:
                    fixed = 'circles'
                else:
                    fixed = sorted(available)[0] if available else 'waves'

                if fixed != current:
                    print(f"Warning: Invalid current algorithm '{algo_settings.get('current_algorithm')}'. Using '{fixed}'.")
                    self.settings['algorithm']['current_algorithm'] = fixed
                    if auto_save:
                        self.save()
            else:
                # Write back normalized casing if necessary
                if algo_settings.get('current_algorithm') != current:
                    self.settings['algorithm']['current_algorithm'] = current
                    if auto_save:
                        self.save()
        except Exception:
            # If the algorithms package can't be imported for any reason,
            # keep the setting as-is to avoid breaking startup.
            return
    
    def _merge_settings(self, default: Dict, loaded: Dict):
        """Recursively merge loaded settings into defaults."""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_settings(default[key], value)
            else:
                default[key] = value
    
    def save(self) -> bool:
        """
        Save current settings to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings to {self.settings_path}: {e}")
            return False
    
    # ==========================================
    # PLOTTER SETTINGS
    # ==========================================
    
    def get_plotter_settings(self) -> Dict[str, Any]:
        """Get all plotter settings."""
        return self._deep_copy(self.settings['plotter'])
    
    def set_plotter_setting(self, key: str, value: Any, auto_save: bool = True):
        """
        Set a plotter setting.
        
        Args:
            key: Setting key
            value: Setting value
            auto_save: Automatically save to file (default: True)
        """
        self.settings['plotter'][key] = value
        if auto_save:
            self.save()
    
    def update_plotter_settings(self, settings: Dict[str, Any], auto_save: bool = True):
        """
        Update multiple plotter settings.
        
        Args:
            settings: Dict of settings to update
            auto_save: Automatically save to file (default: True)
        """
        self.settings['plotter'].update(settings)
        if auto_save:
            self.save()
    
    # ==========================================
    # ALGORITHM SETTINGS
    # ==========================================
    
    def get_current_algorithm(self) -> str:
        """Get the name of the current algorithm."""
        return self.settings['algorithm']['current_algorithm']
    
    def set_current_algorithm(self, algorithm_name: str, auto_save: bool = True):
        """
        Set the current algorithm.
        
        Args:
            algorithm_name: Name of algorithm to use
            auto_save: Automatically save to file (default: True)
        """
        self.settings['algorithm']['current_algorithm'] = algorithm_name
        if auto_save:
            self.save()
    
    def get_algorithm_settings(self, algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get settings for a specific algorithm.
        
        Args:
            algorithm_name: Algorithm name (default: current algorithm)
            
        Returns:
            Algorithm settings dict
        """
        if algorithm_name is None:
            algorithm_name = self.get_current_algorithm()
        
        if algorithm_name not in self.settings['algorithm']:
            print(f"Warning: No settings found for algorithm '{algorithm_name}'")
            return {}
        
        return self._deep_copy(self.settings['algorithm'][algorithm_name])
    
    def set_algorithm_setting(self, key: str, value: Any, 
                             algorithm_name: Optional[str] = None,
                             auto_save: bool = True):
        """
        Set a setting for a specific algorithm.
        
        Args:
            key: Setting key
            value: Setting value
            algorithm_name: Algorithm name (default: current algorithm)
            auto_save: Automatically save to file (default: True)
        """
        if algorithm_name is None:
            algorithm_name = self.get_current_algorithm()
        
        if algorithm_name not in self.settings['algorithm']:
            self.settings['algorithm'][algorithm_name] = {}
        
        self.settings['algorithm'][algorithm_name][key] = value
        if auto_save:
            self.save()
    
    def update_algorithm_settings(self, settings: Dict[str, Any],
                                  algorithm_name: Optional[str] = None,
                                  auto_save: bool = True):
        """
        Update multiple settings for a specific algorithm.
        
        Args:
            settings: Dict of settings to update
            algorithm_name: Algorithm name (default: current algorithm)
            auto_save: Automatically save to file (default: True)
        """
        if algorithm_name is None:
            algorithm_name = self.get_current_algorithm()
        
        if algorithm_name not in self.settings['algorithm']:
            self.settings['algorithm'][algorithm_name] = {}
        
        self.settings['algorithm'][algorithm_name].update(settings)
        if auto_save:
            self.save()
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._deep_copy(self.settings)
    
    def reset_to_defaults(self, auto_save: bool = True):
        """
        Reset all settings to defaults.
        
        Args:
            auto_save: Automatically save to file (default: True)
        """
        self.settings = self._deep_copy(self.DEFAULT_SETTINGS)
        if auto_save:
            self.save()
    
    def export_settings(self, export_path: str) -> bool:
        """
        Export settings to a different file.
        
        Args:
            export_path: Path to export to
            
        Returns:
            True if successful
        """
        try:
            with open(export_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"Settings exported to {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, import_path: str, auto_save: bool = True) -> bool:
        """
        Import settings from a different file.
        
        Args:
            import_path: Path to import from
            auto_save: Automatically save after import
            
        Returns:
            True if successful
        """
        try:
            with open(import_path, 'r') as f:
                imported = json.load(f)
                self.settings = self._deep_copy(self.DEFAULT_SETTINGS)
                self._merge_settings(self.settings, imported)
                if auto_save:
                    self.save()
                print(f"Settings imported from {import_path}")
                return True
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False
