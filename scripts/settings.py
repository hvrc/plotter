"""
settings.py - Centralized settings management

This module handles all application settings including:
- Plotter hardware configuration
- Algorithm parameters
- Application preferences
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class SettingsManager:
    """
    Manages all application settings with automatic persistence.
    
    Settings are stored in settings/settings.json
    Default settings live in settings/default.json
    """

    def __init__(
        self,
        settings_path: str = "settings/settings.json",
        defaults_path: Optional[str] = None,
    ):
        """
        Initialize settings manager.
        
        Args:
            settings_path: Path to settings file
            defaults_path: Path to defaults file (default: settings/default.json)
        """
        self.settings_path = Path(settings_path)
        self.defaults_path = Path(defaults_path) if defaults_path else (self.settings_path.parent / "default.json")
        self.defaults = self._load_defaults()
        self.settings = self._deep_copy(self.defaults)
        
        # Ensure settings directory exists
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.defaults_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing settings
        self.load()

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as e:
            print(f"Warning: Could not read JSON from {path}: {e}")
            return None

    def _write_json(self, path: Path, data: Dict[str, Any]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error writing JSON to {path}: {e}")
            return False

    def _generate_defaults(self) -> Dict[str, Any]:
        """Generate a reasonable defaults file from algorithm/plotter defaults."""
        defaults: Dict[str, Any] = {
            'plotter': {},
            'algorithm': {
                'current_algorithm': 'waves'
            }
        }

        try:
            from plotter import PlotterController

            defaults['plotter'] = PlotterController.DEFAULT_CONFIG.copy()
        except Exception:
            defaults['plotter'] = {}

        try:
            from algorithms import get_algorithm_names, get_algorithm

            algo_names = get_algorithm_names()
            available = [name.lower() for name in algo_names]
            if available:
                defaults['algorithm']['current_algorithm'] = 'waves' if 'waves' in available else available[0]

            for name in algo_names:
                try:
                    algo = get_algorithm(name)
                    defaults['algorithm'][name] = getattr(algo, 'DEFAULT_CONFIG', algo.get_config())
                except Exception:
                    continue
        except Exception:
            pass

        return defaults

    def _load_defaults(self) -> Dict[str, Any]:
        defaults = self._read_json(self.defaults_path)
        if defaults:
            return defaults

        generated = self._generate_defaults()
        self._write_json(self.defaults_path, generated)
        return generated
    
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
        loaded = self._read_json(self.settings_path)
        if loaded:
            # Merge loaded settings with defaults (in case new settings were added)
            self._merge_settings(self.settings, loaded)
            print(f"Settings loaded from {self.settings_path}")
        else:
            print(f"No settings file found. Creating settings at {self.settings_path}")
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
        return self._write_json(self.settings_path, self.settings)
    
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
        self.defaults = self._load_defaults()
        self.settings = self._deep_copy(self.defaults)
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
            with open(export_path, 'w', encoding='utf-8') as f:
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
            with open(import_path, 'r', encoding='utf-8') as f:
                imported = json.load(f)
            self.defaults = self._load_defaults()
            self.settings = self._deep_copy(self.defaults)
            if isinstance(imported, dict):
                self._merge_settings(self.settings, imported)
                if auto_save:
                    self.save()
                print(f"Settings imported from {import_path}")
                return True
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False
