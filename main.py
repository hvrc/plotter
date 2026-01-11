#!/usr/bin/env python3
"""
main.py - Entry point for the plot generator application

This script initializes all components and launches the interface.
"""

import sys
import os

# Ensure 'scripts' directory is importable (algorithms is a package)
project_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(project_dir, "scripts")

if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from manager import DatabaseManager
from plotter import PlotterController
from interface import CommandLineInterface
from settings import SettingsManager


def main():
    """
    Initialize and run the application.
    """
    print("Initializing Plot Generator...")
    
    # Initialize centralized settings manager
    settings_manager = SettingsManager(settings_path="settings/settings.json")
    
    # Initialize database manager
    db_manager = DatabaseManager(base_path="database")
    
    # Initialize plotter controller with saved settings
    plotter = PlotterController(config=settings_manager.get_plotter_settings())
    
    # Check plotter availability
    if not plotter.is_available():
        print("\nWarning: AxiDraw library (pyaxidraw) not found.")
        print("Plotter control features will be disabled.")
        print("To enable plotter features, install with: pip install pyaxidraw\n")
    
    # Initialize command-line interface
    cli = CommandLineInterface(
        db_manager=db_manager,
        plotter=plotter,
        settings_manager=settings_manager
    )
    
    # Run the interface
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()