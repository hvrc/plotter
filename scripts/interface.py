"""
interface.py - Command line interface

This module provides the CLI for the plotter application.
It's designed to be replaceable with other interface types (GUI, web, etc.)
"""

import sys
from typing import Optional, Dict, Any
from manager import DatabaseManager
from plotter import PlotterController
from settings import SettingsManager
from waves import get_algorithm, list_algorithms


class CommandLineInterface:
    """
    Command-line interface for the plotter application.
    
    This interface is designed to be one of potentially many interface types.
    Other interfaces (GUI, web) could be implemented with the same backend.
    """
    
    def __init__(self, db_manager: DatabaseManager, 
                 plotter: PlotterController,
                 settings_manager: SettingsManager):
        """
        Initialize the CLI.
        
        Args:
            db_manager: Database manager instance
            plotter: Plotter controller instance
            settings_manager: Centralized settings manager
        """
        self.db = db_manager
        self.plotter = plotter
        self.settings = settings_manager
        
        # Load current algorithm settings
        self.algorithm_name = self.settings.get_current_algorithm()
        self.algorithm_config = self.settings.get_algorithm_settings(self.algorithm_name)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Print a section header."""
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}\n")
    
    def get_input(self, prompt: str) -> str:
        """Get user input with prompt."""
        return input(f"{prompt}: ").strip()
    
    def get_number(self, prompt: str) -> Optional[int]:
        """Get numeric input from user."""
        try:
            return int(self.get_input(prompt))
        except ValueError:
            print("\nInvalid number")
            return None
    
    def pause(self):
        """Wait for user to press enter."""
        input("\nPress Enter to continue...")
    
    # ==========================================
    # MAIN MENU
    # ==========================================
    
    def main_menu(self):
        """Display and handle main menu."""
        while True:
            self.print_header("PLOT GENERATOR - MAIN MENU")
            
            # Dynamic menu text based on algorithm
            if self.algorithm_name == 'fabric':
                displacement_mode = self.algorithm_config.get('displacement_mode', 'random')
                if displacement_mode == 'image':
                    print("1. Generate Plot from Image")
                else:
                    print("1. Generate Plot (Procedural)")
            else:
                print("1. Generate Plot from Image")
            
            print("2. Run Existing Plot on Plotter")
            print("3. Manual Plotter Control")
            print("4. Algorithm Settings")
            print("5. Plotter Settings")
            print("6. View Database Info")
            print("7. Exit")
            
            choice = self.get_number("\nSelect option")
            
            if choice == 1:
                self.generate_plot_menu()
            elif choice == 2:
                self.run_plot_menu()
            elif choice == 3:
                self.plotter_control_menu()
            elif choice == 4:
                self.algorithm_settings_menu()
            elif choice == 5:
                self.plotter_settings_menu()
            elif choice == 6:
                self.database_info_menu()
            elif choice == 7:
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print("\nInvalid selection")
                self.pause()
    
    # ==========================================
    # GENERATE PLOT MENU
    # ==========================================
    
    def generate_plot_menu(self):
        """Menu for generating plots."""
        # Check if current algorithm generates from scratch or needs an image
        if self.algorithm_name == 'fabric':
            # Check if fabric is in 'image' displacement mode
            displacement_mode = self.algorithm_config.get('displacement_mode', 'random')
            if displacement_mode == 'image':
                # Fabric in image mode - needs an image input
                self._generate_plot_from_image()
            else:
                # Fabric generates procedurally, no input image needed
                self._generate_plot_procedural()
        else:
            # Image-based algorithms
            self._generate_plot_from_image()
    
    def _generate_plot_from_image(self):
        """Menu for generating plots from images."""
        self.print_header("GENERATE PLOT FROM IMAGE")
        
        images = self.db.list_images()
        
        if not images:
            print("No images found in database/images/")
            print("Please add images to the database/images/ directory")
            self.pause()
            return
        
        print("Available images:")
        for i, img in enumerate(images, 1):
            print(f"{i}. {img}")
        print(f"{len(images) + 1}. Back")
        
        choice = self.get_number("\nSelect image")
        
        if choice is None or choice == len(images) + 1:
            return
        
        if 1 <= choice <= len(images):
            selected_image = images[choice - 1]
            self._generate_plot(selected_image)
        else:
            print("\nInvalid selection")
            self.pause()
    
    def _generate_plot_procedural(self):
        """Generate a procedural plot (no input image required)."""
        import os
        import time
        
        self.print_header("GENERATE PROCEDURAL PLOT")
        
        # Generate name based on algorithm
        # For fabric, use consistent name to replace previous versions
        if self.algorithm_name == 'fabric':
            base_name = 'fabric'
        else:
            # For other algorithms, use timestamp for unique names
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{self.algorithm_name}_{timestamp}"
        
        print(f"Generating plot: {base_name}")
        print(f"Using algorithm: {self.algorithm_name}")
        
        try:
            # Get algorithm instance
            algorithm = get_algorithm(self.algorithm_name, self.algorithm_config)
            
            # Generate paths (no image path needed)
            print("Generating paths...")
            paths, metadata = algorithm.generate_paths()
            
            print(f"Generated {len(paths)} path segments")
            
            # Generate SVG
            print("Creating SVG...")
            svg_content = algorithm.generate_svg(paths, metadata)
            
            # Generate preview
            print("Creating preview...")
            preview_image = algorithm.generate_preview(paths, metadata)
            
            # Save everything
            print("Saving plot...")
            metadata['algorithm'] = self.algorithm_name
            metadata['algorithm_config'] = self.algorithm_config
            
            saved_files = self.db.save_plot(
                plot_name=base_name,
                svg_content=svg_content,
                preview_image=preview_image,
                original_image_path=None,  # No source image for procedural generation
                metadata=metadata
            )
            
            print("\n✓ Plot generated successfully!")
            print(f"  Plot name: {base_name}")
            print(f"  Location: {self.db.get_plot_path(base_name)}")
            print(f"  Files created:")
            for file_type, path in saved_files.items():
                print(f"    - {file_type}: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"\n✗ Error generating plot: {e}")
            import traceback
            traceback.print_exc()
        
        self.pause()
    
    def _generate_plot(self, image_filename: str):
        """Generate a plot from an image."""
        import os
        
        image_path = self.db.get_image_path(image_filename)
        base_name = os.path.splitext(image_filename)[0]
        
        print(f"\nProcessing '{image_filename}'...")
        print(f"Using algorithm: {self.algorithm_name}")
        
        try:
            # Get algorithm instance
            algorithm = get_algorithm(self.algorithm_name, self.algorithm_config)
            
            # Generate paths
            print("Generating paths...")
            paths, metadata = algorithm.generate_paths(image_path)
            
            num_colors = metadata.get('numColors', 1)
            if num_colors > 1:
                total_segments = sum(len(color_paths) for color_paths in paths.values())
                print(f"Generated {total_segments} path segments across {num_colors} colors")
            else:
                print(f"Generated {len(paths)} path segments")
            
            # Generate SVG
            print("Creating SVG...")
            svg_content = algorithm.generate_svg(paths, metadata)
            
            # Generate preview
            print("Creating preview...")
            preview_image = algorithm.generate_preview(paths, metadata)
            
            # Save everything
            print("Saving plot...")
            metadata['algorithm'] = self.algorithm_name
            metadata['algorithm_config'] = self.algorithm_config
            
            saved_files = self.db.save_plot(
                plot_name=base_name,
                svg_content=svg_content,
                preview_image=preview_image,
                original_image_path=image_path,
                metadata=metadata
            )
            
            print("\n✓ Plot generated successfully!")
            print(f"  Plot name: {base_name}")
            print(f"  Location: {self.db.get_plot_path(base_name)}")
            print(f"  Files created:")
            for file_type, path in saved_files.items():
                print(f"    - {file_type}: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"\n✗ Error generating plot: {e}")
        
        self.pause()
    
    # ==========================================
    # RUN PLOT MENU
    # ==========================================
    
    def run_plot_menu(self):
        """Menu for running plots on the plotter."""
        self.print_header("RUN PLOT ON PLOTTER")
        
        if not self.plotter.is_available():
            print("✗ AxiDraw library not available")
            print("  Install with: pip install pyaxidraw")
            self.pause()
            return
        
        plots = self.db.list_plots()
        
        if not plots:
            print("No plots found in database/plots/")
            print("Generate a plot first from the main menu")
            self.pause()
            return
        
        print("Available plots:")
        for i, plot in enumerate(plots, 1):
            print(f"{i}. {plot}")
        print(f"{len(plots) + 1}. Back")
        
        choice = self.get_number("\nSelect plot")
        
        if choice is None or choice == len(plots) + 1:
            return
        
        if 1 <= choice <= len(plots):
            selected_plot = plots[choice - 1]
            self._run_plot(selected_plot)
        else:
            print("\nInvalid selection")
            self.pause()
    
    def _run_plot(self, plot_name: str):
        """Run a plot on the plotter."""
        svg_files = self.db.get_plot_svgs(plot_name)
        
        if not svg_files:
            print(f"\n✗ No SVG file found for plot '{plot_name}'")
            self.pause()
            return
        
        # If multiple SVG files, let user choose
        if len(svg_files) > 1:
            import os
            print(f"\nPlot: {plot_name}")
            print(f"\nFound {len(svg_files)} SVG files:")
            for i, svg_path in enumerate(svg_files, 1):
                print(f"{i}. {os.path.basename(svg_path)}")
            print(f"{len(svg_files) + 1}. Plot all files sequentially")
            print(f"{len(svg_files) + 2}. Back")
            
            choice = self.get_number("\nSelect SVG file")
            
            if choice is None or choice == len(svg_files) + 2:
                return
            elif choice == len(svg_files) + 1:
                # Plot all files
                svg_paths_to_plot = svg_files
            elif 1 <= choice <= len(svg_files):
                # Plot single selected file
                svg_paths_to_plot = [svg_files[choice - 1]]
            else:
                print("\nInvalid selection")
                self.pause()
                return
        else:
            # Single SVG file
            svg_paths_to_plot = svg_files
        
        # Show plotting options
        import os
        print(f"\nPlot: {plot_name}")
        if len(svg_paths_to_plot) == 1:
            print(f"SVG: {os.path.basename(svg_paths_to_plot[0])}")
        else:
            print(f"SVGs: {len(svg_paths_to_plot)} files")
        
        print("\n1. Plot on AxiDraw")
        print("2. Preview (simulate without plotting)")
        print("3. Back")
        
        choice = self.get_number("\nSelect option")
        
        if choice == 1:
            for svg_path in svg_paths_to_plot:
                print(f"\nPlotting: {os.path.basename(svg_path)}")
                success = self.plotter.plot_file(svg_path, preview=False)
                if not success:
                    break
            if success:
                print("\n✓ Plot complete!")
        elif choice == 2:
            for svg_path in svg_paths_to_plot:
                print(f"\nPreviewing: {os.path.basename(svg_path)}")
                success = self.plotter.plot_file(svg_path, preview=True)
                if not success:
                    break
            if success:
                print("\n✓ Preview complete!")
        
        self.pause()
    
    # ==========================================
    # PLOTTER CONTROL MENU
    # ==========================================
    
    def plotter_control_menu(self):
        """Menu for manual plotter control."""
        if not self.plotter.is_available():
            self.print_header("MANUAL PLOTTER CONTROL")
            print("✗ AxiDraw library not available")
            self.pause()
            return
        
        while True:
            self.print_header("MANUAL PLOTTER CONTROL")
            
            commands = self.plotter.get_available_commands()
            menu_items = list(commands.items())
            
            for i, (cmd, desc) in enumerate(menu_items, 1):
                print(f"{i}. {desc}")
            print(f"{len(menu_items) + 1}. Back")
            
            choice = self.get_number("\nSelect control")
            
            if choice is None or choice == len(menu_items) + 1:
                return
            
            if 1 <= choice <= len(menu_items):
                command = menu_items[choice - 1][0]
                self.plotter.execute_command(command)
                self.pause()
            else:
                print("\nInvalid selection")
                self.pause()
    
    # ==========================================
    # ALGORITHM SETTINGS MENU
    # ==========================================
    
    def algorithm_settings_menu(self):
        """Menu for algorithm settings."""
        while True:
            self.print_header("ALGORITHM SETTINGS")
            print(f"Current Algorithm: {self.algorithm_name}\n")
            
            # Display current settings
            settings_list = list(self.algorithm_config.items())
            for i, (key, value) in enumerate(settings_list, 1):
                print(f"{i}. {key}: {value}")
            
            print(f"\n{len(settings_list) + 1}. Change Algorithm")
            print(f"{len(settings_list) + 2}. Reset to Defaults")
            print(f"{len(settings_list) + 3}. Back")
            
            choice = self.get_number("\nSelect setting")
            
            if choice is None:
                continue
            elif choice == len(settings_list) + 3:
                return
            elif choice == len(settings_list) + 1:
                self._change_algorithm()
            elif choice == len(settings_list) + 2:
                algorithm = get_algorithm(self.algorithm_name)
                self.algorithm_config = algorithm.DEFAULT_CONFIG.copy()
                # Save to centralized settings
                self.settings.update_algorithm_settings(self.algorithm_config, self.algorithm_name)
                print("\n✓ Settings reset to defaults and saved")
                self.pause()
            elif 1 <= choice <= len(settings_list):
                key = settings_list[choice - 1][0]
                self._edit_algorithm_setting(key)
            else:
                print("\nInvalid selection")
                self.pause()
    
    def _change_algorithm(self):
        """Change the current algorithm."""
        print("\nAvailable algorithms:")
        print("1. waves - Wave-based plot generation")
        print("2. circles - Concentric circles with squiggles")
        print("3. fabric - Procedurally generated fabric texture")
        print("4. Back")
        
        choice = self.get_number("\nSelect algorithm")
        
        if choice == 1:
            self.algorithm_name = 'waves'
            self.settings.set_current_algorithm('waves')
            self.algorithm_config = self.settings.get_algorithm_settings('waves')
            print(f"\n✓ Algorithm changed to: {self.algorithm_name}")
            self.pause()
        elif choice == 2:
            self.algorithm_name = 'circles'
            self.settings.set_current_algorithm('circles')
            self.algorithm_config = self.settings.get_algorithm_settings('circles')
            print(f"\n✓ Algorithm changed to: {self.algorithm_name}")
            self.pause()
        elif choice == 3:
            self.algorithm_name = 'fabric'
            self.settings.set_current_algorithm('fabric')
            self.algorithm_config = self.settings.get_algorithm_settings('fabric')
            print(f"\n✓ Algorithm changed to: {self.algorithm_name}")
            self.pause()
    
    def _edit_algorithm_setting(self, key: str):
        """Edit a specific algorithm setting."""
        current_value = self.algorithm_config[key]
        print(f"\nCurrent value: {current_value}")
        
        new_value_str = self.get_input("Enter new value (or press Enter to cancel)")
        
        if not new_value_str:
            return
        
        try:
            # Try to match the type of the current value
            if isinstance(current_value, bool):
                new_value = new_value_str.lower() in ['true', '1', 'yes']
            elif isinstance(current_value, int):
                new_value = int(new_value_str)
            elif isinstance(current_value, float):
                new_value = float(new_value_str)
            else:
                new_value = new_value_str
            
            self.algorithm_config[key] = new_value
            # Save to centralized settings
            self.settings.set_algorithm_setting(key, new_value, self.algorithm_name)
            print(f"\n✓ {key} updated to: {new_value}")
        except ValueError:
            print(f"\n✗ Invalid value for {key}")
        
        self.pause()
    
    # ==========================================
    # PLOTTER SETTINGS MENU
    # ==========================================
    
    def plotter_settings_menu(self):
        """Menu for plotter hardware settings."""
        while True:
            self.print_header("PLOTTER SETTINGS")
            
            settings = self.settings.get_plotter_settings()
            settings_list = list(settings.items())
            
            for i, (key, value) in enumerate(settings_list, 1):
                print(f"{i}. {key}: {value}")
            
            print(f"\n{len(settings_list) + 1}. Reset to Defaults")
            print(f"{len(settings_list) + 2}. Back")
            
            choice = self.get_number("\nSelect setting")
            
            if choice is None:
                continue
            elif choice == len(settings_list) + 2:
                return
            elif choice == len(settings_list) + 1:
                # Reset plotter settings to defaults
                from plotter import PlotterController
                self.settings.update_plotter_settings(PlotterController.DEFAULT_CONFIG)
                self.plotter.set_config(self.settings.get_plotter_settings())
                print("\n✓ Settings reset to defaults and saved")
                self.pause()
            elif 1 <= choice <= len(settings_list):
                key = settings_list[choice - 1][0]
                self._edit_plotter_setting(key)
            else:
                print("\nInvalid selection")
                self.pause()
    
    def _edit_plotter_setting(self, key: str):
        """Edit a specific plotter setting."""
        plotter_settings = self.settings.get_plotter_settings()
        current_value = plotter_settings.get(key)
        print(f"\nCurrent value: {current_value}")
        
        new_value_str = self.get_input("Enter new value (or press Enter to cancel)")
        
        if not new_value_str:
            return
        
        try:
            # Try to match the type of the current value
            if isinstance(current_value, bool):
                new_value = new_value_str.lower() in ['true', '1', 'yes']
            elif isinstance(current_value, int):
                new_value = int(new_value_str)
            elif isinstance(current_value, float):
                new_value = float(new_value_str)
            else:
                new_value = new_value_str
            
            # Save to centralized settings
            self.settings.set_plotter_setting(key, new_value)
            # Update plotter controller
            self.plotter.set_config({key: new_value})
            print(f"\n✓ {key} updated to: {new_value}")
        except ValueError:
            print(f"\n✗ Invalid value for {key}")
        
        self.pause()
    
    # ==========================================
    # DATABASE INFO MENU
    # ==========================================
    
    def database_info_menu(self):
        """Display database information."""
        self.print_header("DATABASE INFORMATION")
        
        stats = self.db.get_database_stats()
        
        print(f"Images: {stats['images_count']}")
        print(f"  Location: {stats['images_path']}")
        
        print(f"\nPlots: {stats['plots_count']}")
        print(f"  Location: {stats['plots_path']}")
        
        # List recent plots
        plots = self.db.list_plots()
        if plots:
            print("\nRecent plots:")
            for plot in plots[-5:]:  # Show last 5
                info = self.db.get_plot_info(plot)
                if info:
                    print(f"  - {plot} ({len(info['files'])} files)")
        
        self.pause()
    
    def run(self):
        """Start the interface."""
        # self.clear_screen()
        self.main_menu()