import os
import sys
import time
import traceback
from typing import Optional
from manager import DatabaseManager
from plotter import PlotterController
from settings import SettingsManager

from algorithms import get_algorithm, list_algorithms

class CommandLineInterface:
    def __init__(self, db_manager: DatabaseManager, 
                 plotter: PlotterController,
                 settings_manager: SettingsManager):

        self.db = db_manager
        self.plotter = plotter
        self.settings = settings_manager
        
        # Load current algorithm settings
        self.algorithm_name = self.settings.get_current_algorithm()
        self.algorithm_config = self.settings.get_algorithm_settings(self.algorithm_name)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _get_algorithm_instance(self):
        return get_algorithm(self.algorithm_name, self.algorithm_config)

    def _algorithm_is_procedural(self) -> bool:
        try:
            return self._get_algorithm_instance().is_procedural()
        except Exception:
            return False

    def _generation_needs_image(self) -> bool:
        """Whether plot generation should prompt for an image."""
        if not self._algorithm_is_procedural():
            return True
        # Special case: fabric can be procedural or image-driven depending on config
        if self.algorithm_name == 'fabric':
            return self.algorithm_config.get('displacement_mode', 'random') == 'image'
        return False

    def _generate_menu_label(self) -> str:
        return "Generate Plot from Image" if self._generation_needs_image() else "Generate Plot (Procedural)"

    def _select_from_list(self, items: list[str], prompt: str, header: Optional[str] = None) -> Optional[str]:
        """Render a numbered list with a Back option. Returns selected item or None."""
        if header:
            self.print_header(header)

        if not items:
            return None

        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
        print(f"{len(items) + 1}. Back")

        choice = self.get_number(prompt)
        if choice is None or choice == len(items) + 1:
            return None
        if 1 <= choice <= len(items):
            return items[choice - 1]

        print("\nInvalid selection")
        self.pause()
        return None

    def _parse_typed_value(self, current_value, new_value_str: str):
        """Parse user input into the type of current_value."""
        if isinstance(current_value, bool):
            return new_value_str.lower() in ['true', '1', 'yes', 'y']
        if isinstance(current_value, int):
            return int(new_value_str)
        if isinstance(current_value, float):
            return float(new_value_str)
        return new_value_str

    def _set_current_algorithm(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.settings.set_current_algorithm(algorithm_name)
        self.algorithm_config = self.settings.get_algorithm_settings(algorithm_name)
    
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

            print(f"1. {self._generate_menu_label()}")
            
            print("2. Run Existing Plot on Plotter")
            print("3. Manual Plotter Control")
            print("4. Algorithm Settings")
            print("5. Plotter Settings")
            print("6. View Database Info")
            print("7. Merge SVGs")
            print("8. Exit")
            
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
                self.merge_svgs_menu()
            elif choice == 8:
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
        if self._generation_needs_image():
            self._generate_plot_from_image()
            return
        self._generate_plot_procedural()
    
    def _generate_plot_from_image(self):
        """Menu for generating plots from images."""
        images = self.db.list_images()

        if not images:
            self.print_header("GENERATE PLOT FROM IMAGE")
            print("No images found in database/images/")
            print("Please add images to the database/images/ directory")
            self.pause()
            return

        print("Available images:")
        selected_image = self._select_from_list(images, "\nSelect image", header="GENERATE PLOT FROM IMAGE")
        if selected_image:
            self._generate_plot(selected_image)
    
    def _generate_plot_procedural(self):
        """Generate a procedural plot (no input image required)."""
        self.print_header("GENERATE PROCEDURAL PLOT")
        
        # Generate name based on algorithm
        # For procedural algorithms, use a consistent name to replace previous versions
        base_name = 'spirals' if self.algorithm_name == 'spiral' else self.algorithm_name
        
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
            traceback.print_exc()
        
        self.pause()
    
    def _generate_plot(self, image_filename: str):
        """Generate a plot from an image."""
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
            traceback.print_exc()
        
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
        selected_plot = self._select_from_list(plots, "\nSelect plot")
        if selected_plot:
            self._run_plot(selected_plot)
    
    def _run_plot(self, plot_name: str):
        """Run a plot on the plotter."""
        svg_files = self.db.get_plot_svgs(plot_name)
        
        if not svg_files:
            print(f"\n✗ No SVG file found for plot '{plot_name}'")
            self.pause()
            return
        
        # If multiple SVG files, let user choose
        if len(svg_files) > 1:
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
        print(f"\nPlot: {plot_name}")
        if len(svg_paths_to_plot) == 1:
            print(f"SVG: {os.path.basename(svg_paths_to_plot[0])}")
        else:
            print(f"SVGs: {len(svg_paths_to_plot)} files")
        
        print("\n1. Plot on AxiDraw")
        print("2. Preview (simulate without plotting)")
        print("3. Back")
        
        choice = self.get_number("\nSelect option")
        
        if choice in (1, 2):
            preview = (choice == 2)
            verb = "Previewing" if preview else "Plotting"
            success = True
            for svg_path in svg_paths_to_plot:
                print(f"\n{verb}: {os.path.basename(svg_path)}")
                success = self.plotter.plot_file(svg_path, preview=preview)
                if not success:
                    break
            if success:
                print("\n✓ Preview complete!" if preview else "\n✓ Plot complete!")
        
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
        self.print_header("CHANGE ALGORITHM")

        algorithms = list_algorithms()
        if not algorithms:
            print("No algorithms found.")
            self.pause()
            return

        for i, algo in enumerate(algorithms, 1):
            procedural = "procedural" if algo.get('procedural') else "image"
            desc = algo.get('description', '')
            print(f"{i}. {algo['name']} ({procedural}) - {desc}")
        print(f"{len(algorithms) + 1}. Back")

        choice = self.get_number("\nSelect algorithm")
        if choice is None or choice == len(algorithms) + 1:
            return
        if 1 <= choice <= len(algorithms):
            selected = algorithms[choice - 1]['name']
            self._set_current_algorithm(selected)
            print(f"\n✓ Algorithm changed to: {self.algorithm_name}")
            self.pause()
            return

        print("\nInvalid selection")
        self.pause()
    
    def _edit_algorithm_setting(self, key: str):
        """Edit a specific algorithm setting."""
        current_value = self.algorithm_config[key]
        print(f"\nCurrent value: {current_value}")
        
        new_value_str = self.get_input("Enter new value (or press Enter to cancel)")
        
        if not new_value_str:
            return
        
        try:
            new_value = self._parse_typed_value(current_value, new_value_str)
            
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
            new_value = self._parse_typed_value(current_value, new_value_str)
            
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
    
    # ==========================================
    # MERGE SVGS MENU
    # ==========================================
    
    def merge_svgs_menu(self):
        """Menu for merging SVGs."""
        while True:
            self.print_header("MERGE SVGS - SELECT PLOT")
            # 1. Select Plot
            plots = self.db.list_plots()
            if not plots:
                print("\nNo plots found.")
                self.pause()
                return

            plot_name = self._select_from_list(
                plots, 
                "Select plot containing SVGs to merge", 
            )
            
            if not plot_name:
                return
            
            # 2. List SVGs and Merge
            while True:
                svg_files = self.db.get_plot_svgs(plot_name)
                
                if not svg_files:
                    print(f"\nNo SVG files found in plot '{plot_name}'")
                    self.pause()
                    break
                    
                self.print_header(f"MERGE SVGS - {plot_name}")
                print(f"Found {len(svg_files)} SVG files:")
                
                # List files with full names
                for i, f_path in enumerate(svg_files, 1):
                    f_name = os.path.basename(f_path)
                    print(f"{i}. {f_name}")
                
                # Add extra option for "Plot all files sequentially"
                print(f"{len(svg_files) + 1}. Plot all files sequentially")
                print(f"{len(svg_files) + 2}. Back")
                
                response = self.get_input(f"\nEnter file numbers to merge (e.g. 2, 3, 5)")
                
                # Handle Back or empty
                if not response:
                    continue
                
                # Parse response
                # Check for explicit menu options (single number)
                try:
                    if response.isdigit():
                        val = int(response)
                        if val == len(svg_files) + 2:
                            break # Back
                        if val == len(svg_files) + 1:
                            print("\nFeature 'Plot all files sequentially' is not implemented in this version.")
                            self.pause()
                            continue
                except:
                    pass

                # Parse comma separated list
                try:
                    indices = [int(x.strip()) for x in response.split(',')]
                    
                    selected_files = []
                    for idx in indices:
                        if 1 <= idx <= len(svg_files):
                            selected_files.append(os.path.basename(svg_files[idx-1]))
                        # Ignore out of range or special menu options if mixed (though user shouldn't mix)
                    
                    if not selected_files:
                        print("\nNo valid files selected.")
                        self.pause()
                        continue
                        
                    # Perform merge
                    print(f"\nMerging {len(selected_files)} files...")
                    
                    # Generate output name
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_name = f"merged_{timestamp}.svg"
                    
                    result = self.db.merge_svgs(plot_name, selected_files, output_name)
                    
                    if result:
                        print(f"\n✓ Successfully merged {len(selected_files)} files into '{output_name}'")
                    else:
                        print("\n✗ Failed to merge files")
                    
                    self.pause()
                    
                except ValueError as e:
                    print(f"\nInvalid input: {e}")
                    self.pause()

    def run(self):
        """Start the interface."""
        # self.clear_screen()
        self.main_menu()