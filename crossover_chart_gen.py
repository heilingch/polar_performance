# crossover_chart_gen.py

"""
Sail Crossover Chart Generator

This script generates a sail crossover chart based on a vessel's polar performance file
and a sail inventory defined in a JSON file. The chart visualizes the most suitable 
sail for a given combination of True Wind Speed (TWS) and True Wind Angle (TWA).

Based on the concept in 'sail_crossover_chart_generator.md'.
"""

# --- Step 1: Import necessary libraries ---
# We will need:
# - json: To load the sail inventory from the .json file.
# - numpy: For efficient numerical operations, especially for creating the TWS/TWA grid.
# - matplotlib.pyplot: For plotting the final crossover chart.
# - PolarProcessor: The existing class from our project to handle polar data calculations.

import json
import numpy as np
import matplotlib.pyplot as plt
from polar_processor import PolarProcessor
import math # To handle potential math errors
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QTextEdit)
from PyQt6.QtGui import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from scipy.ndimage import center_of_mass
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Step 2: Define the CrossoverChartGenerator Class ---
# This class will encapsulate all the logic for loading data,
# performing calculations, and generating the plot.

class CrossoverChartGenerator:
    def __init__(self, polar_file_path, sail_inventory_path):
        """
        Initializes the generator with paths to the polar and sail inventory files.
        
        Args:
            polar_file_path (str): The file path for the boat's .pol file.
            sail_inventory_path (str): The file path for the sail inventory .json file.
        """
        # 2a: Initialize the PolarProcessor and load the polar data.
        # This gives us access to performance calculation methods like get_target_speed()
        # and the apparent wind calculation tw_2_aw().
        self.polar_processor = PolarProcessor()
        success, message = self.polar_processor.load_polar_file(polar_file_path)
        if not success:
            raise IOError(f"Failed to load polar file: {message}")
        print(f"Successfully loaded polar file: {polar_file_path}")

        # 2b: Load the sail inventory from the specified JSON file.
        # This will be a dictionary mapping sail names to their operational limits.
        self.sail_inventory = self._load_sail_inventory(sail_inventory_path)
        print(f"Successfully loaded sail inventory: {sail_inventory_path}")
        
        # 2c: Categorize sails for combination logic based on the 'type' attribute.
        self.mainsails = {s: d for s, d in self.sail_inventory.items() if d.get('type') == 'main_sail'}
        headsails = {s: d for s, d in self.sail_inventory.items() if d.get('type') == 'head_sail'}
        
        # Further distinguish headsails for priority logic (spinnakers vs. jibs)
        self.jibs = {s: d for s, d in headsails.items() if 'spinnaker' not in s}
        self.spinnakers = {s: d for s, d in headsails.items() if 'spinnaker' in s}
        
        print(f"Categorized sails: {len(self.mainsails)} mains, {len(headsails)} headsails ({len(self.jibs)} jibs, {len(self.spinnakers)} spinnakers).")


    def _load_sail_inventory(self, file_path):
        """
        Loads the sail inventory from a JSON file into a dictionary.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise IOError(f"Could not read or parse sail inventory file: {e}")

    # --- Step 3: Generate the Crossover Data ---
    # This is the core calculation step. We will create a grid of TWS/TWA values
    # and, for each point, determine the best sail.
    
    def generate_crossover_data(self, tws_range, twa_range):
        """
        Generates a dictionary of boolean grids, one for each valid sail combination.

        Args:
            tws_range (np.array): An array of True Wind Speeds to evaluate.
            twa_range (np.array): An array of True Wind Angles to evaluate.

        Returns:
            A dictionary where keys are sail combination names (e.g., "main_full & jib_full")
            and values are 2D numpy boolean arrays.
        """
        # 3a: Create a dictionary to hold a grid for each possible sail combination.
        sail_combo_grids = {
            f"{main} & {jib}": np.zeros((len(twa_range), len(tws_range)), dtype=bool)
            for main in self.mainsails for jib in self.jibs
        }
        sail_combo_grids.update({
            f"{main} & {spinnaker}": np.zeros((len(twa_range), len(tws_range)), dtype=bool)
            for main in self.mainsails for spinnaker in self.spinnakers
        })
        # Add grids for main-sail-only configurations
        sail_combo_grids.update({
            main: np.zeros((len(twa_range), len(tws_range)), dtype=bool)
            for main in self.mainsails
        })

        # 3b: Iterate over every TWA and TWS combination.
        for i, twa in enumerate(twa_range):
            for j, tws in enumerate(tws_range):
                
                # 3c: Calculate Apparent Wind for the current TWA/TWS.
                speed_success, speed_or_msg = self.polar_processor.get_target_speed(twa, tws)
                if not speed_success:
                    continue

                boat_speed = float(speed_or_msg)
                aw_data = self.polar_processor.tw_2_aw(twa, tws, boat_speed)
                if not aw_data or math.isnan(aw_data['awa']) or math.isnan(aw_data['aws']):
                    continue
                
                awa = aw_data['awa']
                aws = aw_data['aws']

                # 3d: Find all suitable mainsails and headsails for the calculated AWA/AWS.
                suitable_mains = self._find_suitable_sails(awa, aws, self.mainsails)
                suitable_jibs = self._find_suitable_sails(awa, aws, self.jibs)
                suitable_spinnakers = self._find_suitable_sails(awa, aws, self.spinnakers)

                # 3e: Populate the grids for valid combinations.
                # This logic prioritizes spinnakers over jibs, and then jibs over main-only.
                if suitable_spinnakers:
                    for main in suitable_mains:
                        for spinnaker in suitable_spinnakers:
                            combo_name = f"{main} & {spinnaker}"
                            sail_combo_grids[combo_name][i, j] = True
                elif suitable_jibs:
                    for main in suitable_mains:
                        for jib in suitable_jibs:
                            combo_name = f"{main} & {jib}"
                            sail_combo_grids[combo_name][i, j] = True
                elif suitable_mains:
                    # Handle the case where only a mainsail is suitable
                    for main in suitable_mains:
                        sail_combo_grids[main][i, j] = True
        
        return sail_combo_grids

    def _find_suitable_sails(self, awa, aws, sail_category):
        """
        Checks the calculated AWA and AWS against a specific category of sails.

        Args:
            awa (float): The calculated apparent wind angle.
            aws (float): The calculated apparent wind speed.
            sail_category (dict): The dictionary of sails to check (e.g., self.mainsails).

        Returns:
            A list of sail names that are suitable for the given conditions.
        """
        suitable_sails = []
        for sail_name, limits in sail_category.items():
            if (limits['min_awa'] <= awa <= limits['max_awa'] and
                limits['min_aws'] <= aws <= limits['max_aws']):
                suitable_sails.append(sail_name)
        return suitable_sails

    # --- Step 4: Plot the Crossover Chart ---
    # This function will take the generated data grid and use matplotlib
    # to create and display the visual chart.
    
    def plot_chart(self, sail_combo_grids, tws_range, twa_range, style='twa_on_y', current_tws=None, current_twa=None, canvas=None):
        """
        Generates and displays the crossover chart using layered contours.

        Args:
            sail_combo_grids (dict): The sail combination grids generated by 
                                    `generate_crossover_data()`.
            tws_range (np.array): The range of True Wind Speeds.
            twa_range (np.array): The range of True Wind Angles.
            style (str): The style of the chart ('twa_on_y' or 'tws_on_y').
            current_tws (float, optional): The current TWS to plot as a marker.
            current_twa (float, optional): The current TWA to plot as a marker.
            canvas (FigureCanvasQTAgg, optional): The canvas to draw the plot on.
        """
        # 4a: Set up the plot figure and axes.
        if canvas:
            fig = canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.grid(True, linestyle=':', linewidth=0.5)
        TWS, TWA = np.meshgrid(tws_range, twa_range)

        # 4b: Define a list of colors for the sails and prepare for the legend.
        # Using a colormap ensures distinct colors for each sail combination.
        num_combos = len(sail_combo_grids)
        colors = plt.cm.get_cmap('tab20', num_combos)
        legend_artists = []
        legend_labels = []

        if SCIPY_AVAILABLE:
            # --- NEW: Logic for inline labels ---
            # This block now runs exclusively if scipy is found.
            ax.set_title('Sail Crossover Chart')

            # 4c: Plot each sail combination as a semi-transparent layer.
            # The arguments to contourf are swapped based on the desired style.
            for i, (sail_name, grid) in enumerate(sail_combo_grids.items()):
                if np.any(grid):
                    if style == 'twa_on_y':
                        ax.contourf(TWS, TWA, grid, levels=[0.5, 1.5], colors=[colors(i)], alpha=0.5)
                    else: # 'tws_on_y'
                        # The fix is to transpose the coordinate grids (TWA.T, TWS.T)
                        # to match the transposed data grid (grid.T).
                        ax.contourf(TWA.T, TWS.T, grid.T, levels=[0.5, 1.5], colors=[colors(i)], alpha=0.5)
                    
                    # Calculate the center of the colored region to place the label
                    center_row, center_col = center_of_mass(grid)
                    center_twa = twa_range[int(center_row)]
                    center_tws = tws_range[int(center_col)]

                    # Add the text label to the plot
                    if style == 'twa_on_y':
                        ax.text(center_tws, center_twa, sail_name, ha='center', va='center', color='black', fontsize=8, rotation=45)
                    else: # 'tws_on_y'
                        ax.text(center_twa, center_tws, sail_name, ha='center', va='center', color='black', fontsize=8, rotation=45)

            # 4e: Plot the current wind condition if provided.
            if current_tws is not None and current_twa is not None:
                if style == 'twa_on_y':
                    ax.plot(current_tws, current_twa, marker='o', color='darkgreen', markersize=10, zorder=10)
                    ax.set_xlabel('True Wind Speed (knots)')
                    ax.set_ylabel('True Wind Angle (degrees)')
                else: # 'tws_on_y'
                    ax.plot(current_twa, current_tws, marker='o', color='darkgreen', markersize=10, zorder=10)
                    ax.set_ylabel('True Wind Speed (knots)')
                    ax.set_xlabel('True Wind Angle (degrees)')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

            # 4f: Add the legend to the plot.
            # Place the legend outside the main plot area to avoid obscuring data.
            # ax.legend(legend_artists, legend_labels, title="Sail Combinations", bbox_to_anchor=(1.05, 1), loc='upper left')

            # 4g: Adjust layout to prevent labels from being cut off.
            fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        else:
            # --- FALLBACK: Logic for legend-based plotting ---
            # This block now runs exclusively if scipy is NOT found.
            print("\n---")
            print("Warning: 'scipy' not found. Plotting with a legend instead of inline labels.")
            print("For inline labels, please install scipy: pip install scipy")
            print("---\n")
            ax.set_title('Sail Crossover Chart (Legend)')

            # 4c: Plot each sail combination as a semi-transparent layer.
            # The arguments to contourf are swapped based on the desired style.
            for i, (sail_name, grid) in enumerate(sail_combo_grids.items()):
                if np.any(grid):
                    if style == 'twa_on_y':
                        ax.contourf(TWS, TWA, grid, levels=[0.5, 1.5], colors=[colors(i)], alpha=0.5)
                    else: # 'tws_on_y'
                        # The fix is to transpose the coordinate grids (TWA.T, TWS.T)
                        # to match the transposed data grid (grid.T).
                        ax.contourf(TWA.T, TWS.T, grid.T, levels=[0.5, 1.5], colors=[colors(i)], alpha=0.5)
                    
                    # Create a proxy artist (a colored rectangle) for the legend.
                    legend_artists.append(plt.Rectangle((0, 0), 1, 1, color=colors(i), alpha=0.5))
                    legend_labels.append(sail_name)

            # 4e: Plot the current wind condition if provided.
            if current_tws is not None and current_twa is not None:
                if style == 'twa_on_y':
                    ax.plot(current_tws, current_twa, marker='o', color='darkgreen', markersize=10, zorder=10)
                    ax.set_xlabel('True Wind Speed (knots)')
                    ax.set_ylabel('True Wind Angle (degrees)')
                else: # 'tws_on_y'
                    ax.plot(current_twa, current_tws, marker='o', color='darkgreen', markersize=10, zorder=10)
                    ax.set_ylabel('True Wind Speed (knots)')
                    ax.set_xlabel('True Wind Angle (degrees)')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

            # 4f: Add the legend to the plot.
            # Place the legend outside the main plot area to avoid obscuring data.
            #ax.legend(legend_artists, legend_labels, title="Sail Combinations", bbox_to_anchor=(1.05, 1), loc='upper left')

            # 4g: Adjust layout to prevent labels from being cut off.
            fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        # 4h: Display the plot.
        if canvas:
            canvas.draw()
        else:
            plt.show()

class CrossoverChartGUI(QMainWindow):
    def __init__(self, polar_file, sail_inventory_file):
        super().__init__()
        self.polar_file = polar_file
        self.sail_inventory_file = sail_inventory_file
        self.generator = CrossoverChartGenerator(self.polar_file, self.sail_inventory_file)
        
        self.setWindowTitle("Sail Crossover Chart Generator")
        self.setGeometry(100, 100, 1200, 800)
        
        self._create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        control_layout = QVBoxLayout()
        
        # TWS input
        control_layout.addWidget(QLabel("True Wind Speed (TWS):"))
        self.tws_input = QLineEdit("10")
        control_layout.addWidget(self.tws_input)
        
        # TWA input
        control_layout.addWidget(QLabel("True Wind Angle (TWA):"))
        self.twa_input = QLineEdit("90")
        control_layout.addWidget(self.twa_input)
        
        # Submit button
        self.submit_button = QPushButton("Generate Chart & Suggest Sail")
        control_layout.addWidget(self.submit_button)
        
        # Output text box
        control_layout.addWidget(QLabel("Optimum Sail:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        control_layout.addWidget(self.output_text)
        
        control_layout.addStretch()
        
        # Right panel for the chart
        self.canvas = FigureCanvas(plt.figure(figsize=(14, 8)))
        
        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.canvas, 3)
        
        self.submit_button.clicked.connect(self.generate_chart_and_suggestion)
        self.generate_chart_and_suggestion()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        load_sail_action = QAction("Load Sail Inventory", self)
        load_sail_action.triggered.connect(self.load_sail_inventory)
        file_menu.addAction(load_sail_action)

    def load_sail_inventory(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Sail Inventory", "", "JSON Files (*.json)")
        if file_name:
            self.sail_inventory_file = file_name
            try:
                self.generator = CrossoverChartGenerator(self.polar_file, self.sail_inventory_file)
                self.generate_chart_and_suggestion()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load sail inventory: {e}")

    def generate_chart_and_suggestion(self):
        try:
            current_tws = float(self.tws_input.text())
            current_twa = float(self.twa_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "TWS and TWA must be numbers.")
            return

        tws_range = np.arange(4, 40, 0.1)
        twa_range = np.arange(30, 181, 1)
        
        sail_combo_grids = self.generator.generate_crossover_data(tws_range, twa_range)
        
        self.generator.plot_chart(sail_combo_grids, tws_range, twa_range, 
                                  style='tws_on_y',
                                  current_tws=current_tws, current_twa=current_twa, 
                                  canvas=self.canvas)
        
        # Find optimum sail
        optimum_sail = "No suitable sail found"
        for sail_name, grid in sail_combo_grids.items():
            twa_idx = np.abs(twa_range - current_twa).argmin()
            tws_idx = np.abs(tws_range - current_tws).argmin()
            if grid[twa_idx, tws_idx]:
                optimum_sail = sail_name
                break
        self.output_text.setText(optimum_sail)
        
        self.submit_button.clicked.connect(self.generate_chart_and_suggestion)
        self.generate_chart_and_suggestion()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        load_sail_action = QAction("Load Sail Inventory", self)
        load_sail_action.triggered.connect(self.load_sail_inventory)
        file_menu.addAction(load_sail_action)

    def load_sail_inventory(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Sail Inventory", "", "JSON Files (*.json)")
        if file_name:
            self.sail_inventory_file = file_name
            try:
                self.generator = CrossoverChartGenerator(self.polar_file, self.sail_inventory_file)
                self.generate_chart_and_suggestion()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load sail inventory: {e}")

    def generate_chart_and_suggestion(self):
        try:
            current_tws = float(self.tws_input.text())
            current_twa = float(self.twa_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "TWS and TWA must be numbers.")
            return

        tws_range = np.arange(4, 40, 0.5)
        twa_range = np.arange(30, 181, 1)
        
        sail_combo_grids = self.generator.generate_crossover_data(tws_range, twa_range)
        
        self.generator.plot_chart(sail_combo_grids, tws_range, twa_range, 
                                  style='tws_on_y',
                                  current_tws=current_tws, current_twa=current_twa, 
                                  canvas=self.canvas)
        
        # Find optimum sail
        optimum_sail = "No suitable sail found"
        for sail_name, grid in sail_combo_grids.items():
            twa_idx = np.abs(twa_range - current_twa).argmin()
            tws_idx = np.abs(tws_range - current_tws).argmin()
            if grid[twa_idx, tws_idx]:
                optimum_sail = sail_name
                break
        self.output_text.setText(optimum_sail)

# --- Main execution block ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Define the initial file paths. These can be changed via the GUI.
    POLAR_FILE = 'test_polars/elan-impression-444.pol'
    SAIL_INVENTORY_FILE = 'ariadne_sails.json'
    
    try:
        window = CrossoverChartGUI(POLAR_FILE, SAIL_INVENTORY_FILE)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        # Basic error handling for GUI startup
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Application Error", f"An error occurred: {e}")
        sys.exit(1)
