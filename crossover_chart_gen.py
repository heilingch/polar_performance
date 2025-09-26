# crossover_chart_gen.py

"""
Sail Crossover Chart Generator

This script generates a sail crossover chart based on a vessel's polar performance file
and a sail inventory defined in a JSON file. The chart visualizes the most suitable 
sail for a given combination of True Wind Speed (TWS) and True Wind Angle (TWA).

Based on the concept in 'sail_crossover_chart_generator.md'.
"""

# --- Step 1: Import necessary libraries ---
import json
import numpy as np
import matplotlib
matplotlib.use('QtAgg') # Ensure Matplotlib uses the QtAgg backend
import matplotlib.pyplot as plt
from polar_processor import PolarProcessor
import math 
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QTextEdit)
from PyQt6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Scipy is now a hard dependency for inline labels
try:
    from scipy.ndimage import center_of_mass
except ImportError:
    raise ImportError("The 'scipy' library is required for inline sail labels. Please install it: pip install scipy")

class CrossoverChartGenerator:
    def __init__(self, polar_file_path, sail_inventory_path):
        """
        Initializes the generator with paths to the polar and sail inventory files.
        """
        self.polar_processor = PolarProcessor()
        success, message = self.polar_processor.load_polar_file(polar_file_path)
        if not success:
            raise IOError(f"Failed to load polar file: {message}")
        print(f"Successfully loaded polar file: {polar_file_path}")

        self.sail_inventory = self._load_sail_inventory(sail_inventory_path)
        print(f"Successfully loaded sail inventory: {sail_inventory_path}")
        
        self.mainsails = {s: d for s, d in self.sail_inventory.items() if d.get('type') == 'main_sail'}
        headsails = {s: d for s, d in self.sail_inventory.items() if d.get('type') == 'head_sail'}
        
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
    
    def generate_crossover_data(self, tws_range, twa_range):
        """
        Generates a dictionary of boolean grids, one for each valid sail combination.
        """
        sail_combo_grids = {}
        for main in self.mainsails:
            for jib in self.jibs:
                sail_combo_grids[f"{main} & {jib}"] = np.zeros((len(twa_range), len(tws_range)), dtype=bool)
            for spinnaker in self.spinnakers:
                sail_combo_grids[f"{main} & {spinnaker}"] = np.zeros((len(twa_range), len(tws_range)), dtype=bool)
            sail_combo_grids[main] = np.zeros((len(twa_range), len(tws_range)), dtype=bool)

        for i, twa in enumerate(twa_range):
            for j, tws in enumerate(tws_range):
                
                speed_success, speed_or_msg = self.polar_processor.get_target_speed(twa, tws)
                if not speed_success:
                    continue

                boat_speed = float(speed_or_msg)
                aw_data = self.polar_processor.tw_2_aw(twa, tws, boat_speed)
                if not aw_data or math.isnan(aw_data['awa']) or math.isnan(aw_data['aws']):
                    continue
                
                awa = aw_data['awa']
                aws = aw_data['aws']

                suitable_mains = self._find_suitable_sails(awa, aws, self.mainsails)
                suitable_jibs = self._find_suitable_sails(awa, aws, self.jibs)
                suitable_spinnakers = self._find_suitable_sails(awa, aws, self.spinnakers)

                if suitable_spinnakers:
                    for main in suitable_mains:
                        for spinnaker in suitable_spinnakers:
                            combo_name = f"{main} & {spinnaker}"
                            if combo_name in sail_combo_grids: # Ensure combo exists
                                sail_combo_grids[combo_name][i, j] = True
                elif suitable_jibs:
                    for main in suitable_mains:
                        for jib in suitable_jibs:
                            combo_name = f"{main} & {jib}"
                            if combo_name in sail_combo_grids: # Ensure combo exists
                                sail_combo_grids[combo_name][i, j] = True
                elif suitable_mains:
                    for main in suitable_mains:
                        if main in sail_combo_grids: # Ensure combo exists
                            sail_combo_grids[main][i, j] = True
        
        return sail_combo_grids

    def _find_suitable_sails(self, awa, aws, sail_category):
        """
        Checks the calculated AWA and AWS against a specific category of sails.
        """
        suitable_sails = []
        for sail_name, limits in sail_category.items():
            if (limits['min_awa'] <= awa <= limits['max_awa'] and
                limits['min_aws'] <= aws <= limits['max_aws']):
                suitable_sails.append(sail_name)
        return suitable_sails
    
    def plot_chart(self, ax, sail_combo_grids, tws_range, twa_range, current_tws=None, current_twa=None, clear_base_plot=True):
        """
        Generates and displays the crossover chart using layered contours on the provided axes.
        
        Args:
            ax (matplotlib.axes.Axes): The Matplotlib axes to draw on.
            sail_combo_grids (dict): The sail combination grids.
            tws_range (np.array): The range of True Wind Speeds.
            twa_range (np.array): The range of True Wind Angles.
            current_tws (float, optional): The current TWS to plot as a marker.
            current_twa (float, optional): The current TWA to plot as a marker.
            clear_base_plot (bool): If True, clears and redraws the base plot (contours and labels).
                                    If False, assumes base plot is present and only updates the marker.
        """
        
        if clear_base_plot:
            ax.clear()
            ax.grid(True, linestyle=':', linewidth=0.5)
            TWS, TWA = np.meshgrid(tws_range, twa_range)

            num_combos = len(sail_combo_grids)
            colors = plt.cm.get_cmap('tab20', num_combos)
            
            ax.set_title('Sail Crossover Chart')

            for i, (sail_name, grid) in enumerate(sail_combo_grids.items()):
                if np.any(grid):
                    # Using 'tws_on_y' style as per GUI's implementation
                    ax.contourf(TWA.T, TWS.T, grid.T, levels=[0.5, 1.5], colors=[colors(i)], alpha=0.5)
                    
                    # Calculate center of mass for label placement
                    center_row, center_col = center_of_mass(grid)
                    center_twa = twa_range[int(center_row)]
                    center_tws = tws_range[int(center_col)]

                    ax.text(center_twa, center_tws, sail_name, ha='center', va='center', color='black', fontsize=8, rotation=45)
            
            ax.set_ylabel('True Wind Speed (knots)')
            ax.set_xlabel('True Wind Angle (degrees)')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.figure.tight_layout() # Use tight_layout for better spacing

        # Remove existing marker if any
        if hasattr(ax, '_current_condition_marker') and ax._current_condition_marker is not None:
            ax._current_condition_marker.remove()
            ax._current_condition_marker = None

        # Plot the current wind condition if provided.
        if current_tws is not None and current_twa is not None:
            # Using 'tws_on_y' style
            marker, = ax.plot(current_twa, current_tws, marker='o', color='darkgreen', markersize=10, zorder=10)
            ax._current_condition_marker = marker # Store the marker for easy removal

class CrossoverChartGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.polar_file = None
        self.sail_inventory_file = None
        self.generator = None

        self.sail_combo_grids = None
        self.tws_range = None
        self.twa_range = None
        self.chart_drawn_once = False 

        self.setWindowTitle("Sail Crossover Chart Generator")
        self.setGeometry(100, 100, 1200, 800)

        self._create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_layout = QVBoxLayout()

        control_layout.addWidget(QLabel("True Wind Speed (TWS):"))
        self.tws_input = QLineEdit("10")
        control_layout.addWidget(self.tws_input)

        control_layout.addWidget(QLabel("True Wind Angle (TWA):"))
        self.twa_input = QLineEdit("90")
        control_layout.addWidget(self.twa_input)

        self.submit_button = QPushButton("Generate/Update Chart & Suggest Sail")
        self.submit_button.setEnabled(False)
        control_layout.addWidget(self.submit_button)

        control_layout.addWidget(QLabel("Optimum Sail:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        control_layout.addWidget(self.output_text)

        control_layout.addStretch()
        
        # Create a Matplotlib figure and canvas
        self.fig = plt.Figure(figsize=(14, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111) # Get the axes from the figure

        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.canvas, 3)

        self.submit_button.clicked.connect(self.generate_chart_and_suggestion)

        self.output_text.setText("Please load both a polar file and a sail inventory file from the File menu.")
        
        # Initialize the marker attribute on the axes
        self.ax._current_condition_marker = None

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        load_polar_action = QAction("Load Polar File", self)
        load_polar_action.triggered.connect(self.load_polar_file)
        file_menu.addAction(load_polar_action)

        load_sail_action = QAction("Load Sail Inventory", self)
        load_sail_action.triggered.connect(self.load_sail_inventory)
        file_menu.addAction(load_sail_action)

    def load_polar_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Polar File", "", "POL Files (*.pol)")
        if file_name:
            self.polar_file = file_name
            self.try_initialize_generator()

    def load_sail_inventory(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Sail Inventory", "", "JSON Files (*.json)")
        if file_name:
            self.sail_inventory_file = file_name
            self.try_initialize_generator()

    def try_initialize_generator(self):
        if self.polar_file and self.sail_inventory_file:
            try:
                self.generator = CrossoverChartGenerator(self.polar_file, self.sail_inventory_file)
                self.submit_button.setEnabled(True)
                self.output_text.setText("Files loaded. Click 'Generate/Update Chart & Suggest Sail' to draw the chart.")
                self.chart_drawn_once = False 
                self.sail_combo_grids = None
                self.tws_range = None
                self.twa_range = None
                self.ax.clear() # Clear the plot when new files are loaded
                self.canvas.draw()
            except Exception as e:
                self.output_text.setText(f"Error loading files: {e}")
                self.submit_button.setEnabled(False)

    def generate_chart_and_suggestion(self):
        if not self.generator:
            self.output_text.setText("Please load both a polar file and a sail inventory file from the File menu.")
            return
        
        try:
            current_tws = float(self.tws_input.text())
            current_twa = float(self.twa_input.text())
        except ValueError:
            self.output_text.setText("Invalid TWS or TWA input. Please enter numerical values.")
            return

        if not self.chart_drawn_once:
            self.tws_range = np.arange(4, 40, 0.1)
            self.twa_range = np.arange(30, 181, 0.5)
            print("Generating crossover data (this may take a moment)...")
            self.sail_combo_grids = self.generator.generate_crossover_data(self.tws_range, self.twa_range)
            print("Data generation complete. Plotting base chart.")
            self.generator.plot_chart(self.ax, self.sail_combo_grids, self.tws_range, self.twa_range,
                                      current_tws=current_tws, current_twa=current_twa,
                                      clear_base_plot=True)
            self.chart_drawn_once = True
        else:
            print("Updating marker on existing chart.")
            self.generator.plot_chart(self.ax, self.sail_combo_grids, self.tws_range, self.twa_range,
                                      current_tws=current_tws, current_twa=current_twa,
                                      clear_base_plot=False)
            
        optimum_sail = "No suitable sail found"
        if self.sail_combo_grids is not None and self.twa_range is not None and self.tws_range is not None:
            # Find the closest index for current_twa and current_tws
            # Use np.searchsorted for more efficient lookup if ranges are sorted
            i = np.searchsorted(self.twa_range, current_twa)
            j = np.searchsorted(self.tws_range, current_tws)
            
            # Adjust indices if current_twa/tws fall between grid points
            # This is a simplification; for precise lookup, interpolate or find exact grid cell
            if i > 0 and (i == len(self.twa_range) or abs(self.twa_range[i] - current_twa) > abs(self.twa_range[i-1] - current_twa)):
                i -= 1
            if j > 0 and (j == len(self.tws_range) or abs(self.tws_range[j] - current_tws) > abs(self.tws_range[j-1] - current_tws)):
                j -= 1

            if 0 <= i < len(self.twa_range) and 0 <= j < len(self.tws_range):
                for sail_name, grid in self.sail_combo_grids.items():
                    if grid[i, j]:
                        optimum_sail = sail_name
                        break
        self.output_text.setText(optimum_sail)
        self.canvas.draw() # Ensure the canvas is redrawn after updating the plot

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        window = CrossoverChartGUI()
        window.show()
        sys.exit(app.exec())
    except ImportError as ie:
        QMessageBox.critical(None, "Missing Dependency", f"Error: {ie}\nThis application requires the 'scipy' library.")
        sys.exit(1)
    except Exception as e:
        QMessageBox.critical(None, "Application Error", f"An unexpected error occurred: {e}")
        sys.exit(1)