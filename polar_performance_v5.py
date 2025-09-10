"""
I want to write a program in python which reads a polar performance file from a sailing yacht (polar files define how fast a yacht can go dependent on the wind angle and the wind speed) and does the following things with it:

[X] It reads the file and generates internally an interpolation function
[X] Add possibilities to extrapolate the polar data
[X] Allows to write out extrapolated polar data to a new file
[X] It allows to enter a true wind angle (TWA) and wind speed (TWS) value and generates a polar plot
[X] It allows to enter an actual vessel velocity value and indicated the percentage of polar performance.
[X] It shows the optimum TWA for running and beating
[X] Show the optimum run and beat angles also in AWA
[X] Add AWS to target speed display
[X] What is the meaning of optimum run and beat angles in the ORC?
[X] Allows to write out a modified version of the polar file, which is adapted by a percentage to be used for weather routing under realistic conditions.
[X] Change / optimize text size of the GUI a bit
[X] Structures program in separate files for the GUI and complex classes
[X] Added function to plot second reference polar file (e.g., for different boat or sail configuration)
[X] Fix instability when GPS is active and data is entered manually at the same time
[X] Updated polar processor to allow for different file delimiter formats
[X] Added UDP Network input for SOG, TWS, TWA
[X] Fixed AttributeError for polar_data_loaded
[X] Fix bug with negative TWA coming from UDP
[X] Add Median Filter for incoming UDP/GPSD data (This version addresses this)

Author: Christian Heiling
Date: 2025-05-24
Version: 5.0 (Updated to add Median Filter)
"""

# Standard Library Imports
import sys
import numpy as np
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QLabel, QComboBox,
    QMenuBar, QMenu, QStatusBar, QTextEdit 
)
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt6.QtGui import QFont, QIcon, QAction

# Reason: Added imports for Median Filter implementation.
from collections import deque
import statistics

# Software specific imports
from gpsd_handler import GPSSpeedReader
from polar_processor import PolarProcessor
from swell_adjustment_tool import SwellAdjustmentTool
try:
    from udp_gps_reader import UDPNetworkReader # Assuming udp_gps_reader.py contains the fixed UDPNetworkReader class
except ImportError:
    print("**ERR: Could not import UDPNetworkReader. Make sure 'udp_gps_reader.py' is accessible.")
    UDPNetworkReader = None 


# Definition of constants
GPS_UPDATE_INTERVAL_MS = 1000  # Update GPS speed every second (determines filter sample rate)
UDP_UPDATE_INTERVAL_MS = 1000  # Update UDP data every second (determines filter sample rate)
UDP_DEFAULT_HOST = '127.0.0.1' 
UDP_DEFAULT_PORT = 10110       

# GUI classes

class PolarAnalyzerGUI(QMainWindow): 
    def __init__(self):
        super().__init__()
        self.setup_ui()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polar_performance.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Warning: Icon not found at {icon_path}")

        
        self.setWindowTitle("Polar Performance Analyzer (@ Christian Heiling) V5.0") 
        QApplication.setApplicationName("PolarPerformanceAnalyzer")
        self.polar_processor = PolarProcessor()
        self.ref_polar_processor = PolarProcessor()  
        
        self.main_polar_loaded = False
        self.has_ref_polar = False 
        
        self.gps_speed_reader = None
        self.gps_timer = QTimer()
        self.gps_timer.timeout.connect(self.update_gps_speed)

        self.udp_network_reader = None
        self.udp_timer = QTimer()
        self.udp_timer.timeout.connect(self.update_udp_data)
        

        # Initialize Deques and filter settings for Median Filter.
        self.filter_options = {"None": 1, "15 s": 15, "30 s": 30, "1 min": 60, "5 min": 300}
        self.current_filter_size = 1 # Default to "None" (size 1)
        self.sog_history = deque(maxlen=self.current_filter_size)
        self.tws_history = deque(maxlen=self.current_filter_size)
        self.twa_history = deque(maxlen=self.current_filter_size)

        main_widget = QWidget()
        self.main_layout = QVBoxLayout(main_widget)
        
        self.setFixedSize(1000, 900)
        
        self.setup_menu()
        
        source_filter_section = QHBoxLayout()
        self.speed_source_label = QLabel("Speed Source:")
        self.speed_source_combo = QComboBox()
        self.speed_source_combo.addItems(["Manual Entry", "GPSD", "UDP Network"])
        self.speed_source_combo.setFixedSize(150, 30)
        self.speed_source_combo.currentIndexChanged.connect(self.speed_source_changed)
        
        # Add Filter label and QComboBox. Add both to the HBox.
        source_filter_section.addWidget(self.speed_source_label)
        source_filter_section.addWidget(self.speed_source_combo)

        self.filter_label = QLabel("Filter:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.filter_options.keys())
        self.filter_combo.setFixedSize(100, 30)
        self.filter_combo.currentIndexChanged.connect(self.filter_changed)
        source_filter_section.addWidget(self.filter_label)
        source_filter_section.addWidget(self.filter_combo)
        
        source_filter_section.addStretch()
        self.main_layout.addLayout(source_filter_section)

        input_area = QVBoxLayout()
        input_row = QHBoxLayout()

        twa_container = QVBoxLayout()
        self.twa_input = QLineEdit()
        self.twa_input.setPlaceholderText("True Wind Angle")
        self.twa_label = QLabel("TWA (degrees):")
        twa_container.addWidget(self.twa_label)
        twa_container.addWidget(self.twa_input)

        tws_container = QVBoxLayout()
        self.tws_input = QLineEdit()
        self.tws_input.setPlaceholderText("True Wind Speed")
        self.tws_label = QLabel("TWS (knots):")
        tws_container.addWidget(self.tws_label)
        tws_container.addWidget(self.tws_input)

        boat_speed_container = QVBoxLayout()
        self.boat_speed_input = QLineEdit()
        self.boat_speed_input.setPlaceholderText("Boat Speed (SOG)")
        self.boat_speed_label = QLabel("SOG (knots):")
        boat_speed_container.addWidget(self.boat_speed_label)
        boat_speed_container.addWidget(self.boat_speed_input)

        calc_container = QVBoxLayout()
        calc_container.addWidget(QLabel(""))  
        self.calc_button = QPushButton("Calculate Target Speed")
        self.calc_button.clicked.connect(self.calculate_target_speed)
        calc_container.addWidget(self.calc_button)

        input_row.addLayout(twa_container)
        input_row.addLayout(tws_container)
        input_row.addLayout(boat_speed_container)
        input_row.addLayout(calc_container)
        
        result_row = QHBoxLayout()
        self.result_label = QLabel("Target speed: -")
        result_row.addWidget(self.result_label)

        optimal_angles_layout = QVBoxLayout()
        self.beating_label = QLabel("Optimum beating angle: -")
        self.beating_vmg_label = QLabel("VMG upwind: -")
        self.running_label = QLabel("Optimum running angle: -")
        self.running_vmg_label = QLabel("VMG downwind: -")
        optimal_angles_layout.addWidget(self.beating_label)
        optimal_angles_layout.addWidget(self.beating_vmg_label)
        optimal_angles_layout.addWidget(self.running_label)
        optimal_angles_layout.addWidget(self.running_vmg_label)

        input_area.addLayout(input_row)
        input_area.addLayout(result_row)
        input_area.addLayout(optimal_angles_layout)
        self.main_layout.addLayout(input_area)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.main_layout.addWidget(self.canvas)
        
        self.status_text = QTextEdit()
        self.status_text.setFixedHeight(80)  
        self.status_text.setReadOnly(True)
        self.main_layout.addWidget(self.status_text)
        
        self.log_status("Welcome to Polar Performance Analyzer. Please load a polar file to begin.")
        
        self.setCentralWidget(main_widget)
        
        self.speed_source_changed(0) 

    # Add a static method to calculate the median, handling NaNs.
    @staticmethod
    def calculate_median(data_deque):
        """Calculates the median of the data in the deque, ignoring NaNs."""
        valid_data = [x for x in data_deque if not math.isnan(x)]
        if not valid_data:
            return math.nan
        return statistics.median(valid_data)

    # Add a method to handle changes in the filter selection.
    def filter_changed(self, index):
        """Called when the user changes the filter option."""
        selected_text = self.filter_combo.currentText()
        self.current_filter_size = self.filter_options[selected_text]
        self.log_status(f"Filter set to {selected_text} ({self.current_filter_size} samples).")

        # Create new deques with the updated maxlen and clear old data
        self.sog_history = deque(maxlen=self.current_filter_size)
        self.tws_history = deque(maxlen=self.current_filter_size)
        self.twa_history = deque(maxlen=self.current_filter_size)

    def setup_menu(self):
        menubar = self.menuBar()
        polars_menu = menubar.addMenu("Polars")
        load_polar_action = QAction("Load Polar File", self)
        load_polar_action.triggered.connect(self.load_file)
        polars_menu.addAction(load_polar_action)
        load_ref_action = QAction("Load Reference Polar", self)
        load_ref_action.triggered.connect(self.load_reference_file)
        polars_menu.addAction(load_ref_action)
        tools_menu = menubar.addMenu("Tools")
        swell_tool_action = QAction("Swell Adjustment Tool", self)
        swell_tool_action.triggered.connect(self.open_swell_adjustment_tool)
        tools_menu.addAction(swell_tool_action)

    def log_status(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}]: {message}")
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def speed_source_changed(self, index):
        selected_source = self.speed_source_combo.currentText()
        self.log_status(f"Speed source changed to {selected_source}")

        if self.gps_timer.isActive(): self.gps_timer.stop()
        if self.gps_speed_reader is not None: self.gps_speed_reader.close(); self.gps_speed_reader = None
        if self.udp_timer.isActive(): self.udp_timer.stop()
        if self.udp_network_reader is not None: self.udp_network_reader.close(); self.udp_network_reader = None

        if selected_source == "GPSD":
            self.twa_input.setReadOnly(False) 
            self.tws_input.setReadOnly(False) 
            self.boat_speed_input.setReadOnly(True) 
            self.calc_button.setEnabled(True) 
            
            # Enable filter selection for GPSD mode.
            self.filter_combo.setEnabled(True) 
            if self.gps_speed_reader is None: self.gps_speed_reader = GPSSpeedReader()
            if self.gps_speed_reader.gpsd is not None: self.gps_timer.start(GPS_UPDATE_INTERVAL_MS)
            else: self.log_status("**ERR: Could not connect to GPSD..."); self.boat_speed_input.setText("GPSD Error")

        elif selected_source == "UDP Network":
            self.twa_input.setReadOnly(True)    
            self.tws_input.setReadOnly(True)    
            self.boat_speed_input.setReadOnly(True) 
            self.calc_button.setEnabled(False) 

            # Enable filter selection for UDP mode.
            self.filter_combo.setEnabled(True) 

            if UDPNetworkReader is None: self.log_status("**ERR: UDPNetworkReader not available..."); return
            if self.udp_network_reader is None: self.udp_network_reader = UDPNetworkReader(host=UDP_DEFAULT_HOST, port=UDP_DEFAULT_PORT, connection_type='udp')
            if self.udp_network_reader.sock is not None: self.udp_timer.start(UDP_UPDATE_INTERVAL_MS); self.log_status(f"UDP listener started.")
            else: self.log_status(f"**ERR: Could not start UDP listener."); self.twa_input.setText("UDP Error"); self.tws_input.setText("UDP Error"); self.boat_speed_input.setText("UDP Error")
        
        else: # Manual Entry mode
            self.twa_input.setReadOnly(False)
            self.tws_input.setReadOnly(False)
            self.boat_speed_input.setReadOnly(False)
            self.calc_button.setEnabled(True)
            # Reason: Disable filter and set to "None" for Manual Entry mode.
            self.filter_combo.setEnabled(False)
            self.filter_combo.setCurrentText("None") # Resets filter_changed if needed

    def update_gps_speed(self):
        if self.gps_speed_reader is not None:
            success, speed_kn = self.gps_speed_reader.read_gps_speed()
            if success:
                # Apply median filter to GPS SOG data.
                self.sog_history.append(speed_kn)
                sog_to_use = self.calculate_median(self.sog_history)
                self.boat_speed_input.setText(f"{sog_to_use:.2f}" if not math.isnan(sog_to_use) else "---")

                if self.twa_input.text() and self.tws_input.text():
                    try:
                        float(self.twa_input.text())
                        float(self.tws_input.text())
                        self.calculate_target_speed() 
                    except ValueError: pass # Ignore if TWA/TWS not valid yet
            else:
                self.log_status("Waiting for valid GPS speed data...")
                self.boat_speed_input.setText("---")


    def update_udp_data(self):
        if self.udp_network_reader is not None:
            latest_data = self.udp_network_reader.get_latest_data() 
            sog_valid, sog_knots = self.udp_network_reader.get_sog_knots()
            wind_valid, tws_knots, twd_degrees, twa_degrees = self.udp_network_reader.get_true_wind()

            # Apply median filter to incoming SOG, TWS, TWA UDP data.
            sog_to_use = math.nan
            tws_to_use = math.nan
            twa_to_use = math.nan
            
            # Process SOG
            if sog_valid:
                self.sog_history.append(sog_knots)
                sog_to_use = self.calculate_median(self.sog_history)
            else:
                # If no new data, use last median or NaN if empty
                sog_to_use = self.calculate_median(self.sog_history)

            # Process Wind
            if wind_valid:
                # Convert negative TWA to positive before filtering
                if twa_degrees < 0:
                    twa_degrees = abs(twa_degrees)
                
                self.tws_history.append(tws_knots)
                self.twa_history.append(twa_degrees)

                tws_to_use = self.calculate_median(self.tws_history)
                twa_to_use = self.calculate_median(self.twa_history)
            else:
                # If no new data, use last median or NaN if empty
                tws_to_use = self.calculate_median(self.tws_history)
                twa_to_use = self.calculate_median(self.twa_history)

            # Update QLineEdit fields with filtered values
            self.boat_speed_input.setText(f"{sog_to_use:.2f}" if not math.isnan(sog_to_use) else "---")
            self.tws_input.setText(f"{tws_to_use:.1f}" if not math.isnan(tws_to_use) else "---")
            self.twa_input.setText(f"{twa_to_use:.1f}" if not math.isnan(twa_to_use) else "---")

            # Check if we have enough valid *filtered* data to trigger calculation
            sog_valid_filtered = not math.isnan(sog_to_use)
            wind_valid_filtered = not math.isnan(tws_to_use) and not math.isnan(twa_to_use)

            if sog_valid_filtered and wind_valid_filtered:
                self.calculate_target_speed()
            elif latest_data['last_update_time'] == 0 and self.speed_source_combo.currentText() == "UDP Network": 
                self.log_status("Waiting for initial UDP data...")

    def validate_float_input(self, text, field_name):
        if text == "---" or text == "UDP Error" or text == "GPSD Error": 
            return False, f"{field_name} data not available."
        try:
            value = float(text)
            return True, value
        except ValueError:
            return False, f"Invalid {field_name}: '{text}'."

    def calculate_target_speed(self):
        if not self.main_polar_loaded:
            self.result_label.setText("Load a polar file first.")
            return

        twa_success, twa_value_or_msg = self.validate_float_input(self.twa_input.text(), "TWA")
        if not twa_success: self.result_label.setText(str(twa_value_or_msg)); return

        tws_success, tws_value_or_msg = self.validate_float_input(self.tws_input.text(), "TWS")
        if not tws_success: self.result_label.setText(str(tws_value_or_msg)); return
        
        boat_speed_success, boat_speed_value_or_msg = self.validate_float_input(self.boat_speed_input.text(), "Boat Speed")
        # Allow calculation even if SOG is bad, just won't show performance %.
        boat_speed_value = float(boat_speed_value_or_msg) if boat_speed_success else math.nan

        twa_value = float(twa_value_or_msg) 
        tws_value = float(tws_value_or_msg) 

        success, target_speed_or_msg = self.polar_processor.get_target_speed(twa_value, tws_value)
        
        if not success: 
            self.result_label.setText(str(target_speed_or_msg))
            self.log_status(f"Target speed calc error: {target_speed_or_msg}")
            return

        target_speed = float(target_speed_or_msg) 
        aws_data = self.polar_processor.tw_2_aw(twa_value, tws_value, target_speed)
        aws = aws_data['aws'] if aws_data else math.nan 
        
        polar_performance_str = "N/A"
        if not math.isnan(boat_speed_value):
            perf_success, polar_performance_or_msg = self.polar_processor.get_polar_performance(twa_value, tws_value, boat_speed_value)
            if perf_success: polar_performance_str = f"{round(float(polar_performance_or_msg))}%"
            else: polar_performance_str = "Error"

        aws_str = f"{aws:.2f} kn" if not math.isnan(aws) else "---"
        self.result_label.setText(f"Target speed: {target_speed:.2f} kn (Polar Performance: {polar_performance_str}) " + 10*' ' + f"AWS: {aws_str}")
            
        opt_success, opt_result_or_msg = self.polar_processor.find_optimal_angles(tws_value)
        if opt_success:
            opt_result = opt_result_or_msg 
            self.beating_label.setText(f"   Optimum Beat Angle: TWA: {opt_result['beat_angle_twa']:.1f}°   AWA: {opt_result['beat_angle_awa']:.1f}°")
            self.beating_vmg_label.setText(f"   Beat VMG: {opt_result['beat_vmg']:.2f} kn")
            self.running_label.setText(f"   Optimum Run Angle: TWA: {opt_result['run_angle_twa']:.1f}° AWA: {opt_result['run_angle_awa']:.1f}°")
            self.running_vmg_label.setText(f"   Run VMG: {opt_result['run_vmg']:.2f} kn")
        else: 
            self.log_status(f"Optimal angle calc error: {opt_result_or_msg}")
            self.beating_label.setText("   Optimum Beat Angle: - (Error)")
            self.running_label.setText("   Optimum Run Angle: - (Error)")

        self.update_polar_plot(twa_value, tws_value, boat_speed_value)


    def update_polar_plot(self, twa=None, tws=None, boat_speed=None):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='polar')
            plot_title = 'Polar Performance Plot'

            if not self.main_polar_loaded:
                ax.text(0.5, 0.5, "No polar data loaded", ha='center', va='center', transform=ax.transAxes)
            elif tws is not None and not math.isnan(tws):
                plot_title = f'Polar Performance Plot for TWS {tws:.1f} kn'
                success, result = self.polar_processor.generate_polar_plot_data(tws)
                if success:
                    angles, speeds = result
                    ax.plot(np.radians(angles), speeds, 'b-', label=f'TWS {tws:.1f} kn (Main)')

                if self.has_ref_polar: 
                    ref_success, ref_result = self.ref_polar_processor.generate_polar_plot_data(tws)
                    if ref_success:
                        ref_angles, ref_speeds = ref_result
                        ax.plot(np.radians(ref_angles), ref_speeds, 'gray', linestyle='dashed', label='Reference Polar')

                if twa is not None and not math.isnan(twa):
                    target_success, target_speed_val_or_msg = self.polar_processor.get_target_speed(twa, tws)
                    if target_success:
                        ax.plot(np.radians(twa), float(target_speed_val_or_msg), 'ro', markersize=8, label='Target Speed')

                if twa is not None and not math.isnan(twa) and boat_speed is not None and not math.isnan(boat_speed):
                    ax.plot(np.radians(twa), boat_speed, 'go', markersize=8, label='Actual SOG')
                
                ax.legend(loc='lower left', bbox_to_anchor=(-0.3, -0.15)) 
            else: 
                 ax.text(0.5, 0.5, "Enter TWS to see plot", ha='center', va='center', transform=ax.transAxes)

            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)       
            ax.set_rlabel_position(0); ax.grid(True); ax.set_facecolor('lightyellow')
            ax.set_thetagrids(range(0, 360, 15))
            
            self.figure.suptitle(plot_title, fontsize=14)
            self.figure.tight_layout(rect=[0, 0, 1, 0.96]) 
            self.canvas.draw()
        except Exception as e:
            self.log_status(f"Error updating plot: {str(e)}")


    def load_file(self):
        pol_file, _ = QFileDialog.getOpenFileName(self, "Select Polar File", "", "POLAR Files (*.pol);;All Files (*)")
        if pol_file:
            success, message = self.polar_processor.load_polar_file(pol_file)
            self.main_polar_loaded = success
            self.log_status(message)
            if success: self.log_status(f" * Main polar file {pol_file} loaded"); self.calculate_target_speed()
            self.update_polar_plot()

    def load_reference_file(self):
        ref_file, _ = QFileDialog.getOpenFileName(self, "Select Reference Polar File", "", "POLAR Files (*.pol);;All Files (*)")
        if ref_file:
            success, message = self.ref_polar_processor.load_polar_file(ref_file)
            self.has_ref_polar = success 
            self.log_status(message)
            if success: self.log_status(f" * Reference polar file {ref_file} loaded")
            self.update_polar_plot()

    def open_swell_adjustment_tool(self):
        if not self.main_polar_loaded:
            self.log_status("Cannot open Swell Tool: Main polar file not loaded.")
            return
        self.swell_adjustment_tool = SwellAdjustmentTool() 
        self.swell_adjustment_tool.show()
        
    def closeEvent(self, event):
        self.log_status("Exiting Polar Performance Analyzer...")
        if self.gps_timer.isActive(): self.gps_timer.stop()
        if self.gps_speed_reader is not None: self.gps_speed_reader.close()
        if self.udp_timer.isActive(): self.udp_timer.stop()
        if self.udp_network_reader is not None: self.udp_network_reader.close()
        if hasattr(self, 'swell_adjustment_tool') and self.swell_adjustment_tool.isVisible(): self.swell_adjustment_tool.close()
        event.accept()

# Start the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
    except Exception as e:
        from PyQt6.QtWidgets import QMessageBox 
        QMessageBox.critical(None, "Startup Error", f"Could not initialize: {e}")
        sys.exit(1) 
    sys.exit(app.exec())