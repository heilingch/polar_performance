""" 
Swell Adjustment Tool
This tool is designed to assist in the adjustment of polar files for sailing boats
based on swell conditions. It provides a user interface for inputting boat parameters,
swell conditions, and custom performance degradation factors. The tool calculates
performance degradation based on the provided inputs and allows for saving and loading
boat configurations and polar files.

Author: Christian Heiling
Date: 17.04.2025
Version: 1.4

-------
Revision History:

1.2   ...   Added method to calculate performance degradation based on swell conditions.
1.3   ...   Added method to read boat parameters from the UI and validate inputs.
1.4   ...   Added method to save polar file
"""


# Standard Library Imports
import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QMainWindow
#from PyQt6 import uic
from swell_adjustment_ui import Ui_SwellAdjustmentMain
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QDir

# Software specific imports
from polar_processor import PolarProcessor
from polar_processor import SwellImpactModel



class SwellAdjustmentTool(QMainWindow, Ui_SwellAdjustmentMain):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connect signals and slots here
        #self.boat_name.textChanged.connect(self.on_boat_name_changed)
        #self.boat_model.textChanged.connect(self.on_boat_model_changed)
        #self.boat_lwl.textChanged.connect(self.on_boat_lwl_changed)
        #self.boat_beam.textChanged.connect(self.on_boat_beam_changed)
        #self.boat_displacement.textChanged.connect(self.on_boat_displacement_changed)
        #self.boat_draft.textChanged.connect(self.on_boat_draft_changed)
        #self.swell_wavehight.textChanged.connect(self.on_swell_waveheight_changed)
        #self.swell_period.textChanged.connect(self.on_swell_period_changed)
        #self.custom_deg_beat.textChanged.connect(self.on_custom_deg_beat_changed)
        #self.custom_deg_run.textChanged.connect(self.on_custom_deg_run_changed)
        self.calc_degradation.clicked.connect(self.on_calc_degradation_clicked)
        
        # Connect menu bar actions
        self.actionOpen_Polar_File.triggered.connect(self.on_action_open_polar_file_triggered)
        self.actionSave_Polar_File_as.triggered.connect(self.on_action_save_polar_file_as_triggered)
        self.actionClose.triggered.connect(self.on_action_close_triggered)
        self.actionLoad_Boat_Configuration.triggered.connect(self.on_action_load_boat_configuration_triggered)
        self.actionSave_Boad_Configuration.triggered.connect(self.on_action_save_boat_configuration_triggered)

        self.polar_processor = PolarProcessor()
        

    def read_boat_parameters(self):
        """
        Read boat parameters from the UI and return them as a dictionary.
        :Return: Dictionary with boat parameters
        """
        boat_params = {}
        length_success, length = self.validate_float_input(self.boat_lwl.text(), "Length")
        if length_success:
            boat_params['length'] = length
        else:
            self.event_log.appendPlainText(length)

        beam_success, beam = self.validate_float_input(self.boat_beam.text(), "Beam")
        if beam_success:
            boat_params['beam'] = beam
        else:
            self.event_log.appendPlainText(beam)

        draft_success, draft = self.validate_float_input(self.boat_draft.text(), "Draft")
        if draft_success:
            boat_params['draft'] = draft
        else:
            self.event_log.appendPlainText(draft)

        weight_success, weight = self.validate_float_input(self.boat_displacement.text(), "Weight")
        if weight_success:
            boat_params['weight'] = weight
        else:
            self.event_log.appendPlainText(weight)
        return boat_params

    # Define general methods
    def validate_float_input(self, text, field_name):
        """Validate that the input can be converted to a float"""
        try:
            value = float(text)
            return True, value
        except ValueError:
            return False, f"Invalid {field_name} value. Please enter a number."
    
    # Define slot methods
  
    def on_calc_degradation_clicked(self):
        
        self.boat_params = self.read_boat_parameters()
        self.swell_impact_model = SwellImpactModel(self.boat_params)

        # Retrieve and validate input values for custom factors
        cust_beat_factor_success, cust_beat_factor = self.validate_float_input(self.custom_deg_beat.text(), "Custom Beating Factor")
        cust_run_factor_success, cust_run_factor = self.validate_float_input(self.custom_deg_run.text(), "Custom Running Factor")
        # Retrieve and validate input values for wind parameters
        twa_success, twa = self.validate_float_input(self.wind_twa.text(), "TWA")
        tws_success, tws = self.validate_float_input(self.wind_tws.text(), "TWS")


        if cust_beat_factor_success:
            self.swell_impact_model.custom_beating_factor = cust_beat_factor
        else:
            self.event_log.appendPlainText("Custom Beating Factor is not a number. Using 1 instead.")
            self.swell_impact_model.custom_beating_factor = 1.0

        if cust_run_factor_success:
            self.swell_impact_model.custom_running_factor = cust_run_factor
        else:
            self.event_log.appendPlainText("Custom Running Factor is not a number. Using 1 instead.")
            self.swell_impact_model.custom_running_factor = 1.0

        # Retrieve and validate input values for swell parameters
        swell_waveheight_success, swell_waveheight = self.validate_float_input(self.swell_wavehight.text(), "Swell Wave Height")
        swell_period_success, swell_period = self.validate_float_input(self.swell_period.text(), "Swell Period")

        if swell_waveheight_success and swell_period_success and twa_success and tws_success:

            k_deg_beating = self.swell_impact_model.calculate_degradation_factor(60, tws, swell_waveheight, swell_period)
            k_deg_reaching = self.swell_impact_model.calculate_degradation_factor(120, tws, swell_waveheight, swell_period)
            k_deg_running = self.swell_impact_model.calculate_degradation_factor(160, tws, swell_waveheight, swell_period)
            k_deg = self.swell_impact_model.calculate_degradation_factor(twa, tws, swell_waveheight, swell_period)
            
            # Write the calculated degradation factors to the UI
            self.K_beat.setText(str(round(k_deg_beating,2)))
            self.K_reach.setText(str(round(k_deg_reaching,2)))
            self.K_run.setText(str(round(k_deg_running,2)))
            self.K_deg.setText(str(round(k_deg,2)))

        else:
            self.event_log.appendPlainText("Invalid input values. Please check your inputs.")


    # Define slot methods for menu bar actions
    def on_action_open_polar_file_triggered(self):
        """Handle the file loading process"""
        pol_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Polar File",
            "",
            "POLAR Files (*.pol);;All Files (*)"
        )
        if pol_file:
            success, message = self.polar_processor.load_polar_file(pol_file)
            
            if success:
                self.event_log.appendPlainText(f"Polar file loaded:\n{pol_file}")
        

    def on_action_save_polar_file_as_triggered(self):
        """
        Handles the "Save Polar File As" action triggered from the UI.
        This function performs the following steps:
        1. Reads and validates boat parameters and swell input values from the UI.
        2. Creates a SwellImpactModel using the boat parameters.
        3. Optionally sets custom degradation factors for the swell model.
        4. Generates a swell-adjusted polar using the polar processor.
        5. Prompts the user to select a save location for the adjusted polar file.
        6. Saves the adjusted polar file to the specified location.
        If any step fails, the function logs an appropriate message to the event log
        and aborts the operation.
        Returns:
            None
        """

        # Check if a polar file is loaded
        if self.polar_processor.interpolator is None:
            self.event_log.appendPlainText("Error: No polar file loaded. Please load a polar file before saving an adjusted version.")
            return
    
        # Step 1: Read and validate all necessary input from UI
        boat_params = self.read_boat_parameters()
        if not boat_params:
            self.event_log.appendPlainText("Aborted: Invalid boat parameters.")
            return

        swell_waveheight_success, wave_height = self.validate_float_input(self.swell_wavehight.text(), "Swell Wave Height")
        swell_period_success, wave_period = self.validate_float_input(self.swell_period.text(), "Swell Period")

        if not (swell_waveheight_success and swell_period_success):
            self.event_log.appendPlainText("Aborted: Invalid swell input values.")
            return

        # Step 2: Create a SwellImpactModel
        swell_model = SwellImpactModel(boat_params)

        # Step 3: Set custom degradation factors (optional)
        beat_success, beat_factor = self.validate_float_input(self.custom_deg_beat.text(), "Custom Beating Factor")
        run_success, run_factor = self.validate_float_input(self.custom_deg_run.text(), "Custom Running Factor")
        swell_model.set_custom_factors(
            beating_factor=beat_factor if beat_success else 1.0,
            reaching_factor=1.0,  # default
            running_factor=run_factor if run_success else 1.0
        )

        # Step 4: Generate the adjusted polar
        success, new_polar = self.polar_processor.generate_swell_adjusted_polar(
            swell_model, wave_height, wave_period
        )
        if not success:
            self.event_log.appendPlainText(f"Failed to generate adjusted polar: {new_polar}")
            return

        # Step 5: Prompt user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Adjusted Polar File",
            "",
            "POLAR Files (*.pol);;All Files (*)"
        )

        if not file_path:
            self.event_log.appendPlainText("Save operation canceled.")
            return

        # Ensure file has correct extension
        if not file_path.lower().endswith(".pol"):
            file_path += ".pol"

        # Step 6: Save the polar to file
        success, message = new_polar.save_adjusted_polar(file_path)
        self.event_log.appendPlainText(message)


    def on_action_close_triggered(self):
        print("Quit action triggered")
        self.close()  # Close the application

    def on_action_load_boat_configuration_triggered(self):
        """
        Load boat configuration from a JSON file via a file dialog.
        Populates input field data, from a JSON file, and logs the operation.
        :Return: None
        """
        print("Load Boat Configuration action triggered")
        
        # Open QFileDialog to select a file to load the boat configuration
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Boat Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
        )

        if not file_path:
            print("Load operation canceled.")
            return

        # Add logic to load a boat configuration
        with open(file_path, 'r') as f:
            boat_config = json.load(f)

        # Populate the UI fields with the loaded configuration
        self.boat_name.setText(boat_config.get('name', ''))
        self.boat_model.setText(boat_config.get('model', ''))
        self.boat_lwl.setText(boat_config.get('lwl', ''))
        self.boat_beam.setText(boat_config.get('beam', ''))
        self.boat_displacement.setText(boat_config.get('displacement', ''))
        self.boat_draft.setText(boat_config.get('draft', ''))

        # Write message to log text box
        self.event_log.appendPlainText(f"Boat configuration loaded from:\n{file_path}")

    def on_action_save_boat_configuration_triggered(self):
        """
        Save boat configuration to a JSON file via a file dialog.
        Collects input field data, saves it as JSON, and logs the operation.
        :Return: None
        """
        
        # Open QFileDialog to define path and file name to save the boat configuration
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Boat Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
        )

        # Ensure the file has a .json suffix
        if file_path and not file_path.endswith(".json"):
            file_path += ".json"

        if not file_path:
            print("Save operation canceled.")
            return

        # Add logic to save a boat configuration
        boat_name = self.boat_name.text()
        boat_model = self.boat_model.text()
        boat_lwl = self.boat_lwl.text()
        boat_beam = self.boat_beam.text()
        boat_displacement = self.boat_displacement.text()
        boat_draft = self.boat_draft.text()

        # Generate a dictionary with the boat configuration
        boat_config = {
            'name': boat_name,
            'model': boat_model,
            'lwl': boat_lwl,
            'beam': boat_beam,
            'displacement': boat_displacement,
            'draft': boat_draft
        }

        # save the config. json to a file
        with open(file_path, 'w') as f:
            json.dump(boat_config, f)

        # Write message to log text box
        self.event_log.appendPlainText(f"Boat configuration saved as:\n{file_path}")


# This block ensures that the script runs only when executed directly,
# and not when imported as a module in another script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SwellAdjustmentTool()
    window.show()
    sys.exit(app.exec())