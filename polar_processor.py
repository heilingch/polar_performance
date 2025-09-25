
""" 
Class for handling all polar data processing and calculations.
Author: Christian Heiling
Date: 17.04.2025
Version: 1.5

-------
Revision History:

1.2   ...   Added method to calculate performance degradation based on swell conditions.
1.3   ...   Changes swell model calculation to use wave period instead of frequency.
1.4   ...   Some fixed for saving modified polar data.
1.5   ...   Improved calculation accuracy for low sea state

"""
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd

# Simple functions for small tasks

def extract_filename_from_path(path):
    return path.split("/")[-1]

class PolarProcessor:
    def __init__(self):
        # Initialize empty arrays for our data
        self.twa_values = None  # Will store True Wind Angles
        self.tws_values = None  # Will store True Wind Speeds
        self.speed_matrix = None  # Will store boat speeds
        self.interpolator = None  # Will store our interpolation function
        self.twa_range = None  # Will store the valid TWA range
        self.tws_range = None  # Will store the valid TWS range

    def detect_polar_file_delimiter(self, filename):
        """
        Detects the delimiter of a polar file based on its content.
        Args:
            filename (str): Path to the polar file.
        
        Returns:
            str: Detected delimiter (comma, space, tab, or semicolon).
        """
        # First try to detect the delimiter
        with open(filename, 'r') as f:
            sample = f.readline() + f.readline()  # Read first two lines for better detection
    
        # Check for common delimiters
        potential_delimiters = [',', ';', '\t', ' ']
        delimiter_counts = {delimiter: sample.count(delimiter) for delimiter in potential_delimiters}
    
        # If spaces are used, make sure we're not counting spaces within other data
        if ' ' in delimiter_counts and '\t' in delimiter_counts:
            # If tabs exist and have a reasonable count, prefer them over spaces
            if delimiter_counts['\t'] > 0:
                delimiter_counts.pop(' ')
        
        # Select the delimiter with the highest count
        detected_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
        return detected_delimiter
    
    
    def load_polar_file(self, filename):
        """
        Loads and processes a polar performance file.
        Expected format: CSV with TWS values as columns and TWA values as rows.
        First column should contain TWA values, first row should contain TWS values.
        Automatically detects file delimiter (comma, space, tab, or semicolon).
        """
        try:
            # Detect the delimiter of used polar file
            detected_delimiter = self.detect_polar_file_delimiter(filename)
            print(f"   *INF: Detected delimiter in polar file {filename}: {detected_delimiter}")

            # Read the CSV file
            df = pd.read_csv(filename, delimiter=detected_delimiter, skipinitialspace=True)
            
            # Extract TWA values (first column, skipping header)
            self.twa_values = df.iloc[1:, 0].astype(float).values
            
            # Extract TWS values (first row, skipping first column)
            self.tws_values = df.columns[1:].astype(float).values
            
            # Extract the speed matrix (all data except first row and column)
            self.speed_matrix = df.iloc[1:, 1:].astype(float).values
            
            # Print the data for debugging
            print("Polar file data:")
            print(16*"=")
            print(df)

            # Create the interpolation function
            self.interpolator = RectBivariateSpline(
                self.twa_values, 
                self.tws_values, 
                self.speed_matrix,
                kx=3,  # cubic spline interpolation
                ky=3
            )

            # Store the valid TWA and TWS ranges from loaded data
            self.twa_range = (min(self.twa_values), max(self.twa_values))
            #self.tws_range = (min(self.tws_values), max(self.tws_values))
            self.tws_range = (min(self.tws_values), 50)

            short_filename = extract_filename_from_path(filename)

            return True, f"Polar file {short_filename} loaded successfully"
        except Exception as e:
            return False, f"Error loading polar file: {str(e)}"

    def get_target_speed(self, twa, tws):
        """
        Calculate the target boat speed for given TWA and TWS values.
        
        Args:
            twa (float): True Wind Angle in degrees
            tws (float): True Wind Speed in knots
            
        Returns:
            tuple: (success, result)
                success (bool): Whether the calculation was successful
                result (float or str): Target speed if successful, error message if not
        """
        try:
            # Check if we have loaded polar data
            if self.interpolator is None:
                return False, "**ERR: No valid polar data loaded"

            # Validate input ranges
            if not (self.twa_range[0] <= twa <= self.twa_range[1]):
                return False, f"TWA must be between {self.twa_range[0]} and {self.twa_range[1]} degrees"
            
            if not (self.tws_range[0] <= tws <= self.tws_range[1]):
                return False, f"TWS must be between {self.tws_range[0]} and {self.tws_range[1]} knots"

            # Calculate target speed using our interpolator
            target_speed = float(self.interpolator(twa, tws))
            return True, target_speed

        except Exception as e:
            return False, f"Error calculating target speed: {str(e)}"

    def get_polar_performance(self, twa, tws, boat_speed):
        """
        Calculate the relative performance of the boat based on polar data.
        
        Args:
            twa (float): True Wind Angle in degrees
            tws (float): True Wind Speed in knots
            boat_speed (float): Boat speed in knots
            
        Returns:
            tuple: (success, result)
                success (bool): Whether the calculation was successful
                result (float or str): Performance percentage if successful, error message if not
        """

        # Calculate target speed using our interpolator
        success, target_speed = self.get_target_speed(twa, tws)
        
        # Calculate performance percentage
        if success:
            performance = (boat_speed / target_speed) * 100
            return True, performance
        else:
            return False, 0.0
        

    def generate_polar_plot_data(self, tws):
        """
        Generates data for plotting a polar curve at a specific wind speed.
        
        Args:
            tws (float): True Wind Speed to generate the curve for
            
        Returns:
            tuple: (success, result)
                success (bool): Whether the data generation was successful
                result: Either (angles, speeds) tuple or error message string
        """
        try:
            if self.interpolator is None:
                return False, "No polar data loaded"
                
            # Create array of angles from 0 to 360 degrees
            angles = np.linspace(0, 360, 361)  # One degree steps
            speeds = []
            
            # Calculate speed for each angle
            for angle in angles:
                # Use symmetry: speed at angle is the same as speed at 360 - angle
                if angle <= 180:
                    success, speed = self.get_target_speed(angle, tws)
                else:
                    success, speed = self.get_target_speed(360 - angle, tws)
                
                if success:
                    speeds.append(speed)
                else:
                    speeds.append(0)  # Use 0 for invalid points
                    
            return True, (angles, np.array(speeds))
        
        except Exception as e:
            return False, f"Error generating plot data: {str(e)}"
        
    def calculate_vmg(self, twa, boat_speed):
        """
        Calculate the Velocity Made Good (VMG) for a given angle and boat speed.
        
        Args:
            twa (float): True Wind Angle in degrees
            boat_speed (float): Boat speed in knots
            
        Returns:
            float: VMG value (positive for upwind, negative for downwind)
        """
        # Convert angle to radians for numpy
        twa_rad = np.radians(twa)
        return boat_speed * np.cos(twa_rad)
    
    def tw_2_aw(self, twa, tws, boat_speed):
        """
        Calculate the Apparent Wind Angle (AWA) for a given TWA, TWS, and boat speed.
        Formulas taken from:
        https://orc.org/uploads/files/Rules-Regulations/2025/Speed-Guide-Explanation-2025.pdf
        Args:
            twa (float): True Wind Angle in degrees
            tws (float): True Wind Speed in knots
            boat_speed (float): Boat speed in knots
            mode: 'beat' or 'run'
            
        Returns:
            float: Apparent Wind Angle in degrees
            float: Apparent Wind Speed in knots
        """

        # Convert angles to radians for numpy
        btw = np.radians(twa) # bearing of true wind
        baw = np.arctan((tws*np.sin(btw))/(tws*np.cos(btw) + boat_speed))
            
        # Calculate AWS
        # vaw = np.sqrt(tws**2 + boat_speed**2 - 2 * tws * boat_speed * np.cos(phi_rad))
        vaw = np.sqrt((tws*np.sin(btw))**2 + ((tws*np.cos(btw) + boat_speed)**2))

        # Calculate AWA
        #baw = np.arccos(-1*(tws**2 - aws**2 - boat_speed**2) / (2 * tws * aws)) 
        baw = np.arctan((tws*np.sin(btw))/(tws*np.cos(btw) + boat_speed))
        awa = np.degrees(baw)
        if awa < 0:
            #print("   **DEBUG: Applied wind angle is negative, adding 180 degrees")
            awa = awa + 180
        
        return {'awa': round(awa,2), 'aws': round(vaw,2)}
        
    
    def find_optimal_angles(self, tws):
        """
        Find optimal beating and running angles and their corresponding VMGs.
        
        Args:
            tws (float): True Wind Speed in knots
            
        Returns:
            tuple: (success, result)
                success (bool): Whether calculation was successful
                result: Either dictionary with optimal values or error message
        """
        try:
            if self.interpolator is None:
                return False, "No polar data loaded"
                
            # Initialize variables for tracking optimal values
            best_beat_vmg = 0
            best_beat_angle_twa = 0
            best_beat_angle_awa = 0
            
            best_run_vmg = 0
            best_run_angle_twa = 0
            best_run_angle_awa = 0
            
            # Check angles from 0 to 180 in small increments
            for twa in np.arange(0, 181, 0.5):  # Half-degree steps for precision
                success, boat_speed = self.get_target_speed(twa, tws)
                if not success:
                    continue  # Skip invalid points    
                vmg = self.calculate_vmg(twa, boat_speed)
                
                # Update beating (upwind) optimum
                if 20 <= twa <= 90:  # Reasonable beating angle range
                    if vmg > best_beat_vmg:
                        best_beat_vmg = vmg
                        best_beat_angle_twa = twa
                
                # Update running (downwind) optimum
                if 90 <= twa <= 180:  # Running angle range
                    if -vmg > best_run_vmg:  # Note: downwind VMG is negative
                        best_run_vmg = -vmg  # Store as positive value
                        best_run_angle_twa = twa
            
            # Calculate AWA for best beating angle
            success, best_beat_speed = self.get_target_speed(best_beat_angle_twa, tws)
            if success:
                best_beat_angle_awa = self.tw_2_aw(best_beat_angle_twa, tws, best_beat_speed)['awa']
            
            # Calculate AWA for best running angle
            success, best_run_speed = self.get_target_speed(best_run_angle_twa, tws)
            if success:
                best_run_angle_awa = self.tw_2_aw(best_run_angle_twa, tws, best_run_speed)['awa']

            return True, {
                'beat_angle_twa': best_beat_angle_twa,
                'beat_angle_awa': best_beat_angle_awa,
                'beat_vmg': best_beat_vmg,
                'run_angle_twa': best_run_angle_twa,
                'run_angle_awa': best_run_angle_awa,
                'run_vmg': best_run_vmg
            }
        except Exception as e:
            return False, f"Error finding optimal angles: {str(e)}"
            
        except Exception as e:
            return False, f"Error finding optimal angles: {str(e)}"

    def modify_polar(self, percentage):
        # Create modified version of polar data
        pass
    
    def generate_swell_adjusted_polar(self, swell_model, wave_height, wave_frequency):
        """
        Generate a new polar with performance degraded by swell conditions.
        
        Args:
            swell_model (SwellImpactModel): Configured swell impact model
            wave_height (float): Wave height in meters
            wave_frequency (float): Wave frequency in seconds
            
        Returns:
            tuple: (success, result)
                success (bool): Whether generation was successful
                result: Either new PolarProcessor or error message
        """
        try:
            # Create a copy of self
            new_polar = PolarProcessor()
            
            # Copy over the basic attributes
            new_polar.twa_values = self.twa_values.copy()
            new_polar.tws_values = self.tws_values.copy()
            
            # Create a new speed matrix with adjusted values
            new_speed_matrix = np.zeros_like(self.speed_matrix)
            
            # Apply degradation to each point in the polar
            for i, twa in enumerate(self.twa_values):
                for j, tws in enumerate(self.tws_values):
                    # Get original speed
                    boat_speed = self.speed_matrix[i, j]
                    
                    # Calculate degradation factor
                    degradation = swell_model.calculate_degradation_factor(
                        twa, tws, wave_height, wave_frequency
                    )
                    
                    # Apply degradation
                    new_speed_matrix[i, j] = boat_speed * degradation
            
            # Set the new speed matrix
            new_polar.speed_matrix = new_speed_matrix
            
            # Create new interpolator with the modified data
            new_polar.interpolator = RectBivariateSpline(
                new_polar.twa_values, 
                new_polar.tws_values, 
                new_polar.speed_matrix,
                kx=3,  # cubic spline interpolation
                ky=3
            )
            
            # Copy the range information
            new_polar.twa_range = self.twa_range
            new_polar.tws_range = self.tws_range
            
            return True, new_polar
            
        except Exception as e:
            return False, f"Error generating swell-adjusted polar: {str(e)}"

    def save_adjusted_polar(self, filename):
        """
        Save the current polar data to a file.
        
        Args:
            filename (str): Path to save the polar file
            comment (str, optional): Comment to add to the file header
            
        Returns:
            tuple: (success, message)
        """
        try:
            '''
            # Create DataFrame for saving
            df = pd.DataFrame()
            
            # Add TWA as first column
            df['TWA/TWS'] = np.append(['TWA/TWS'], self.twa_values)
            
            # Add TWS columns with speeds
            for i, tws in enumerate(self.tws_values):
                col_name = str(tws)
                # Create column with header as TWS and values as speeds
                df[col_name] = np.append([col_name], self.speed_matrix[:, i])
            '''
            # Build DataFrame directly from speed matrix
            df = pd.DataFrame(
                self.speed_matrix,
                index=self.twa_values,
                columns=self.tws_values
            )

            # Reset index to make TWA a column
            df.reset_index(inplace=True)
            df.rename(columns={"index": "TWA/TWS"}, inplace=True)


            # Save the resulting polar data to a CSV file
            df.to_csv(filename, index=False)
                
            return True, f"Polar saved successfully to {filename}"
            
        except Exception as e:
            return False, f"Error saving polar file: {str(e)}"
    
    
    def generate_swell_adjusted_polar(self, swell_model, wave_height, wave_period):
        """
        Generate a new polar with performance degraded by swell conditions.
        
        Args:
            swell_model (SwellImpactModel): Configured swell impact model
            wave_height (float): Wave height in meters
            wave_period (float): Wave period in seconds
            
        Returns:
            tuple: (success, result)
                success (bool): Whether generation was successful
                result: Either new PolarProcessor or error message
        """
        try:
            # Create a copy of self
            new_polar = PolarProcessor()
            
            # Copy over the basic attributes
            new_polar.twa_values = self.twa_values.copy()
            new_polar.tws_values = self.tws_values.copy()
            
            # Create a new speed matrix with adjusted values
            new_speed_matrix = np.zeros_like(self.speed_matrix)
            
            # Apply degradation to each point in the polar
            for i, twa in enumerate(self.twa_values):
                for j, tws in enumerate(self.tws_values):
                    # Get original speed
                    boat_speed = self.speed_matrix[i, j]
                    
                    # Calculate degradation factor
                    degradation = swell_model.calculate_degradation_factor(
                        twa, tws, wave_height, wave_period
                    )
                    
                    # Apply degradation
                    new_speed_matrix[i, j] = round(boat_speed * degradation, 1)
            
            # Set the new speed matrix
            new_polar.speed_matrix = new_speed_matrix
            
            # Create new interpolator with the modified data
            new_polar.interpolator = RectBivariateSpline(
                new_polar.twa_values, 
                new_polar.tws_values, 
                new_polar.speed_matrix,
                kx=3,  # cubic spline interpolation
                ky=3
            )
            
            # Copy the range information
            new_polar.twa_range = self.twa_range
            new_polar.tws_range = self.tws_range
            
            return True, new_polar
            
        except Exception as e:
            return False, f"Error generating swell-adjusted polar: {str(e)}"

class SwellImpactModel:
    def __init__(self, boat_params):
        """
        Initialize the swell impact model with boat parameters.
        
        Args:
            boat_params (dict): Dictionary containing boat parameters:
                - length: Boat length in meters
                - beam: Boat beam in meters
                - draft: Boat draft in meters
                - weight: Boat weight in kg
                - type: Optional boat type ('cruiser', 'racer', 'catamaran', etc.)
        """
        self.boat_params = boat_params
        self.custom_beating_factor = 1.0  # Default: no additional degradation
        self.custom_reaching_factor = 1.0  # Default: no additional degradation
        self.custom_running_factor = 1.0   # Default: no additional degradation
        
        # Set boat type factor if provided, otherwise use default
        self.boat_type = boat_params.get('type', 'cruiser')
        
        # Define threshold values for when degradation begins to occur
        # These values are in meters and can be adjusted based on experience
        self.min_wave_threshold = 0.20  # Minimum wave height before any degradation (20cm)
        self.beating_threshold = 0.25   # Minimum threshold for beating mode
        self.reaching_threshold = 0.30  # Minimum threshold for reaching mode
        self.running_threshold = 0.35   # Minimum threshold for running mode
        
    def set_custom_factors(self, beating_factor, reaching_factor, running_factor):
        """
        Set custom adjustment factors based on observations.
        
        Args:
            beating_factor (float): Custom degradation factor for beating (0-1)
            reaching_factor (float): Custom degradation factor for reaching (0-1)
            running_factor (float): Custom degradation factor for running (0-1)
        """
        self.custom_beating_factor = max(0.0, min(1.0, beating_factor))
        self.custom_reaching_factor = max(0.0, min(1.0, reaching_factor))
        self.custom_running_factor = max(0.0, min(1.0, running_factor))
    
    def set_threshold_values(self, min_wave, beating, reaching, running):
        """
        Set custom threshold values for when degradation begins to occur.
        
        Args:
            min_wave (float): Global minimum wave height in meters before any degradation
            beating (float): Minimum wave height in meters for beating degradation
            reaching (float): Minimum wave height in meters for reaching degradation
            running (float): Minimum wave height in meters for running degradation
        """
        self.min_wave_threshold = max(0.0, min_wave)
        self.beating_threshold = max(self.min_wave_threshold, beating)
        self.reaching_threshold = max(self.min_wave_threshold, reaching)
        self.running_threshold = max(self.min_wave_threshold, running)
        
    def calculate_degradation_factor(self, twa, tws, wave_height, wave_period):
        """
        Calculate speed degradation factor based on TWA, TWS, and sea state.
        
        Args:
            twa (float): True Wind Angle in degrees
            tws (float): True Wind Speed in knots
            wave_height (float): Significant wave height in meters
            wave_period (float): Wave period in seconds
            
        Returns:
            float: Degradation factor between 0 and 1
        """
        # Global minimum check - return 1.0 (no degradation) if waves are negligible
        if wave_height < self.min_wave_threshold:
            return 1.0
            
        # Base calculations for different sailing modes based on TWA
        if 0 <= twa <= 60:  # Close hauled/beating
            mode_factor = self._calculate_beating_degradation(twa, tws, wave_height, wave_period)
            mode_factor *= self.custom_beating_factor
            
        elif 60 < twa <= 120:  # Reaching
            mode_factor = self._calculate_reaching_degradation(twa, tws, wave_height, wave_period)
            mode_factor *= self.custom_reaching_factor
            
        else:  # Running/downwind
            mode_factor = self._calculate_running_degradation(twa, tws, wave_height, wave_period)
            mode_factor *= self.custom_running_factor
            
        # Ensure values are between 0 and 1
        return max(0.1, min(1.0, mode_factor))  # Minimum 10% performance
    
    def _apply_threshold(self, wave_height, threshold, factor=1.5):
        """
        Apply non-linear threshold scaling to wave height impact.
        Returns 0 for waves below threshold, then gradually scales up.
        
        Args:
            wave_height (float): Significant wave height in meters
            threshold (float): Wave height threshold in meters
            factor (float): Steepness factor for the curve
            
        Returns:
            float: Scaled wave impact (0 to wave_height)
        """
        import math
        
        if wave_height <= threshold:
            return 0.0
        
        # Apply non-linear scaling above threshold using a sigmoid-like function
        normalized = (wave_height - threshold) / (threshold * factor)
        scaled = normalized / (1.0 + normalized)
        
        # Scale to the original wave height range, with maximum impact at 2x threshold
        return wave_height * scaled
    
    def _calculate_beating_degradation(self, twa, tws, wave_height, wave_period):
        """Calculate degradation factor for beating (private method)"""
        import math
        
        # Return 1.0 (no degradation) if below the beating threshold
        if wave_height < self.beating_threshold:
            return 1.0
            
        # Get boat parameters
        length = self.boat_params['length']
        beam = self.boat_params['beam']
        draft = self.boat_params['draft']
        weight = self.boat_params['weight']
        
        # Calculate length-to-beam ratio (higher is generally better in waves)
        length_beam_ratio = length / beam
        
        # Calculate displacement-to-length ratio (measure of how heavy the boat is for its length)
        displacement_length = weight / (length ** 3)
        
        # Wave steepness factor (bigger impact from steep waves)
        # Wave steepness = height / wavelength
        # Wavelength ≈ 1.56 * T² (where T is period in seconds)
        wavelength = 1.56 * (wave_period ** 2)
        wave_steepness = wave_height / wavelength
        
        # Apply threshold to wave height using non-linear scaling
        effective_wave_height = self._apply_threshold(wave_height, self.beating_threshold)
        
        # Calculate effective wave to length ratio with the thresholded value
        wave_to_length_ratio = effective_wave_height / length
        
        # Base degradation from wave height - more severe for upwind sailing
        # Using quadratic relationship for more realistic impact
        # This applies less degradation for small waves and more for big waves
        base_degradation = 1.0 - (wave_to_length_ratio ** 1.5) * 0.8
        
        # Adjust for boat characteristics
        # Heavier boats with more draft and higher length/beam ratio handle waves better
        boat_factor = min(1.2, (0.8 + 0.1 * length_beam_ratio + 0.1 * draft + 0.1 * (displacement_length / 100)))
        
        # Wind factor - higher winds make wave impact worse due to more heel
        # Progressive reduction with increasing wind, with less impact at low wave heights
        wind_impact_scale = min(1.0, wave_to_length_ratio * 10)  # Scale wind impact by wave size
        wind_factor = max(0.7, 1.0 - (0.015 * tws * wind_impact_scale))
        
        # Adjust for wave period - shorter period waves are more punishing upwind
        period_factor = max(0.7, min(1.0, wave_period / 8))
        
        # Combine factors
        degradation_factor = base_degradation * boat_factor * wind_factor * period_factor
        
        # Wave steepness penalty (steeper waves are harder to handle)
        steepness_factor = 1.0
        if wave_steepness > 0.05:  # Moderately steep
            steepness_impact = (wave_steepness - 0.05) * 3
            # Scale steepness impact by wave size (less impact for small waves)
            steepness_impact *= min(1.0, wave_to_length_ratio * 5)
            steepness_factor = (1.0 - steepness_impact)
            
        degradation_factor *= steepness_factor
        
        return degradation_factor
    
    def _calculate_reaching_degradation(self, twa, tws, wave_height, wave_period):
        """Calculate degradation factor for reaching (private method)"""
        import math
        
        # Return 1.0 (no degradation) if below the reaching threshold
        if wave_height < self.reaching_threshold:
            return 1.0
            
        # Get boat parameters
        length = self.boat_params['length']
        beam = self.boat_params['beam']
        draft = self.boat_params['draft']
        weight = self.boat_params['weight']
        
        # Apply threshold to wave height using non-linear scaling
        effective_wave_height = self._apply_threshold(wave_height, self.reaching_threshold)
        
        # Wave to length ratio with thresholded value
        wave_to_length_ratio = effective_wave_height / length
        
        # Base degradation - less severe than beating, using non-linear relationship
        base_degradation = 1.0 - (wave_to_length_ratio ** 1.3) * 0.5
        
        # Beam factor - wider boats roll more in beam seas
        beam_factor = 1.0 - (0.05 * (beam / length))
        
        # Weight factor - heavier boats roll less
        weight_factor = 0.9 + (0.1 * min(1.0, weight / 10000))
        
        # Wave period impact - waves near boat's natural roll period cause more rolling
        # Estimate natural roll period (simplified formula)
        est_roll_period = 0.8 * beam / math.sqrt(draft)
        
        # Calculate resonance effect (maximum impact when wave period matches boat's natural period)
        period_diff = abs(wave_period - est_roll_period)
        
        # Scale resonance effect by wave size (smaller effect for small waves)
        resonance_scale = min(1.0, wave_to_length_ratio * 5)
        resonance_factor = 1.0 - 0.2 * max(0, 1.0 - (period_diff / est_roll_period)) * resonance_scale
        
        # For slower reaching angles (closer to beating), apply more degradation
        angle_factor = 1.0
        if twa < 90:
            # Linear increase in degradation as we move from 90° to 60°
            angle_factor = 1.0 - (0.2 * (90 - twa) / 30) * min(1.0, wave_to_length_ratio * 3)
        
        # Combine factors
        degradation_factor = base_degradation * beam_factor * weight_factor * resonance_factor * angle_factor
        
        return degradation_factor
    
    def _calculate_running_degradation(self, twa, tws, wave_height, wave_period):
        """Calculate degradation factor for running (private method)"""
        import math
        
        # Return 1.0 (no degradation) if below the running threshold
        if wave_height < self.running_threshold:
            return 1.0
            
        # Get boat parameters
        length = self.boat_params['length']
        beam = self.boat_params['beam']
        draft = self.boat_params['draft']
        weight = self.boat_params['weight']
        
        # Calculate wavelength
        wavelength = 1.56 * (wave_period ** 2)
        
        # Apply threshold to wave height using non-linear scaling
        effective_wave_height = self._apply_threshold(wave_height, self.running_threshold)
        
        # Wave to length ratio with thresholded value
        wave_to_length_ratio = effective_wave_height / length
        
        # Wavelength to boat length ratio (affects surfing potential)
        wavelength_ratio = wavelength / length
        
        # Base degradation
        # For running, smaller waves can improve performance (surfing effect)
        if 1.0 < wavelength_ratio < 2.0 and wave_height > self.running_threshold:
            # Potential improvement from surfing when wavelength is suitable
            # Scale the improvement based on how much above threshold we are
            surf_potential = min(0.3, (wave_height - self.running_threshold) / length * 2)
            
            if wave_to_length_ratio < 0.15:
                # Improvement from surfing when wave height is suitable
                base_degradation = 1.0 + surf_potential
            else:
                # Gradually transition to degradation as waves get larger
                base_degradation = 1.0 + surf_potential - ((wave_to_length_ratio - 0.15) ** 1.5)
        else:
            # Less degradation for small waves using non-linear relationship
            base_degradation = 1.0 - max(0, (wave_to_length_ratio - 0.05)) ** 1.2
        
        # Length-beam ratio factor - longer, narrower boats track better downwind
        lb_ratio = length / beam
        tracking_factor = 0.9 + (0.1 * min(1.0, lb_ratio / 3.5))
        
        # Displacement factor - heavier boats are more stable downwind
        weight_factor = 0.9 + (0.1 * min(1.0, weight / 10000))
        
        # Wind-wave interaction - higher winds + bigger waves = risk of broaching
        # Scale by wave size for more realistic effect at low wave heights
        wind_wave_scale = min(1.0, (wave_height / self.running_threshold - 1) * 3)
        wind_wave_scale = max(0, wind_wave_scale)  # Ensure non-negative
        wind_wave_factor = 1.0 - (0.005 * tws * wave_to_length_ratio * 10) * wind_wave_scale
        
        # Wave period factor - longer period waves are generally easier to handle
        period_factor = max(0.8, min(1.0, wave_period / 6))
        
        # Boat type adjustments
        boat_type_factor = 1.0
        if hasattr(self, 'boat_type'):
            if self.boat_type == 'catamaran':
                # Catamarans generally perform better downwind in waves
                boat_type_factor = 1.1
            elif self.boat_type == 'racer':
                # Racing boats may be more difficult to control in waves
                boat_type_factor = 0.95
        
        # Combine factors
        degradation_factor = base_degradation * tracking_factor * wind_wave_factor * period_factor * weight_factor * boat_type_factor
        
        return degradation_factor
        
    def analyze_factors(self, twa, tws, wave_height, wave_period):
        """
        Analyze and return the individual factors that contribute to the final degradation.
        Useful for debugging and understanding the model.
        
        Returns:
            dict: Dictionary with detailed breakdown of factors
        """
        import math
        
        # Get boat parameters
        length = self.boat_params['length']
        beam = self.boat_params['beam']
        draft = self.boat_params['draft']
        weight = self.boat_params['weight']
        
        result = {
            "input": {
                "twa": twa,
                "tws": tws,
                "wave_height": wave_height,
                "wave_period": wave_period,
                "boat": self.boat_params
            },
            "thresholds": {
                "global_min": self.min_wave_threshold,
                "beating": self.beating_threshold,
                "reaching": self.reaching_threshold,
                "running": self.running_threshold
            },
            "calculated": {
                "wavelength": 1.56 * (wave_period ** 2),
                "wave_to_length_ratio": wave_height / length,
                "length_beam_ratio": length / beam
            },
            "factors": {}
        }
        
        # Calculate wavelength and add to results
        wavelength = 1.56 * (wave_period ** 2)
        result["calculated"]["wavelength_ratio"] = wavelength / length
        result["calculated"]["wave_steepness"] = wave_height / wavelength
        
        # Global minimum check
        if wave_height < self.min_wave_threshold:
            result["factors"]["mode"] = "below_threshold"
            result["factors"]["threshold_applied"] = True
            result["final_degradation_factor"] = 1.0
            return result
        
        # Determine which calculation to use based on TWA
        if 0 <= twa <= 60:  # Beating
            mode = "beating"
            threshold = self.beating_threshold
            custom_factor = self.custom_beating_factor
            
            # Check if below mode-specific threshold
            if wave_height < threshold:
                result["factors"]["mode"] = mode
                result["factors"]["threshold_applied"] = True
                result["final_degradation_factor"] = 1.0
                return result
                
            # Calculate effective wave height after threshold
            effective_wave_height = self._apply_threshold(wave_height, threshold)
            wave_to_length_ratio = effective_wave_height / length
            
            # Add detailed factors from beating calculation
            length_beam_ratio = length / beam
            displacement_length = weight / (length ** 3)
            wave_steepness = wave_height / wavelength
            
            base_degradation = 1.0 - (wave_to_length_ratio ** 1.5) * 0.8
            boat_factor = min(1.2, (0.8 + 0.1 * length_beam_ratio + 0.1 * draft + 0.1 * (displacement_length / 100)))
            
            wind_impact_scale = min(1.0, wave_to_length_ratio * 10)
            wind_factor = max(0.7, 1.0 - (0.015 * tws * wind_impact_scale))
            
            period_factor = max(0.7, min(1.0, wave_period / 8))
            
            result["factors"]["mode"] = mode
            result["factors"]["effective_wave_height"] = effective_wave_height
            result["factors"]["effective_wave_to_length_ratio"] = wave_to_length_ratio
            result["factors"]["base_degradation"] = base_degradation
            result["factors"]["boat_factor"] = boat_factor
            result["factors"]["wind_factor"] = wind_factor
            result["factors"]["wind_impact_scale"] = wind_impact_scale
            result["factors"]["period_factor"] = period_factor
            
            steepness_factor = 1.0
            if wave_steepness > 0.05:
                steepness_impact = (wave_steepness - 0.05) * 3
                steepness_impact *= min(1.0, wave_to_length_ratio * 5)
                steepness_factor = (1.0 - steepness_impact)
            
            result["factors"]["steepness_factor"] = steepness_factor
            
            degradation = base_degradation * boat_factor * wind_factor * period_factor * steepness_factor
            
        elif 60 < twa <= 120:  # Reaching
            mode = "reaching"
            threshold = self.reaching_threshold
            custom_factor = self.custom_reaching_factor
            
            # Check if below mode-specific threshold
            if wave_height < threshold:
                result["factors"]["mode"] = mode
                result["factors"]["threshold_applied"] = True
                result["final_degradation_factor"] = 1.0
                return result
                
            # Calculate effective wave height after threshold
            effective_wave_height = self._apply_threshold(wave_height, threshold)
            wave_to_length_ratio = effective_wave_height / length
            
            # Add detailed factors from reaching calculation
            est_roll_period = 0.8 * beam / math.sqrt(draft)
            
            base_degradation = 1.0 - (wave_to_length_ratio ** 1.3) * 0.5
            beam_factor = 1.0 - (0.05 * (beam / length))
            weight_factor = 0.9 + (0.1 * min(1.0, weight / 10000))
            
            period_diff = abs(wave_period - est_roll_period)
            resonance_scale = min(1.0, wave_to_length_ratio * 5)
            resonance_factor = 1.0 - 0.2 * max(0, 1.0 - (period_diff / est_roll_period)) * resonance_scale
            
            angle_factor = 1.0
            if twa < 90:
                angle_factor = 1.0 - (0.2 * (90 - twa) / 30) * min(1.0, wave_to_length_ratio * 3)
            
            result["factors"]["mode"] = mode
            result["factors"]["effective_wave_height"] = effective_wave_height
            result["factors"]["effective_wave_to_length_ratio"] = wave_to_length_ratio
            result["factors"]["base_degradation"] = base_degradation
            result["factors"]["beam_factor"] = beam_factor
            result["factors"]["weight_factor"] = weight_factor
            result["factors"]["resonance_factor"] = resonance_factor
            result["factors"]["resonance_scale"] = resonance_scale
            result["factors"]["angle_factor"] = angle_factor
            result["factors"]["est_roll_period"] = est_roll_period
            
            degradation = base_degradation * beam_factor * weight_factor * resonance_factor * angle_factor
            
        else:  # Running
            mode = "running"
            threshold = self.running_threshold
            custom_factor = self.custom_running_factor
            
            # Check if below mode-specific threshold
            if wave_height < threshold:
                result["factors"]["mode"] = mode
                result["factors"]["threshold_applied"] = True
                result["final_degradation_factor"] = 1.0
                return result
                
            # Calculate effective wave height after threshold
            effective_wave_height = self._apply_threshold(wave_height, threshold)
            wave_to_length_ratio = effective_wave_height / length
            
            # Add detailed factors from running calculation
            wavelength_ratio = wavelength / length
            lb_ratio = length / beam
            
            if 1.0 < wavelength_ratio < 2.0 and wave_height > threshold:
                surf_potential = min(0.3, (wave_height - threshold) / length * 2)
                
                if wave_to_length_ratio < 0.15:
                    base_degradation = 1.0 + surf_potential
                else:
                    base_degradation = 1.0 + surf_potential - ((wave_to_length_ratio - 0.15) ** 1.5)
            else:
                base_degradation = 1.0 - max(0, (wave_to_length_ratio - 0.05)) ** 1.2
                
            tracking_factor = 0.9 + (0.1 * min(1.0, lb_ratio / 3.5))
            weight_factor = 0.9 + (0.1 * min(1.0, weight / 10000))
            
            wind_wave_scale = min(1.0, (wave_height / threshold - 1) * 3)
            wind_wave_scale = max(0, wind_wave_scale)
            wind_wave_factor = 1.0 - (0.005 * tws * wave_to_length_ratio * 10) * wind_wave_scale
            
            period_factor = max(0.8, min(1.0, wave_period / 6))
            
            boat_type_factor = 1.0
            if hasattr(self, 'boat_type'):
                if self.boat_type == 'catamaran':
                    boat_type_factor = 1.1
                elif self.boat_type == 'racer':
                    boat_type_factor = 0.95
            
            result["factors"]["mode"] = mode
            result["factors"]["effective_wave_height"] = effective_wave_height
            result["factors"]["effective_wave_to_length_ratio"] = wave_to_length_ratio
            result["factors"]["base_degradation"] = base_degradation
            result["factors"]["tracking_factor"] = tracking_factor
            result["factors"]["weight_factor"] = weight_factor
            result["factors"]["wind_wave_factor"] = wind_wave_factor
            result["factors"]["wind_wave_scale"] = wind_wave_scale
            result["factors"]["period_factor"] = period_factor
            result["factors"]["boat_type_factor"] = boat_type_factor
            
            degradation = base_degradation * tracking_factor * wind_wave_factor * period_factor * weight_factor * boat_type_factor
            
        # Apply custom factor
        result["factors"]["raw_degradation"] = degradation
        result["factors"]["custom_factor"] = custom_factor
        final_degradation = max(0.1, min(1.0, degradation * custom_factor))
        result["final_degradation_factor"] = final_degradation
        
        return result