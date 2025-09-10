# Polar Performance Analyzer

**Version:** 5.0
**Author:** Christian Heiling

A PyQt6 desktop application for sailors and naval architects to analyze, visualize, and utilize yacht polar performance data. The tool can calculate target speeds, determine optimal sailing angles (VMG), and assess real-time performance against theoretical data using live data from NMEA sources.

## Key Features

-   **Polar File Parsing**: Loads and interprets standard `.pol` polar data files.
-   **Data Interpolation & Extrapolation**: Creates a complete performance model from the raw polar data.
-   **Interactive Polar Plot**: Visualizes the boat's performance for a given True Wind Speed (TWS) using Matplotlib.
-   **Real-time Performance Analysis**:
    -   Calculates the target boat speed for a given True Wind Angle (TWA) and TWS.
    -   Computes the percentage of polar performance achieved based on actual Speed Over Ground (SOG).
    -   Calculates the expected Apparent Wind Speed (AWS).
-   **VMG Optimization**: Determines the optimal beating and running angles (in both TWA and AWA) and the resulting Velocity Made Good (VMG).
-   **Multiple Data Sources**:
    -   **Manual Entry**: Manually input TWA, TWS, and SOG.
    -   **GPSD**: Connects to a local `gpsd` service to read live SOG.
    -   **UDP Network**: Listens on a UDP port (e.g., for data from OpenCPN, Expedition, or other NMEA broadcasters) for SOG, TWA, and TWS.
-   **Data Filtering**: Includes a configurable median filter to smooth noisy data from GPSD or UDP sources, improving the stability of calculations.
-   **Reference Polar**: Load a second, reference polar file to compare two different performance models on the same plot (e.g., different sail configurations or boats).
-   **Swell Adjustment Tool**: A utility to modify polar data based on sea state.

---

## NMEA Test Software

Included in the `test_software/` directory is a dummy NMEA 0183 data generator (`nmea_dummy_data_generator.py`). This script is essential for testing the UDP network functionality of the main application without needing a live data feed from a boat.

### Test Software Features

-   **Simulated NMEA Stream**: Generates `GPRMC`, `GPGGA`, `GPVTG`, `WIMWD`, and `WIMWV` sentences.
-   **UDP Broadcasting**: Sends the NMEA data stream to a configurable IP address and port, defaulting to `127.0.0.1:10110` to work with the main application out-of-the-box.
-   **Realistic Data**: Simulates gradual changes and random variations in position, speed, course, and wind conditions.
-   **Wind Instrument Simulation**: Includes a `WIND_INSTRUMENTS_ENABLED` switch. Setting this to `False` in the script stops the generation of wind data (`WIMWD`, `WIMWV`), allowing you to test how the Polar Performance Analyzer handles a loss of wind instrument data.

---

## Getting Started

### Requirements

The project is written in Python and requires the following libraries:

-   `PyQt6`
-   `numpy`
-   `matplotlib`
-   `gpsd-py3` (for the GPSD connection)

You can install them using pip:

```bash
pip install PyQt6 numpy matplotlib gpsd-py3
```

### Running the Application

1.  Navigate to the project directory.
2.  Run the main application:
    ```bash
    python polar_performance_v5.py
    ```

### Testing with the NMEA Simulator

1.  In a separate terminal, navigate to the project directory.
2.  Run the NMEA dummy data generator:
    ```bash
    python test_software/nmea_dummy_data_generator.py
    ```
3.  In the Polar Performance Analyzer GUI, select **"UDP Network"** from the "Speed Source" dropdown menu.
4.  The application will now receive and process the simulated data stream. You can test the loss of wind data by stopping the test script, changing `WIND_INSTRUMENTS_ENABLED = False` in `nmea_dummy_data_generator.py`, and restarting it.

---

## Project Structure

-   `polar_performance_v5.py`: The main application file containing the GUI and primary logic.
-   `polar_processor.py`: A class responsible for loading, parsing, and interpolating polar data.
-   `udp_gps_reader.py`: A class for handling incoming NMEA data from a UDP network stream.
-   `gpsd_handler.py`: A class for reading speed data from a `gpsd` service.
-   `swell_adjustment_tool.py`: The GUI and logic for the swell adjustment utility.
-   `test_software/`: Contains the NMEA dummy data generator for testing.
-   `polar_performance.ico`: The application icon.

---

## License

This project is released under the **MIT License**. This means it is free to use, modify, and distribute, even for commercial purposes, as long as the original copyright and license notice are included.

### Support the Project

This software is developed and maintained in my free time. If you find it useful and would like to show your appreciation, please consider making a small donation. It's like buying me a coffee to thank me for the work!

Donations help support the continued development and maintenance of this project.

[**>> Donate via PayPal <<**](https://paypal.me/ChristianHeiling)

Thank you for your support!
