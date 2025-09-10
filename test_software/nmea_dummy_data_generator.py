'''
NMEA 0183 Dummy Data Generator
===============================
This script simulates NMEA 0183 data for GPS and wind instruments, sending it over UDP to a specified host and port.
It generates GPRMC, GPGGA, GPVTG, WIMWD, and WIMWV sentences with realistic but simulated data.
The simulation includes random variations in position, speed, course, wind speed, and wind direction.

Author: Christian Heiling
Revision History:
- v1.0: Initial version
- v1.1: Added wind instrument simulation (allows to enable/disable wind data)

'''

import socket
import time
import math
import random
from datetime import datetime

# --- Configuration ---
TARGET_HOST = '127.0.0.1'  # IP address to send NMEA data to (localhost)
TARGET_PORT = 10110        # Port to send NMEA data to (must match listener)
SEND_INTERVAL = 1.0        # Seconds between sending NMEA sentences
WIND_INSTRUMENTS_ENABLED = False # Set to False to simulate wind instruments being disconnected

# --- Simulation Parameters ---
# Initial values
sim_latitude = 47.0707   # Decimal degrees (e.g., Graz, Austria)
sim_longitude = 15.4395  # Decimal degrees
sim_sog_knots = 5.0      # Speed Over Ground in knots
sim_cog_degrees = 45.0   # Course Over Ground in degrees True
sim_tws_knots = 10.0     # True Wind Speed in knots
sim_twd_degrees = 270.0   # True Wind Direction in degrees True (from North)
sim_mag_variation = -2.5 # Magnetic variation, degrees West (-) or East (+)

# --- Helper Functions ---

def calculate_nmea_checksum(sentence_body: str) -> str:
    """Calculates the NMEA checksum for a sentence body (without '$' or '*')"""
    checksum = 0
    for char in sentence_body:
        checksum ^= ord(char)
    return f"{checksum:02X}" # Return as two-digit hex string

def format_nmea_lat(lat_decimal: float) -> str:
    """Converts decimal latitude to NMEA format (ddmm.mmmm,H)"""
    indicator = 'N' if lat_decimal >= 0 else 'S'
    lat_abs = abs(lat_decimal)
    degrees = int(lat_abs)
    minutes = (lat_abs - degrees) * 60
    return f"{degrees:02d}{minutes:07.4f},{indicator}" # ddmm.mmmm

def format_nmea_lon(lon_decimal: float) -> str:
    """Converts decimal longitude to NMEA format (dddmm.mmmm,H)"""
    indicator = 'E' if lon_decimal >= 0 else 'W'
    lon_abs = abs(lon_decimal)
    degrees = int(lon_abs)
    minutes = (lon_abs - degrees) * 60
    return f"{degrees:03d}{minutes:07.4f},{indicator}" # dddmm.mmmm

# --- NMEA Sentence Generators ---

def create_gprmc(utc_time, lat, lon, sog_knots, cog_degrees, mag_var_deg) -> str:
    """Creates a GPRMC sentence."""
    # $GPRMC,time,status,lat,N/S,lon,E/W,sog,cog,date,mag_var,mag_var_dir*CS
    # Example: $GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
    
    time_str = utc_time.strftime("%H%M%S.%f")[:9] # HHMMSS.ss
    date_str = utc_time.strftime("%d%m%y")
    
    lat_nmea = format_nmea_lat(lat)
    lon_nmea = format_nmea_lon(lon)
    
    sog_str = f"{sog_knots:.1f}"
    cog_str = f"{cog_degrees:.1f}"
    
    mag_var_abs = abs(mag_var_deg)
    mag_var_dir = 'E' if mag_var_deg >= 0 else 'W'
    mag_var_str = f"{mag_var_abs:.1f},{mag_var_dir}"
    
    # Status: A=Active, V=Void
    status = 'A'
    
    body = f"GPRMC,{time_str},{status},{lat_nmea},{lon_nmea},{sog_str},{cog_str},{date_str},{mag_var_str}"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"

def create_gpgga(utc_time, lat, lon, num_sats=8, hdop=1.5, altitude_m=10.0) -> str:
    """Creates a GPGGA sentence."""
    # $GPGGA,time,lat,N/S,lon,E/W,fix_quality,num_sats,hdop,altitude,M,geoid_sep,M,age_dgps,dgps_id*CS
    # Example: $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47

    time_str = utc_time.strftime("%H%M%S.%f")[:9] # HHMMSS.ss
    lat_nmea = format_nmea_lat(lat)
    lon_nmea = format_nmea_lon(lon)
    
    fix_quality = "1" # 1 = GPS fix
    sats_str = f"{num_sats:02d}"
    hdop_str = f"{hdop:.1f}"
    alt_str = f"{altitude_m:.1f}"
    geoid_sep_str = "0.0" # Geoidal separation (dummy)
    
    body = f"GPGGA,{time_str},{lat_nmea},{lon_nmea},{fix_quality},{sats_str},{hdop_str},{alt_str},M,{geoid_sep_str},M,,"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"

def create_gpvtg(cog_true_deg, cog_mag_deg, sog_knots, sog_kmh) -> str:
    """Creates a GPVTG sentence."""
    # $GPVTG,cog_true,T,cog_mag,M,sog_knots,N,sog_kmh,K,mode*CS
    # Example: $GPVTG,054.7,T,,M,005.5,N,010.2,K*48 (Magnetic course often empty if not available)

    cog_t_str = f"{cog_true_deg:.1f}"
    cog_m_str = f"{cog_mag_deg:.1f}" if cog_mag_deg is not None else ""
    sog_n_str = f"{sog_knots:.1f}"
    sog_k_str = f"{sog_kmh:.1f}"
    mode = "A" # A=Autonomous, D=Differential, E=Estimated, N=Not valid, S=Simulator

    body = f"GPVTG,{cog_t_str},T,{cog_m_str},M,{sog_n_str},N,{sog_k_str},K,{mode}"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"

def create_wimwd(twd_true_deg, twd_mag_deg, tws_knots, tws_mps) -> str:
    """Creates a WIMWD sentence (True Wind Direction and Speed)."""
    # $WIMWD,dir_true,T,dir_mag,M,speed_knots,N,speed_mps,M*CS
    # Example: $WIMWD,095.0,T,092.5,M,010.5,N,005.4,M*57

    twd_t_str = f"{twd_true_deg:.1f}"
    twd_m_str = f"{twd_mag_deg:.1f}"
    tws_n_str = f"{tws_knots:.1f}"
    tws_m_str = f"{tws_mps:.1f}"

    body = f"WIMWD,{twd_t_str},T,{twd_m_str},M,{tws_n_str},N,{tws_m_str},M"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"

def create_wimwv_true(twa_deg, tws_knots) -> str:
    """Creates a WIMWV sentence for True Wind (angle relative to bow, speed)."""
    # $WIMWV,wind_angle,reference(R/T),wind_speed,unit(N/K/M),status(A/V)*CS
    # Example for True Wind Angle (TWA) and True Wind Speed (TWS):
    # $WIMWV,110.0,T,12.5,N,A*CS (Here 'T' means the angle and speed are True, not that angle is to True North)
    # This interpretation of MWV for TWA/TWS can vary. Some systems use MWD for TWD/TWS.
    # We'll assume 'T' means the data is True, and the angle is relative to the bow.

    angle_str = f"{abs(twa_deg):.1f}" # Angle is often positive, direction implied by context or other sentences
    ref = "T" # T for True wind data
    speed_str = f"{tws_knots:.1f}"
    unit = "N" # N for knots
    status = "A" # A for Active

    body = f"WIMWV,{angle_str},{ref},{speed_str},{unit},{status}"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"

def create_wimwv_apparent(awa_deg, aws_knots) -> str:
    """Creates a WIMWV sentence for Apparent Wind (angle relative to bow, speed)."""
    # $WIMWV,wind_angle,R,wind_speed,N,A*CS
    angle_str = f"{abs(awa_deg):.1f}"
    ref = "R" # R for Relative (Apparent) wind data
    speed_str = f"{aws_knots:.1f}"
    unit = "N" # N for knots
    status = "A" # A for Active

    body = f"WIMWV,{angle_str},{ref},{speed_str},{unit},{status}"
    checksum = calculate_nmea_checksum(body)
    return f"${body}*{checksum}\r\n"


# --- Main Simulation Loop ---
def run_simulator(host, port, interval):
    """Runs the NMEA simulator, sending data over UDP."""
    global sim_latitude, sim_longitude, sim_sog_knots, sim_cog_degrees
    global sim_tws_knots, sim_twd_degrees, sim_mag_variation

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"NMEA Simulator started. Sending data to {host}:{port} every {interval}s.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            current_utc_time = datetime.utcnow()

            # --- Update simulated values (simple random walk for demonstration) ---
            # Simulate slight position change based on SOG and COG
            # This is a very simplified dead reckoning
            dt_hours = interval / 3600.0
            dist_nm = sim_sog_knots * dt_hours
            # Roughly: 1 degree latitude ~= 60 NM
            # Roughly: 1 degree longitude ~= 60 * cos(latitude) NM
            rad_lat = math.radians(sim_latitude)
            rad_cog = math.radians(sim_cog_degrees)
            
            sim_latitude += (dist_nm / 60.0) * math.cos(rad_cog)
            if abs(sim_latitude) > 90: sim_latitude = math.copysign(90, sim_latitude) # Clamp latitude
            
            # Avoid division by zero at poles for longitude change if using more accurate formula
            if abs(sim_latitude) < 89.99: # Don't calculate if too close to pole
                 sim_longitude += (dist_nm / (60.0 * math.cos(rad_lat))) * math.sin(rad_cog)
            if sim_longitude > 180: sim_longitude -= 360
            if sim_longitude < -180: sim_longitude += 360


            sim_sog_knots += random.uniform(-0.2, 0.2)
            sim_sog_knots = max(0, min(sim_sog_knots, 15.0)) # Clamp SOG

            sim_cog_degrees += random.uniform(-2.0, 2.0)
            sim_cog_degrees %= 360 # Keep COG within 0-359

            sim_tws_knots += random.uniform(-0.3, 0.3)
            sim_tws_knots = max(0, min(sim_tws_knots, 30.0)) # Clamp TWS

            sim_twd_degrees += random.uniform(-3.0, 3.0)
            sim_twd_degrees %= 360 # Keep TWD within 0-359
            
            # --- Derived values ---
            sim_cog_magnetic = (sim_cog_degrees - sim_mag_variation + 360) % 360
            sim_sog_kmh = sim_sog_knots * 1.852 # Knots to km/h
            sim_twd_magnetic = (sim_twd_degrees - sim_mag_variation + 360) % 360
            sim_tws_mps = sim_tws_knots * 0.514444 # Knots to m/s
            
            # Calculate a plausible TWA (True Wind Angle)
            # TWA = TWD - COG. Normalize to -180 to +180.
            twa_calculated = sim_twd_degrees - sim_cog_degrees
            while twa_calculated > 180: twa_calculated -= 360
            while twa_calculated <= -180: twa_calculated += 360

            # Simulate Apparent Wind (very roughly for demo)
            # AWA is usually smaller than TWA when sailing upwind, larger downwind.
            # AWS is usually higher than TWS when sailing upwind/reaching.
            sim_awa_degrees = twa_calculated * random.uniform(0.8, 1.1) 
            sim_aws_knots = sim_tws_knots * random.uniform(0.9, 1.5)
            if sim_sog_knots < 1: # If stationary, AWA approx TWA, AWS approx TWS
                sim_awa_degrees = twa_calculated
                sim_aws_knots = sim_tws_knots


            # --- Generate NMEA Sentences ---
            nmea_gprmc = create_gprmc(current_utc_time, sim_latitude, sim_longitude,
                                      sim_sog_knots, sim_cog_degrees, sim_mag_variation)
            nmea_gpgga = create_gpgga(current_utc_time, sim_latitude, sim_longitude,
                                      num_sats=random.randint(7,12), hdop=random.uniform(0.8, 2.0))
            nmea_gpvtg = create_gpvtg(sim_cog_degrees, sim_cog_magnetic, sim_sog_knots, sim_sog_kmh)

            # --- Send data ---
            full_nmea_packet = nmea_gprmc + nmea_gpgga + nmea_gpvtg
            
            if WIND_INSTRUMENTS_ENABLED:
                nmea_wimwd = create_wimwd(sim_twd_degrees, sim_twd_magnetic, sim_tws_knots, sim_tws_mps)
                # Choose to send True or Apparent wind via MWV, or both if your system expects it.
                # OpenCPN often uses MWD for True and MWV (with 'R') for Apparent.
                # Sending both True and Apparent MWV might be redundant if MWD is also sent.
                # For this example, let's send one MWV for True Wind Angle/Speed and one for Apparent.
                # Note: The 'T' in MWV for true wind angle means the angle is relative to the bow, and the wind data is "True".
                # It does NOT mean the angle is relative to True North (that would be TWD, typically from MWD).
                nmea_wimwv_true = create_wimwv_true(twa_calculated, sim_tws_knots)
                nmea_wimwv_apparent = create_wimwv_apparent(sim_awa_degrees, sim_aws_knots)
                full_nmea_packet += nmea_wimwd + nmea_wimwv_true + nmea_wimwv_apparent
            
            sock.sendto(full_nmea_packet.encode('ascii'), (host, port))
            
            if WIND_INSTRUMENTS_ENABLED:
                wind_info = (f"TWS={sim_tws_knots:.1f}kn, TWD={sim_twd_degrees:.0f}°, TWA={twa_calculated:.0f}°")
            else:
                wind_info = "TWS=---, TWD=---, TWA=---"
            
            print(f"Sent at {current_utc_time.strftime('%H:%M:%S')}: "
                  f"Lat={sim_latitude:.4f}, Lon={sim_longitude:.4f}, SOG={sim_sog_knots:.1f}kn, COG={sim_cog_degrees:.0f}°, "
                  f"{wind_info}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sock.close()
        print("Socket closed.")

if __name__ == "__main__":
    print(f"WIND_INSTRUMENTS_ENABLED = {WIND_INSTRUMENTS_ENABLED}")
    run_simulator(TARGET_HOST, TARGET_PORT, SEND_INTERVAL)