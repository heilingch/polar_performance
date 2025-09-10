import socket
import threading
import time
import math
import re

class UDPNetworkReader:
    """
    Class for reading navigation data from OpenCPN via a network connection (UDP or TCP).
    It parses NMEA0183 sentences to extract Speed Over Ground (SOG), position (Lat/Lon),
    Course Over Ground (COG), fix status, satellite info, HDOP, True Wind Speed (TWS),
    True Wind Direction (TWD), and True Wind Angle (TWA).
    --- MODIFICATION: Added KMPH_TO_KNOTS ---
    """
    MPS_TO_KNOTS = 1.94384  # Conversion factor m/s to knots
    KMPH_TO_KNOTS = 0.539957 # Conversion factor km/h to knots

    def __init__(self, host='0.0.0.0', port=10110, connection_type='udp'):
        """
        Initializes the OpenCPNNetworkReader.

        Args:
            host (str): The host address to bind to (for UDP server) or connect to (for TCP client).
                        '0.0.0.0' listens on all available interfaces.
            port (int): The port number for the NMEA data.
            connection_type (str): 'udp' or 'tcp'. Currently, UDP is primarily implemented.
        """
        self.host = host
        self.port = port
        self.connection_type = connection_type.lower()
        self.sock = None
        self.client_address = None # For UDP, to store sender address if needed

        # Shared data and its lock
        self._lock = threading.Lock()
        self._latest_data = {
            'sog_knots': math.nan,
            'lat': math.nan,
            'lon': math.nan,
            'cog_degrees': math.nan, # Course Over Ground
            'fix_mode': 0,          # 0: No fix, 1: GPS fix, 2: DGPS, 3+ Enhanced
            'hdop': math.nan,
            'satellites_count': 0,
            'tws_knots': math.nan,  # True Wind Speed
            'twd_degrees': math.nan,# True Wind Direction
            'twa_degrees': math.nan, # True Wind Angle
            'last_update_time': 0.0
        }
        
        # --- MODIFICATION START ---
        # Date: 2025-05-24
        # Reason: Add a flag to track if TWA was recently set by WIMWV,T.
        #         This helps prioritize direct TWA over calculated TWA.
        #         We'll use a timestamp. A value of 0 means not set.
        self._twa_mwv_timestamp = 0.0
        self.TWA_MWV_TIMEOUT = 5.0 # Seconds before WIMWV,T is considered stale
        # --- MODIFICATION END ---


        # Polling thread control flag
        self._running = True

        # Initialize network connection
        self._init_socket()

        if self.sock is None:
            print(f"**ERR NET: Failed to initialize {self.connection_type.upper()} socket on {self.host}:{self.port}")
        else:
            print(f"**INFO NET: {self.connection_type.upper()} listener started on {self.host}:{self.port}")
            # Start background thread
            self._thread = threading.Thread(target=self._poll_data_stream)
            self._thread.daemon = True
            self._thread.start()

    def _init_socket(self):
        """Initializes the network socket based on connection_type."""
        try:
            if self.connection_type == 'udp':
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(2.0) # Timeout for recvfrom
                self.sock.bind((self.host, self.port))
            elif self.connection_type == 'tcp':
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0) # Timeout for connect and recv
                self.sock.connect((self.host, self.port))
                print(f"**INFO NET: TCP connection established with {self.host}:{self.port}")
            else:
                raise ValueError("Unsupported connection type. Use 'udp' or 'tcp'.")
        except socket.error as e:
            print(f"**ERR NET: Socket error during initialization: {e}")
            self.sock = None
        except ValueError as e:
            print(f"**ERR NET: {e}")
            self.sock = None


    @staticmethod
    def _nmea_checksum_valid(sentence: str) -> bool:
        """Validates NMEA sentence checksum."""
        sentence = sentence.strip()
        if not sentence.startswith('$') or '*' not in sentence:
            return False
        
        parts = sentence.split('*')
        if len(parts) != 2:
            return False
            
        sentence_body = parts[0][1:] # Remove '$'
        expected_checksum_str = parts[1]
        
        # Calculate checksum
        checksum = 0
        for char in sentence_body:
            checksum ^= ord(char)
        
        try:
            return checksum == int(expected_checksum_str, 16)
        except ValueError:
            return False

    def _parse_nmea_sentence(self, sentence: str):
        """
        Parses a single NMEA0183 sentence and updates internal data.
        """
        if not self._nmea_checksum_valid(sentence):
            # print(f"**WARN NET: Invalid checksum or format: {sentence}")
            return

        fields = sentence.split('*')[0].split(',')
        sentence_id = fields[0][1:] # e.g., GPRMC, WIMWD

        updated_any = False
        with self._lock:
            current_time = time.time() # Get time for potential timestamping
            try:
                if sentence_id.endswith('RMC'): # Recommended Minimum Specific GNSS Data
                    if fields[2] == 'A': # Data valid
                        if fields[3] and fields[4] and fields[5] and fields[6]:
                            self._latest_data['lat'] = self._parse_latlon(fields[3], fields[4])
                            self._latest_data['lon'] = self._parse_latlon(fields[5], fields[6])
                            updated_any = True
                        if fields[7]:
                            self._latest_data['sog_knots'] = float(fields[7])
                            updated_any = True
                        if fields[8]:
                            self._latest_data['cog_degrees'] = float(fields[8])
                            updated_any = True
                        if self._latest_data['fix_mode'] < 1: self._latest_data['fix_mode'] = 1 
                    else: # Void
                        self._latest_data['fix_mode'] = 0
                        self._latest_data['sog_knots'] = math.nan

                elif sentence_id.endswith('VTG'): # Course Over Ground and Ground Speed
                    if len(fields) > 7 and fields[5]:
                        self._latest_data['sog_knots'] = float(fields[5])
                        updated_any = True
                    if len(fields) > 1 and fields[1]:
                        self._latest_data['cog_degrees'] = float(fields[1])
                        updated_any = True
                
                elif sentence_id.endswith('GGA'): # Global Positioning System Fix Data
                    if fields[2] and fields[3] and fields[4] and fields[5]:
                        self._latest_data['lat'] = self._parse_latlon(fields[2], fields[3])
                        self._latest_data['lon'] = self._parse_latlon(fields[4], fields[5])
                        updated_any = True
                    if fields[6]:
                        self._latest_data['fix_mode'] = int(fields[6])
                        updated_any = True
                    if fields[7]:
                        self._latest_data['satellites_count'] = int(fields[7])
                        updated_any = True
                    if fields[8]:
                        self._latest_data['hdop'] = float(fields[8])
                        updated_any = True

                elif sentence_id.endswith('MWD'): # Wind Direction and Speed (often True)
                    # This sentence provides TWD (True Wind Direction) and TWS.
                    # --- MODIFICATION START ---
                    # Date: 2025-05-24
                    # Reason: Ensure MWD *only* sets TWD and TWS. It does *not* provide TWA.
                    if fields[2] == 'T' and fields[1]: # True Wind Direction
                        self._latest_data['twd_degrees'] = float(fields[1])
                        updated_any = True
                    if fields[6] == 'N' and fields[5]: # Wind Speed in Knots
                        self._latest_data['tws_knots'] = float(fields[5])
                        updated_any = True
                    elif fields[8] == 'M' and fields[7]: # Wind Speed in m/s
                        self._latest_data['tws_knots'] = float(fields[7]) * self.MPS_TO_KNOTS
                        updated_any = True
                    # If MWD arrives, TWA might need recalculation. We handle this after parsing.
                    # --- MODIFICATION END ---
                
                elif sentence_id.endswith('MWV'): # Wind Speed and Angle
                    # $WIMWV,angle,R/T,speed,unit,A*CS
                    # R = Relative/Apparent, T = True
                    # Angle is ALWAYS relative to bow (0-359).
                    # --- MODIFICATION START ---
                    # Date: 2025-05-24
                    # Reason: Correctly parse WIMWV. If 'T', it's TWA/TWS. If 'R', it's AWA/AWS.
                    #         Crucially, DO NOT treat the angle as TWD.
                    
                    if len(fields) > 4 and fields[1] and fields[2] and fields[3] and fields[4]:
                        angle_str = fields[1]
                        reference = fields[2]
                        speed_str = fields[3]
                        speed_units = fields[4]
                        
                        speed_val = float(speed_str)
                        current_speed_knots = math.nan
                        if speed_units == 'N':
                            current_speed_knots = speed_val
                        elif speed_units == 'M': # m/s
                            current_speed_knots = speed_val * self.MPS_TO_KNOTS
                        elif speed_units == 'K': # km/h
                            current_speed_knots = speed_val * self.KMPH_TO_KNOTS

                        if reference == 'T' and not math.isnan(current_speed_knots):
                            # This is a TRUE wind sentence, providing TWA and TWS.
                            self._latest_data['tws_knots'] = current_speed_knots
                            
                            twa_0_359 = float(angle_str)
                            # Normalize TWA from 0-359 to -180 to +180
                            twa_normalized = twa_0_359
                            if twa_normalized > 180:
                                twa_normalized -= 360
                            
                            self._latest_data['twa_degrees'] = twa_normalized
                            self._twa_mwv_timestamp = current_time # Mark that we got TWA directly
                            updated_any = True
                        
                        # elif reference == 'R': # Apparent wind (Optional: Add AWA/AWS storage if needed)
                        #     pass
                    # --- MODIFICATION END ---


                # --- MODIFICATION START ---
                # Date: 2025-05-24
                # Reason: Calculate TWA *only if* TWD/COG are available AND TWA wasn't set 
                #         directly and recently by WIMWV,T. This prioritizes direct TWA measurement.
                
                is_mwv_twa_valid = (current_time - self._twa_mwv_timestamp) < self.TWA_MWV_TIMEOUT
                can_calculate = not math.isnan(self._latest_data['twd_degrees']) and \
                                not math.isnan(self._latest_data['cog_degrees'])

                if can_calculate and not is_mwv_twa_valid:
                    # If WIMWV,T is stale or never received, calculate TWA from TWD/COG
                    twa = self._latest_data['twd_degrees'] - self._latest_data['cog_degrees']
                    # Normalize to -180 to +180
                    while twa > 180: twa -= 360
                    while twa <= -180: twa += 360
                    self._latest_data['twa_degrees'] = twa
                    updated_any = True
                elif not can_calculate and not is_mwv_twa_valid:
                     # If we can't calculate and WIMWV,T is stale, TWA is unknown.
                     self._latest_data['twa_degrees'] = math.nan

                # --- MODIFICATION END ---


            except (ValueError, IndexError) as e:
                # print(f"**WARN NET: NMEA parsing error '{sentence_id}': {e} in '{sentence}'")
                pass # Ignore malformed fields/sentences

            if updated_any:
                self._latest_data['last_update_time'] = time.time()

    def _parse_latlon(self, val_str: str, indicator: str) -> float:
        """ Parses NMEA latitude/longitude string (ddmm.mmmm) to decimal degrees. """
        if not val_str: return math.nan
        val = float(val_str)
        degrees = int(val / 100)
        minutes = val % 100
        decimal_degrees = degrees + (minutes / 60.0)
        if indicator in ['S', 'W']:
            decimal_degrees *= -1
        return decimal_degrees

    def _poll_data_stream(self):
        """
        Background thread that polls for NMEA data from the network stream.
        """
        buffer = ""
        while self._running:
            try:
                if self.connection_type == 'udp':
                    data, addr = self.sock.recvfrom(2048) 
                    self.client_address = addr 
                elif self.connection_type == 'tcp':
                    data = self.sock.recv(2048)
                    if not data: 
                        print("**WARN NET: TCP connection closed by peer.")
                        self._running = False 
                        break
                else: 
                    time.sleep(1)
                    continue
                
                buffer += data.decode('ascii', errors='ignore')

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip('\r\n')
                    if line:
                        self._parse_nmea_sentence(line)

            except socket.timeout:
                continue
            except socket.error as e:
                print(f"**ERR NET: Socket error during polling: {e}")
                if self.connection_type == 'tcp':
                    self._running = False 
                time.sleep(1) 
            except Exception as e:
                print(f"**ERR NET: Unexpected error in polling thread: {e}")
                time.sleep(1)
        
        print("**INFO NET: Polling thread stopped.")
        if self.sock and self.connection_type == 'tcp':
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass 
            self.sock.close()
            print("**INFO NET: TCP socket closed.")


    def _print_info(self):
        """
        Prints an informational message with current data.
        """
        now_str = time.strftime("%H:%M:%S", time.localtime(self._latest_data['last_update_time'] if self._latest_data['last_update_time'] > 0 else time.time()))
        with self._lock:
            data_copy = self._latest_data.copy() # Make a copy for safe access

        fix_map = {0: "No Fix", 1: "GPS", 2: "DGPS", 3: "PPS", 4: "RTK", 5: "Float RTK", 6: "Estim", 7: "Manual", 8: "Sim"}
        fix_str = fix_map.get(data_copy['fix_mode'], f"Fix ({data_copy['fix_mode']})")
        
        lat_str = f"{self._format_position(data_copy['lat'])}" if not math.isnan(data_copy['lat']) else "---"
        lon_str = f"{self._format_position(data_copy['lon'], is_lat=False)}" if not math.isnan(data_copy['lon']) else "---" # --- MODIFICATION: Added is_lat=False ---
        pos_str = f"{lat_str}, {lon_str}" if lat_str != "---" else "---"
        
        sog_str = f"{data_copy['sog_knots']:.1f}kn" if not math.isnan(data_copy['sog_knots']) else "---"
        cog_str = f"{data_copy['cog_degrees']:.0f}°" if not math.isnan(data_copy['cog_degrees']) else "---"
        hdop_str = f"{data_copy['hdop']:.1f}" if not math.isnan(data_copy['hdop']) else "---"
        sats_str = f"{data_copy['satellites_count']}" if data_copy['satellites_count'] > 0 else "---"

        tws_str = f"{data_copy['tws_knots']:.1f}kn" if not math.isnan(data_copy['tws_knots']) else "---"
        twd_str = f"{data_copy['twd_degrees']:.0f}°" if not math.isnan(data_copy['twd_degrees']) else "---"
        twa_str = f"{data_copy['twa_degrees']:.0f}°" if not math.isnan(data_copy['twa_degrees']) else "---"

        print(f"{now_str} **NET DATA: Fix: {fix_str}, Sats: {sats_str}, HDOP: {hdop_str}, Pos: {pos_str}")
        print(f"             SOG: {sog_str}, COG: {cog_str}")
        print(f"             TWS: {tws_str}, TWD: {twd_str}, TWA: {twa_str}")

    @staticmethod
    def _format_position(coord_val: float, is_lat: bool = True):
        """
        Converts a decimal degree coordinate into 'DDD MM.MMM' format.
        """
        if math.isnan(coord_val):
            return "---"

        abs_coord = abs(coord_val)
        degrees = int(abs_coord)
        minutes = (abs_coord - degrees) * 60.0
        
        if is_lat:
            direction = 'N' if coord_val >= 0 else 'S'
            return f"{degrees:02d}°{minutes:06.3f}'{direction}"
        else: # Longitude
            direction = 'E' if coord_val >= 0 else 'W'
            return f"{degrees:03d}°{minutes:06.3f}'{direction}"


    def get_latest_data(self) -> dict:
        """ Returns a copy of the latest data dictionary. """
        with self._lock:
            return self._latest_data.copy()

    def get_sog_knots(self) -> tuple[bool, float]:
        """Returns (success, speed_in_knots)."""
        with self._lock:
            sog = self._latest_data['sog_knots']
            fix = self._latest_data['fix_mode'] > 0
        valid = fix and not math.isnan(sog)
        return valid, sog if valid else math.nan

    def get_position(self) -> tuple[bool, float, float]:
        """Returns (success, latitude, longitude)."""
        with self._lock:
            lat = self._latest_data['lat']
            lon = self._latest_data['lon']
            fix = self._latest_data['fix_mode'] > 0
        valid = fix and not math.isnan(lat) and not math.isnan(lon)
        return valid, lat if valid else math.nan, lon if valid else math.nan

    def get_true_wind(self) -> tuple[bool, float, float, float]:
        """Returns (success, tws_knots, twd_degrees, twa_degrees)."""
        with self._lock:
            tws = self._latest_data['tws_knots']
            twd = self._latest_data['twd_degrees']
            twa = self._latest_data['twa_degrees']
            
        # --- MODIFICATION START ---
        # Date: 2025-05-24
        # Reason: Check TWA validity explicitly. TWD might be NaN if only WIMWV,T is available.
        #         Success now means we have TWS and TWA. TWD is a bonus.
        valid = not math.isnan(tws) and not math.isnan(twa)
        # --- MODIFICATION END ---
        return (valid,
                tws if valid else math.nan,
                twd, # Return TWD even if it's NaN, the caller can decide if it's needed
                twa if valid else math.nan)

    def close(self):
        """
        Stops the polling thread and closes the network socket.
        """
        print("**INFO NET: Closing network reader...")
        self._running = False
        
        if self.sock:
            if self.connection_type == 'udp':
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as dummy_sock:
                        dummy_sock.sendto(b'', (self.host if self.host != '0.0.0.0' else '127.0.0.1', self.port))
                except Exception as e:
                    print(f"**WARN NET: Exception while trying to unblock UDP socket: {e}")
            
            if self.connection_type == 'udp' and self.sock:
                # --- MODIFICATION START ---
                # Date: 2025-05-24
                # Reason: Add a try-except block around socket close, as it might already be closed.
                try:
                    self.sock.close()
                    print("**INFO NET: UDP socket closed.")
                except Exception as e:
                    print(f"**WARN NET: Error closing UDP socket: {e}")
                # --- MODIFICATION END ---

        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=5.0) 
            if self._thread.is_alive():
                print("**WARN NET: Polling thread did not terminate cleanly.")
        
        print("**INFO NET: Network reader closed.")

# Example Usage (remains the same)
if __name__ == "__main__":
    print("Starting OpenCPN Network Reader (UDP Server Mode)...")
    ocpn_reader = UDPNetworkReader(host='127.0.0.1', port=10110, connection_type='udp')
    
    if ocpn_reader.sock is None:
        print("Failed to start reader. Exiting.")
        exit()

    try:
        last_print_time = 0
        while True:
            time.sleep(1) 
            current_data = ocpn_reader.get_latest_data()
            if current_data['last_update_time'] > 0 and time.time() - last_print_time > 5:
                ocpn_reader._print_info() 
                last_print_time = time.time()
                
                # --- MODIFICATION: Added a get_true_wind example call ---
                success_wind, tws, twd, twa = ocpn_reader.get_true_wind()
                if success_wind:
                   print(f"             GET_TRUE_WIND: TWS {tws:.1f}kn, TWD {twd:.0f}°, TWA {twa:.0f}°")
                else:
                   print("             GET_TRUE_WIND: No valid TWS/TWA.")
                # --- MODIFICATION END ---


    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")
    finally:
        ocpn_reader.close()
        print("Program terminated.")