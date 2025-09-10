"""
Class for reading navigation data using Signal K.
Author: [Your Name]
Last modified: 2025-05-09

This class connects to a Signal K server to retrieve navigation data including:
- GPS speed (in knots)
- GPS position (latitude/longitude)
- Fix quality and accuracy
- True wind speed (TWS)
- True wind angle (TWA)
- True wind direction (TWD)

The implementation follows a similar pattern to the GPSSpeedReader but expands
functionality to include wind data needed for polar performance calculations.


# sudo apt install python3-websocket
"""


import threading
import time
import math
import json
import websocket
import requests
from urllib.parse import urljoin


class SignalKReader:
    def __init__(self, server_url="http://localhost:3000"):
        """
        Initialize the Signal K reader.
        
        Args:
            server_url: URL of the Signal K server (default: http://localhost:3000)
        """
        self.server_url = server_url.rstrip('/')
        self.ws_url = None
        self.ws = None
        
        # Shared data and its lock
        self._lock = threading.Lock()
        
        # Navigation data
        self._latest_speed = math.nan  # Speed over ground in knots
        self._fix_mode = 0              # 0: no fix, 1: bad fix, 2: 2D fix, 3: 3D fix
        self._lat = None
        self._lon = None
        self._accuracy = None           # Horizontal position accuracy in meters
        self._satellites_count = None   # Number of satellites used in fix
        
        # Wind data
        self._tws = math.nan           # True wind speed in knots
        self._twa = math.nan           # True wind angle in degrees
        self._twd = math.nan           # True wind direction in degrees
        
        # Polling thread control flag
        self._running = True
        
        try:
            # Connect to Signal K server and discover endpoints
            self._discover_endpoints()
            
            # Start background thread
            self._thread = threading.Thread(target=self._connect_and_listen)
            self._thread.daemon = True
            self._thread.start()
            
        except Exception as e:
            print(f"**ERR SIGNALK: Could not connect to Signal K server: {e}")
            self.ws_url = None
    
    def _discover_endpoints(self):
        """
        Discover Signal K endpoints by querying the server's API.
        Sets the WebSocket URL for data streaming.
        """
        try:
            # Get the endpoints from the Signal K server
            response = requests.get(urljoin(self.server_url, "/signalk"))
            if response.status_code != 200:
                print(f"**ERR SIGNALK: Failed to discover endpoints: {response.status_code}")
                return
            
            endpoints = response.json()
            
            # Find the WebSocket stream endpoint
            if 'endpoints' in endpoints and 'v1' in endpoints['endpoints']:
                v1 = endpoints['endpoints']['v1']
                if 'stream' in v1:
                    self.ws_url = v1['stream']
                    
                    # If the WebSocket URL is relative, make it absolute
                    if self.ws_url.startswith('/'):
                        # Convert http(s):// to ws(s)://
                        if self.server_url.startswith('https'):
                            ws_prefix = 'wss://'
                        else:
                            ws_prefix = 'ws://'
                        
                        server_domain = self.server_url.split('://', 1)[1]
                        self.ws_url = ws_prefix + server_domain + self.ws_url
                    
                    print(f"**INF SIGNALK: WebSocket URL: {self.ws_url}")
                    return
            
            print("**ERR SIGNALK: Could not find v1/stream endpoint")
            
        except Exception as e:
            print(f"**ERR SIGNALK: Error discovering endpoints: {e}")
    
    def _connect_and_listen(self):
        """
        Background thread that connects to the Signal K WebSocket
        and listens for navigation data.
        """
        while self._running:
            try:
                if not self.ws_url:
                    print("**ERR SIGNALK: No WebSocket URL available")
                    time.sleep(5)
                    continue
                
                # Connect to the WebSocket with a subscription request
                subscription = {
                    "context": "vessels.self",
                    "subscribe": [
                        {
                            "path": "navigation.speedOverGround",
                            "period": 1000
                        },
                        {
                            "path": "navigation.position",
                            "period": 1000
                        },
                        {
                            "path": "navigation.gnss.methodQuality",
                            "period": 1000
                        },
                        {
                            "path": "navigation.gnss.horizontalDilution",
                            "period": 1000
                        },
                        {
                            "path": "navigation.gnss.satellitesInView",
                            "period": 1000
                        },
                        {
                            "path": "environment.wind.speedTrue",
                            "period": 1000
                        },
                        {
                            "path": "environment.wind.angleTrueWater",
                            "period": 1000
                        },
                        {
                            "path": "environment.wind.directionTrue",
                            "period": 1000
                        }
                    ]
                }
                
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=lambda ws: ws.send(json.dumps(subscription)),
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                print("**INF SIGNALK: Connecting to Signal K server...")
                self.ws.run_forever()
                
                # If we get here, the connection was closed
                if self._running:
                    print("**INF SIGNALK: Connection closed, reconnecting in 5 seconds...")
                    time.sleep(5)
                
            except Exception as e:
                print(f"**ERR SIGNALK: Error in WebSocket connection: {e}")
                time.sleep(5)
    
    def _on_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        Parse the JSON data and update the navigation and wind information.
        """
        try:
            data = json.loads(message)
            updated = False
            
            with self._lock:
                # Check if it's a delta message
                if "updates" in data:
                    for update in data["updates"]:
                        if "values" in update:
                            for value in update["values"]:
                                # Handle navigation data
                                if value["path"] == "navigation.speedOverGround":
                                    # Convert from m/s to knots
                                    self._latest_speed = value["value"] * 1.94384
                                    updated = True
                                    
                                elif value["path"] == "navigation.position":
                                    if "longitude" in value["value"] and "latitude" in value["value"]:
                                        self._lon = value["value"]["longitude"]
                                        self._lat = value["value"]["latitude"]
                                        updated = True
                                        
                                elif value["path"] == "navigation.gnss.methodQuality":
                                    # Map the Signal K quality value to our fix mode
                                    quality = value["value"]
                                    if quality == "no fix":
                                        self._fix_mode = 0
                                    elif quality == "fix":
                                        self._fix_mode = 2
                                    elif quality == "differential":
                                        self._fix_mode = 3
                                    updated = True
                                    
                                elif value["path"] == "navigation.gnss.horizontalDilution":
                                    # HDOP is related to accuracy but needs conversion
                                    # A rough estimate: accuracy in meters is about HDOP * 5
                                    self._accuracy = value["value"] * 5
                                    updated = True
                                    
                                elif value["path"] == "navigation.gnss.satellitesInView":
                                    self._satellites_count = value["value"]
                                    updated = True
                                
                                # Handle wind data
                                elif value["path"] == "environment.wind.speedTrue":
                                    # Convert from m/s to knots
                                    self._tws = value["value"] * 1.94384
                                    updated = True
                                    
                                elif value["path"] == "environment.wind.angleTrueWater":
                                    # Convert from radians to degrees if needed
                                    if isinstance(value["value"], float) and abs(value["value"]) <= math.pi:
                                        self._twa = math.degrees(value["value"])
                                    else:
                                        self._twa = value["value"]
                                    updated = True
                                    
                                elif value["path"] == "environment.wind.directionTrue":
                                    # Convert from radians to degrees if needed
                                    if isinstance(value["value"], float) and abs(value["value"]) <= math.pi * 2:
                                        self._twd = math.degrees(value["value"])
                                    else:
                                        self._twd = value["value"]
                                    updated = True
            
            if updated:
                self._print_info()
                
        except Exception as e:
            print(f"**ERR SIGNALK: Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"**ERR SIGNALK: WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closing"""
        print(f"**INF SIGNALK: WebSocket connection closed: {close_msg} (code: {close_status_code})")
    
    def _print_info(self):
        """
        Prints an informational message with system time, navigation status,
        satellite count, accuracy, position, and wind data.
        """
        now = time.strftime("%H:%M:%S")
        
        with self._lock:
            if self._fix_mode >= 2 and self._lat is not None and self._lon is not None:
                status = "2D" if self._fix_mode == 2 else "3D" if self._fix_mode == 3 else str(self._fix_mode)
                pos_lat = self._format_position(self._lat)
                pos_lon = self._format_position(self._lon)
                pos_str = f"{pos_lat}, {pos_lon}"
                acc_str = f"{self._accuracy:.2f}" if self._accuracy is not None else "---"
                sats_str = str(self._satellites_count) if self._satellites_count is not None else "---"
            else:
                status = "---"
                sats_str = "---"
                acc_str = "---"
                pos_str = "---"
            
            # Format wind data
            tws_str = f"{self._tws:.2f}" if not math.isnan(self._tws) else "---"
            twa_str = f"{self._twa:.1f}" if not math.isnan(self._twa) else "---"
            twd_str = f"{self._twd:.1f}" if not math.isnan(self._twd) else "---"
        
        print(f"{now} **INF: Status: {status}, Satellites: {sats_str}, Accuracy: {acc_str}, Position: {pos_str}")
        print(f"{now} **INF: Wind - TWS: {tws_str}kt, TWA: {twa_str}°, TWD: {twd_str}°")
    
    @staticmethod
    def _format_position(coord):
        """
        Converts a decimal degree coordinate into 'DDD MM.MMM' format.
        For example: 52.123456 becomes '052 07.407'.
        """
        abs_coord = abs(coord)
        degrees = int(abs_coord)
        minutes = (abs_coord - degrees) * 60
        return f"{degrees:03d} {minutes:06.3f}"
    
    def read_gps_speed(self):
        """
        Returns a tuple (success, speed):
         - success: True if a valid speed has been received.
         - speed: the most recent speed in knots, or math.nan if not available.
        """
        with self._lock:
            speed = self._latest_speed
        return (False, speed) if math.isnan(speed) else (True, speed)
    
    def read_wind_data(self):
        """
        Returns a tuple (success, tws, twa, twd):
         - success: True if valid wind data has been received.
         - tws: True Wind Speed in knots, or math.nan if not available.
         - twa: True Wind Angle in degrees, or math.nan if not available.
         - twd: True Wind Direction in degrees, or math.nan if not available.
        """
        with self._lock:
            tws = self._tws
            twa = self._twa
            twd = self._twd
        
        # Return success if at least wind speed and either angle or direction is available
        success = (not math.isnan(tws)) and (not math.isnan(twa) or not math.isnan(twd))
        return (success, tws, twa, twd)
    
    def close(self):
        """
        Stops the WebSocket connection and background thread.
        """
        self._running = False
        if self.ws:
            self.ws.close()
        if hasattr(self, '_thread'):
            self._thread.join()

'''
# Example usage:
if __name__ == "__main__":
    signalk_reader = SignalKReader()
    try:
        while True:
            time.sleep(5)
            success, speed = signalk_reader.read_gps_speed()
            wind_success, tws, twa, twd = signalk_reader.read_wind_data()
            
            if success:
                print(f"Current speed: {speed:.2f} knots")
            else:
                print("No valid speed reading yet.")
                
            if wind_success:
                print(f"Wind data - TWS: {tws:.2f}kt, TWA: {twa:.1f}°, TWD: {twd:.1f}°")
            else:
                print("No valid wind data yet.")
                
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        signalk_reader.close()

'''