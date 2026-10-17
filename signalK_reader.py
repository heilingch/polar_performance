"""
Class for reading navigation and wind data using Signal K.
Author: Christian Heiling
Last modified: 2026-10-17
Version: 7.0 (Native Signal K Support)

This class connects to a Signal K server to retrieve navigation and wind data including:
- GPS speed (in knots)
- Course over ground (COG in degrees)
- GPS position (latitude/longitude)
- Fix quality, satellites count, and accuracy
- True wind speed (TWS in knots)
- True wind angle (TWA in degrees)
- True wind direction (TWD in degrees)

The implementation matches the interface used by UDPNetworkReader and GPSSpeedReader
for seamless integration into the Polar Performance Analyzer.

Dependencies:
    pip install requests websocket-client
    (or: sudo apt install python3-requests python3-websocket)
"""

import threading
import time
import math
import json
from urllib.parse import urljoin, urlparse, urlunparse
import websocket
import requests


class SignalKReader:
    MPS_TO_KNOTS = 1.94384  # conversion factor from m/s to knots

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
        self._fix_mode = 0             # 0: no fix, 1: bad fix, 2: 2D fix, 3: 3D fix
        self._lat = math.nan
        self._lon = math.nan
        self._cog = math.nan           # Course over ground in degrees
        self._heading_true = math.nan  # True heading in degrees
        self._accuracy = math.nan      # Horizontal position accuracy in meters
        self._satellites_count = 0     # Number of satellites used in fix
        
        # Wind data
        self._tws = math.nan           # True wind speed in knots
        self._twa = math.nan           # True wind angle in degrees (-180 to +180)
        self._twd = math.nan           # True wind direction in degrees (0 to 360)
        
        # Server identity tracking (to filter out AIS targets)
        self._self_urn = None
        
        # Timestamp of last data update
        self._last_update_time = 0.0
        
        # Polling thread control flag
        self._running = True
        
        # Discover endpoints and start background listener thread
        try:
            self._discover_endpoints()
        except Exception as e:
            print(f"**WARN SIGNALK: Initial discovery failed ({e}), will retry in background.")
            self.ws_url = None
        
        self._thread = threading.Thread(target=self._connect_and_listen)
        self._thread.daemon = True
        self._thread.start()

    def _resolve_ws_url(self, ws_stream):
        """
        Converts a relative or absolute WebSocket stream path into a fully qualified ws:// or wss:// URL.
        """
        if ws_stream.startswith('/'):
            ws_prefix = 'wss://' if self.server_url.startswith('https') else 'ws://'
            server_domain = self.server_url.split('://', 1)[1]
            return ws_prefix + server_domain + ws_stream
        
        # If the URL is absolute, ensure the hostname matches server_url's host if ws_stream returned localhost
        parsed_server = urlparse(self.server_url)
        parsed_ws = urlparse(ws_stream)
        
        if parsed_server.hostname not in ('localhost', '127.0.0.1', None) and parsed_ws.hostname in ('localhost', '127.0.0.1'):
            ws_netloc = f"{parsed_server.hostname}:{parsed_ws.port or parsed_server.port or 3000}"
            return urlunparse((parsed_ws.scheme, ws_netloc, parsed_ws.path, parsed_ws.params, parsed_ws.query, parsed_ws.fragment))
        
        return ws_stream

    def _discover_endpoints(self):
        """
        Discover Signal K endpoints by querying the server's API.
        Signal K standard specifies 'signalk-ws' in endpoints['v1'].
        """
        try:
            response = requests.get(urljoin(self.server_url, "/signalk"), timeout=2.0)
            if response.status_code == 200:
                endpoints = response.json()
                
                # Check for Signal K v1 endpoint specification ('signalk-ws')
                if 'endpoints' in endpoints and 'v1' in endpoints['endpoints']:
                    v1 = endpoints['endpoints']['v1']
                    # Look for standard 'signalk-ws', with fallbacks to 'stream' or 'ws'
                    ws_stream = v1.get('signalk-ws') or v1.get('stream') or v1.get('ws')
                    if ws_stream:
                        self.ws_url = self._resolve_ws_url(ws_stream)
                        print(f"**INF SIGNALK: WebSocket URL discovered: {self.ws_url}")
                        return

            print(f"**WARN SIGNALK: Discovery at {self.server_url}/signalk did not provide 'signalk-ws', using default endpoint path.")
        except Exception as e:
            print(f"**WARN SIGNALK: Error during endpoint discovery: {e}, falling back to default stream URL.")

        # Default fallback stream endpoint
        parsed = urlparse(self.server_url)
        scheme = 'wss' if parsed.scheme == 'https' else 'ws'
        netloc = parsed.netloc or f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 3000}"
        self.ws_url = f"{scheme}://{netloc}/signalk/v1/stream"
        print(f"**INF SIGNALK: Using fallback WebSocket URL: {self.ws_url}")

    def _on_open(self, ws):
        """
        Called when WebSocket connection is opened.
        Subscribes to navigation and wind paths for the local vessel.
        """
        print(f"**INF SIGNALK: WebSocket connection opened to {self.ws_url}")
        try:
            # Subscribing to wildcard navigation.* and environment.wind.*
            subscription = {
                "context": "vessels.self",
                "subscribe": [
                    {
                        "path": "navigation.*",
                        "period": 1000
                    },
                    {
                        "path": "environment.wind.*",
                        "period": 1000
                    }
                ]
            }
            ws.send(json.dumps(subscription))
        except Exception as e:
            print(f"**WARN SIGNALK: Error sending subscription: {e}")

    def _connect_and_listen(self):
        """
        Background thread that connects to the Signal K WebSocket
        and listens for navigation and wind data.
        """
        while self._running:
            try:
                if not self.ws_url:
                    self._discover_endpoints()
                    if not self.ws_url:
                        time.sleep(3)
                        continue
                
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                print(f"**INF SIGNALK: Connecting to Signal K server at {self.ws_url}...")
                self.ws.run_forever()
                
                if self._running:
                    print("**INF SIGNALK: Connection closed, reconnecting in 3 seconds...")
                    time.sleep(3)
                
            except Exception as e:
                if self._running:
                    print(f"**ERR SIGNALK: Error in WebSocket connection: {e}")
                    time.sleep(3)

    def _on_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        Parse the JSON delta data and update navigation and wind information.
        """
        try:
            data = json.loads(message)
            now = time.time()
            
            # Handle server hello message
            if "self" in data and isinstance(data["self"], str):
                self._self_urn = data["self"]

            # Filter out updates intended for other vessels (e.g. AIS targets)
            msg_context = data.get("context")
            if msg_context and self._self_urn and msg_context not in ("vessels.self", self._self_urn):
                return

            updated = False
            with self._lock:
                if "updates" in data:
                    for update in data["updates"]:
                        if "values" in update:
                            for item in update["values"]:
                                path = item.get("path")
                                val = item.get("value")
                                
                                if val is None:
                                    continue
                                
                                # SOG: m/s -> knots
                                if path == "navigation.speedOverGround":
                                    if isinstance(val, (int, float)):
                                        self._latest_speed = float(val) * self.MPS_TO_KNOTS
                                        if self._fix_mode < 1:
                                            self._fix_mode = 1
                                        updated = True
                                
                                # COG: radians -> degrees (0-360)
                                elif path == "navigation.courseOverGroundTrue":
                                    if isinstance(val, (int, float)):
                                        self._cog = (math.degrees(float(val))) % 360
                                        updated = True

                                # Heading True: radians -> degrees (0-360)
                                elif path == "navigation.headingTrue":
                                    if isinstance(val, (int, float)):
                                        self._heading_true = (math.degrees(float(val))) % 360
                                        updated = True
                                
                                # Position
                                elif path == "navigation.position":
                                    if isinstance(val, dict):
                                        lat = val.get("latitude")
                                        lon = val.get("longitude")
                                        if lat is not None and lon is not None:
                                            self._lat = float(lat)
                                            self._lon = float(lon)
                                            if self._fix_mode < 1:
                                                self._fix_mode = 1
                                            updated = True
                                
                                # GNSS fix quality
                                elif path == "navigation.gnss.methodQuality":
                                    if isinstance(val, str):
                                        q = val.lower()
                                        if "no" in q:
                                            self._fix_mode = 0
                                        elif "diff" in q or "dgps" in q:
                                            self._fix_mode = 3
                                        else:
                                            self._fix_mode = 2
                                    elif isinstance(val, (int, float)):
                                        self._fix_mode = int(val)
                                    updated = True
                                
                                # HDOP / Horizontal dilution
                                elif path == "navigation.gnss.horizontalDilution":
                                    if isinstance(val, (int, float)):
                                        self._accuracy = float(val) * 5.0
                                        updated = True
                                
                                # Satellites
                                elif path in ("navigation.gnss.satellites", "navigation.gnss.satellitesInView"):
                                    if isinstance(val, (int, float)):
                                        self._satellites_count = int(val)
                                        updated = True
                                    elif isinstance(val, dict) and "count" in val:
                                        self._satellites_count = int(val["count"])
                                        updated = True
                                
                                # TWS: m/s -> knots
                                elif path == "environment.wind.speedTrue":
                                    if isinstance(val, (int, float)):
                                        self._tws = float(val) * self.MPS_TO_KNOTS
                                        updated = True
                                
                                # TWA: radians -> degrees (-180 to +180)
                                elif path in ("environment.wind.angleTrueWater", "environment.wind.angleTrueGround"):
                                    if isinstance(val, (int, float)):
                                        deg = math.degrees(float(val))
                                        # Normalize to -180 to +180
                                        while deg > 180:
                                            deg -= 360
                                        while deg <= -180:
                                            deg += 360
                                        self._twa = deg
                                        updated = True
                                
                                # TWD: radians -> degrees (0 to 360)
                                elif path == "environment.wind.directionTrue":
                                    if isinstance(val, (int, float)):
                                        self._twd = (math.degrees(float(val))) % 360
                                        updated = True
                
                # If TWD is not provided directly by the server, calculate it from COG or Heading + TWA
                if math.isnan(self._twd) and not math.isnan(self._twa):
                    ref_angle = self._heading_true if not math.isnan(self._heading_true) else self._cog
                    if not math.isnan(ref_angle):
                        self._twd = (ref_angle + self._twa) % 360

                if updated:
                    self._last_update_time = now

        except Exception as e:
            print(f"**ERR SIGNALK: Error processing message: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        if self._running:
            print(f"**ERR SIGNALK: WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closing"""
        if self._running:
            print(f"**INF SIGNALK: WebSocket connection closed: {close_msg} (code: {close_status_code})")

    def get_latest_data(self) -> dict:
        """
        Returns a dictionary of all current data (matches UDPNetworkReader interface).
        """
        with self._lock:
            return {
                'sog_knots': self._latest_speed,
                'lat': self._lat,
                'lon': self._lon,
                'cog_degrees': self._cog,
                'fix_mode': self._fix_mode,
                'accuracy': self._accuracy,
                'satellites_count': self._satellites_count,
                'tws_knots': self._tws,
                'twa_degrees': self._twa,
                'twd_degrees': self._twd,
                'last_update_time': self._last_update_time
            }

    def get_sog_knots(self) -> tuple[bool, float]:
        """
        Returns (success, speed_in_knots).
        Matches UDPNetworkReader interface.
        """
        with self._lock:
            sog = self._latest_speed
            valid = not math.isnan(sog) and (self._fix_mode > 0 or not math.isnan(sog))
        return valid, sog if valid else math.nan

    def get_true_wind(self) -> tuple[bool, float, float, float]:
        """
        Returns (success, tws_knots, twd_degrees, twa_degrees).
        Matches UDPNetworkReader interface.
        """
        with self._lock:
            tws = self._tws
            twd = self._twd
            twa = self._twa
        valid = not math.isnan(tws) and not math.isnan(twa)
        return (valid,
                tws if valid else math.nan,
                twd if not math.isnan(twd) else math.nan,
                twa if valid else math.nan)

    def read_gps_speed(self) -> tuple[bool, float]:
        """
        Legacy method for GPSSpeedReader compatibility.
        Returns (success, speed_in_knots).
        """
        return self.get_sog_knots()

    def read_wind_data(self) -> tuple[bool, float, float, float]:
        """
        Legacy method returning (success, tws, twa, twd).
        """
        with self._lock:
            tws = self._tws
            twa = self._twa
            twd = self._twd
        valid = not math.isnan(tws) and not math.isnan(twa)
        return (valid, tws, twa, twd)

    def close(self):
        """
        Stops the WebSocket connection and background thread cleanly.
        """
        self._running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("**INF SIGNALK: Signal K reader closed.")


if __name__ == "__main__":
    signalk_reader = SignalKReader()
    try:
        print("Listening for Signal K data (Ctrl+C to stop)...")
        while True:
            time.sleep(2)
            sog_ok, sog = signalk_reader.get_sog_knots()
            wind_ok, tws, twd, twa = signalk_reader.get_true_wind()
            
            sog_str = f"{sog:.2f} kn" if sog_ok else "---"
            wind_str = f"TWS {tws:.1f} kn, TWA {twa:.1f}°, TWD {twd:.1f}°" if wind_ok else "---"
            print(f"[SignalK] SOG: {sog_str} | Wind: {wind_str}")
            
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        signalk_reader.close()