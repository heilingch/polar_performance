"""
Class for reading GPS speed data using gpsd.
Author: Christian Heiling
Last modified: 2025-02-02
"""
"""
Class for reading GPS speed data using gpsd.
Author: Christian Heiling
Last modified: 2025-02-03

Todo:
- [ ] Increase the accuracy of the position in the debug message to 3 decimal places.
- [ ] Check if the accuracy value is correct and if it is in meters.

"""



import threading
import time
import select
import math
from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE

class GPSSpeedReader:
    def __init__(self):
        try:
            # Connect to gpsd.
            self.gpsd = gps()  # create gps object
            # Start streaming. This call is required to tell gpsd to begin sending data.
            self.gpsd.stream(WATCH_ENABLE | WATCH_NEWSTYLE)
        except Exception as e:
            print(f"**ERR GPS: Could not connect to GPSD: {e}")
            self.gpsd = None

        self.MPS_TO_KNOTS = 1.94384  # conversion factor from m/s to knots

        # Shared data and its lock.
        self._latest_speed = math.nan
        self._lock = threading.Lock()

        # Additional data fields.
        self._fix_mode = 0          # 0 or 1: no fix; 2: 2D fix; 3: 3D fix.
        self._lat = None
        self._lon = None
        self._accuracy = None       # horizontal error estimate (meters)
        self._satellites_count = None  # number of satellites used for fix

        # Polling thread control flag.
        self._running = True

        if self.gpsd is None:
            print("**ERR GPS: GPSD server isn't running")
        else:
            # Start background thread.
            self._thread = threading.Thread(target=self._poll_gps)
            self._thread.daemon = True
            self._thread.start()

    def _get_socket(self):
        """
        Attempt to retrieve a socket file descriptor from the gpsd object.
        Checks for both 'socket' and '_sock' attributes.
        Returns the socket if found, otherwise None.
        """
        sock = getattr(self.gpsd, 'socket', None)
        if sock is None:
            sock = getattr(self.gpsd, '_sock', None)
        return sock

    def _poll_gps(self):
        """
        Background thread that polls gpsd every 2 seconds.
        It updates speed, fix status, position, accuracy, and satellite count.
        An informational message is printed whenever new data is processed.
        """
        while self._running:
            try:
                if self.gpsd is None:
                    time.sleep(2)
                    continue

                sock = self._get_socket()
                if sock:
                    # If we have a socket, use select with a timeout.
                    r, _, _ = select.select([sock], [], [], 2.0)
                    if not r:
                        # No data available after 2 seconds.
                        continue
                else:
                    # No socket available; simply wait for 2 seconds before polling.
                    time.sleep(2)

                # Try to read the next report. This may block if no data is ready.
                report = self.gpsd.next()
                updated = False
                with self._lock:
                    # Process TPV reports for fix, position, speed, and accuracy.
                    if report.get('class') == 'TPV':
                        if 'speed' in report and report['speed'] is not None:
                            self._latest_speed = report['speed'] * self.MPS_TO_KNOTS
                        if 'lat' in report and 'lon' in report:
                            self._lat = report['lat']
                            self._lon = report['lon']
                        self._fix_mode = report.get('mode', 0)
                        self._accuracy = report.get('eph')
                        updated = True

                    # Process SKY reports for satellite information.
                    elif report.get('class') == 'SKY':
                        sats = report.get('satellites', [])
                        self._satellites_count = sum(1 for s in sats if s.get('used'))
                        updated = True

                if updated:
                    self._print_info()

            except StopIteration:
                # No new data; continue polling.
                continue
            except Exception as e:
                print(f"**ERR GPS: Error reading GPS data: {e}")
                time.sleep(0.5)

    def _print_info(self):
        """
        Prints an informational message with system time, GPS status,
        satellite count, accuracy, and position.
        If no fix is available (fix_mode < 2), missing values are shown as '---'.
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

        print(f"{now} **INF: Status: {status}, Satellites: {sats_str}, Accuracy: {acc_str}, Position: {pos_str}")

    @staticmethod
    def _format_position(coord):
        """
        Converts a decimal degree coordinate into 'DDD MM.MM' format.
        For example: 52.123456 becomes '052 07.41'.
        """
        abs_coord = abs(coord)
        degrees = int(abs_coord)
        minutes = (abs_coord - degrees) * 60
        return f"{degrees:03d} {minutes:05.3f}"

    def read_gps_speed(self):
        """
        Returns a tuple (success, speed):
         - success: True if a valid speed has been received.
         - speed: the most recent speed in knots, or math.nan if not available.
        """
        with self._lock:
            speed = self._latest_speed
        return (False, speed) if math.isnan(speed) else (True, speed)

    def close(self):
        """
        Stops the polling thread and closes the gpsd connection.
        """
        self._running = False
        if self.gpsd is not None:
            try:
                self.gpsd.close()
            except Exception:
                pass
        self._thread.join()

"""
# Example usage:
if __name__ == "__main__":
    gps_reader = GPSSpeedReader()
    try:
        while True:
            time.sleep(5)
            success, speed = gps_reader.read_gps_speed()
            if success:
                print(f"Current speed: {speed:.2f} knots")
            else:
                print("No valid speed reading yet.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        gps_reader.close()
"""