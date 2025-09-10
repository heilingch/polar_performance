#! /bin/bash
sudo systemctl stop gpsd.socket
sudo systemctl stop gpsd.service
sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock

echo "GPDS Server is now running"
