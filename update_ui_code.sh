#!/bin/bash

# Path to the UI file
UI_FILE="swell_adjustment.ui"

# Path to the output Python file
OUTPUT_FILE="swell_adjustment_ui.py"

# Check if the UI file exists
if [[ ! -f "$UI_FILE" ]]; then
    echo "Error: $UI_FILE not found!"
    exit 1
fi

# Run pyuic6 to convert the UI file to Python code
echo "Updating UI code..."
pyuic6 -x "$UI_FILE" -o "$OUTPUT_FILE"

# Check if the command was successful
if [[ $? -eq 0 ]]; then
    echo "UI code updated successfully: $OUTPUT_FILE"
else
    echo "Error: Failed to update UI code."
    exit 1
fi