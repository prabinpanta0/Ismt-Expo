import serial
import time
import numpy as np
from scipy.signal import butter, filtfilt

SERIAL_PORT = 'COM3'    # adjust as needed
BAUD_RATE = 9600
SAMPLE_RATE = 500
BASELINE_DURATION = 60  # Ensure this is long enough to collect sufficient data

def butter_bandpass(lowcut, highcut, fs, order=2):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500):
    b, a = butter_bandpass(lowcut, highcut, fs)
    # If data is too short, filtfilt will raise an error. Ensure enough samples:
    # filtfilt by default needs at least padlen samples (default padlen=3*(max(len(a),len(b)))=15).
    # Check if we have enough data
    if len(data) <= 15:
        # Not enough data, either skip filtering or return data as-is.
        print("Warning: Not enough data to apply filtfilt. Returning unfiltered data.")
        return data
    # If we have enough data, apply filtfilt
    y = filtfilt(b, a, data)
    return y

# ---------------------
# Data Acquisition for Baseline
# ---------------------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for Arduino reset if needed

print("Recording baseline... please remain calm and still.")
start_time = time.time()
baseline_values = []
baseline_timestamps = []

while True:
    if time.time() - start_time >= BASELINE_DURATION:
        # Stop after baseline duration
        break
    line_in = ser.readline().strip()
    if not line_in:
        continue
    try:
        ecg_value = int(line_in.decode('utf-8'))
    except ValueError:
        continue
    current_time = time.time() - start_time
    baseline_values.append(ecg_value)
    baseline_timestamps.append(current_time)

num_samples = len(baseline_values)
print(f"Baseline data collected: {num_samples} samples")

if num_samples <= 15:
    # Not enough data to filter or analyze
    print("Not enough baseline data collected. Increase BASELINE_DURATION or check your setup.")
else:
    # Apply bandpass filter
    baseline_filtered = bandpass_filter(baseline_values, lowcut=0.5, highcut=40, fs=SAMPLE_RATE)
    # Continue with your processing (e.g., detecting R-peaks, computing HR, HRV, etc.)
    # ...
    
ser.close()
