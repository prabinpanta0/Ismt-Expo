import serial
import numpy as np
import time
from scipy.stats import zscore, skew, kurtosis
from sklearn.preprocessing import StandardScaler

# Configure serial port
ARDUINO_PORT = 'COM5'  # Update to your Arduino's port
BAUD_RATE = 9600

# Number of samples to record
BASELINE_DURATION = 30  # Duration for baseline data collection
QUESTION_DURATION = 10  # Duration for question response collection

# Initialize data arrays
baseline_data = []

def collect_data(ser, duration):
    """Generic function to collect ECG data."""
    start_time = time.time()
    data = []
    while time.time() - start_time < duration:
        try:
            raw_data = ser.readline().decode('utf-8').strip()
            print(f"Raw data: {raw_data}")  # Diagnostic print
            if raw_data.isdigit():
                ecg_value = int(raw_data)
                data.append(ecg_value)
            else:
                print("Non-numeric or invalid data received.")
        except Exception as e:
            print("Error reading data:", e)
    return data

def collect_baseline(ser, duration):
    """Collect baseline data for a given duration."""
    print(f"Collecting baseline data for {duration} seconds. Please remain relaxed.")
    baseline = collect_data(ser, duration)
    print("Baseline data collection complete.")
    return baseline

def analyze_response(question_data, baseline_data):
    """Advanced analysis of response data against baseline."""
    # Preprocessing and scaling
    scaler = StandardScaler()
    scaled_baseline = scaler.fit_transform(np.array(baseline_data).reshape(-1, 1)).flatten()
    scaled_question = scaler.transform(np.array(question_data).reshape(-1, 1)).flatten()

    # Compute baseline metrics
    baseline_mean = np.mean(scaled_baseline)
    baseline_std = np.std(scaled_baseline)
    baseline_skewness = skew(scaled_baseline)
    baseline_kurtosis = kurtosis(scaled_baseline)

    # Compute question metrics
    question_mean = np.mean(scaled_question)
    question_std = np.std(scaled_question)
    question_skewness = skew(scaled_question)
    question_kurtosis = kurtosis(scaled_question)

    # Deviation metrics
    deviation_mean = abs(question_mean - baseline_mean)
    deviation_std = abs(question_std - baseline_std)
    skewness_deviation = abs(question_skewness - baseline_skewness)
    kurtosis_deviation = abs(question_kurtosis - baseline_kurtosis)

    # Display metrics
    print(f"Baseline Mean: {baseline_mean:.2f}, Question Mean: {question_mean:.2f}")
    print(f"Baseline Std Dev: {baseline_std:.2f}, Question Std Dev: {question_std:.2f}")
    print(f"Skewness Deviation: {skewness_deviation:.2f}")
    print(f"Kurtosis Deviation: {kurtosis_deviation:.2f}")

    # Decision logic: tighter thresholds for improved accuracy
    if (deviation_mean > 1.5 or
        deviation_std > 1.5 or
        skewness_deviation > 0.4 or
        kurtosis_deviation > 0.8):
        print("Potential lie detected: Significant deviation from baseline.")
        return True
    else:
        print("No significant deviation detected: Response appears truthful.")
        return False

try:
    # Connect to Arduino
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    print("Connected to Arduino on port:", ARDUINO_PORT)

    # Collect baseline data
    baseline_data = collect_baseline(ser, BASELINE_DURATION)
    print("Baseline data analysis complete.")

    while True:
        prompt = input("Do you want to ask a question? (yes/no): ").strip().lower()
        if prompt == 'yes':
            print("Prepare to ask the question. Collecting data...")
            question_data = collect_data(ser, QUESTION_DURATION)
            analyze_response(question_data, baseline_data)
        elif prompt == 'no':
            print("Exiting program.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

except serial.SerialException as e:
    print("Failed to connect to Arduino:", e)
finally:
    if ser.is_open:
        ser.close()
        print("Serial connection closed.")
