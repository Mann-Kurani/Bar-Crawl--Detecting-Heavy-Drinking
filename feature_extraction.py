### Feature Extraction (BK7610)
import math
import numpy as np
from scipy.signal import butter, lfilter
# Constants
def acceleration(BK7610):
    Gravity = 9.8  # Gravitational acceleration in m/s²

    x_readings = BK7610.x  
    y_readings = BK7610.y  
    z_readings = BK7610.z  
    la=[]

    # Loop through the accelerometer readings
    for i in range(len(x_readings)):
        x = x_readings[i]
        y = y_readings[i]
        z = z_readings[i]

        # Calculate pitch and roll angles from accelerometer data
        pitch = math.atan2(-x, math.sqrt(y**2 + z**2))
        roll = math.atan2(y, z)

        # Calculate linear acceleration by removing gravity
        linear_x = x #- Gravity * math.sin(pitch)
        linear_y = y #- Gravity * math.sin(roll)
        linear_z = z #- Gravity * math.cos(pitch) * math.cos(roll)

        linear_acceleration = math.sqrt(linear_x**2 + linear_y**2 + (linear_z)**2)
        la.append(linear_acceleration)

    BK7610['acc'] = la

# High Pass Filter to remove gravity from sensor data


# Sample accelerometer data (replace with your own data)
def high_pass(BK7610):
    
    accel_data = np.array([BK7610['acc']])

    # Define the filter parameters
    cutoff_freq = 0.05  # Adjust as needed, this is the cutoff frequency in Hz
    filter_order = 4   # Adjust as needed, higher order provides steeper roll-off

    # Design the high-pass Butterworth filter
    def high_pass_filter(data, cutoff_freq, filter_order, sampling_rate):
        nyquist_freq = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(filter_order, normal_cutoff, btype='high', analog=False)
        filtered_data = lfilter(b, a, data)
        return filtered_data

    # Assuming you know the sampling rate of your accelerometer data
    sampling_rate = 40  # Replace with your actual sampling rate

    # Apply the high-pass filter to the accelerometer data
    filtered_accel_data = high_pass_filter(accel_data, cutoff_freq, filter_order, sampling_rate)
    BK7610['acc']=filtered_accel_data[0]
# high_pass(BK7610)
# Thresholding to detect steps
# import numpy as np

# Assuming you have preprocessed accelerometer data stored in the variable "BK7610.acc"
# Define your threshold value (adjust as needed)
threshold = 0.02  # Adjust based on your data and experimentation
# Loop through the filtered accelerometer data
def thresholding(BK7610):
    BK7610['steps'] = 0
    global peak_indices
    peak_indices = []
    for i in range(1, len(BK7610.acc)):
        # Calculate the change in acceleration from the previous data point
        acceleration_change = abs(BK7610.acc[i] - BK7610.acc[i - 1])
        
        # Check if the change exceeds the threshold
        if acceleration_change > threshold:
            BK7610.loc[i,'steps'] = 1
            peak_indices.append(i)

# Calculate step times
def step_time(BK7610):
    step_times = []
    for i in range(BK7610.shape[0]):
        if i in peak_indices:
            step_time = (BK7610.time[i] - BK7610.time[i - 1]).total_seconds()
            step_times.append(step_time)
        else:
            step_times.append(0)

    BK7610['step_time'] = step_times

