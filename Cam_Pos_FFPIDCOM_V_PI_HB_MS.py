##11 mm tube without video
#multi-steps curve
# coding=utf-8

import mvsdk
import numpy as np  
import cv2
import time
import sys
import ctypes
import gc  # Used to disable garbage collection
import os
import psutil  # Used to set process priority
from threading import Thread, Lock
from rpi_hardware_pwm import HardwarePWM
import matplotlib.pyplot as plt
import pandas as pd
import Adafruit_ADS1x15  # Import ADC module for voltage measurement
import atexit

# System parameter constants
SYSTEM_PARAMS = {
    'gravity': 9.8,                # Gravitational acceleration (m/s^2)
    'ball_diameter': 0.01,         # Ball diameter (m)
    'max_voltage': 24.0,           # Maximum supply voltage (V)
    'ball_mass': 4.068*0.001,      # Ball mass (kg)
    'added_mass': 6.06*0.001,       # mean added mass (kg)
    'damping': 0.694,              # Viscous damping coefficient (kg/s)
    'glycol_density': 1113.2,      # Glycol density (kg/m^3)
    'pwm_frequency': 18660,        # PWM frequency (Hz)
    'sample_rate': 100,            # Sampling rate (Hz)
    'sample_time': 0.01,           # Sampling period (s)
}

# Calculate glycol buoyancy
SYSTEM_PARAMS['glycol_buoyancy'] = (SYSTEM_PARAMS['glycol_density'] * 
                                   SYSTEM_PARAMS['gravity'] * 
                                   (4/3) * np.pi * (0.005)**3)  # Glycol buoyancy (N)

# High-precision timer class definition
class Timer:
    """High-precision timer class using Linux kernel clocks."""
    CLOCK_MONOTONIC = 1
    CLOCK_MONOTONIC_RAW = 4

    class Timespec(ctypes.Structure):
        _fields_ = [
            ('tv_sec', ctypes.c_long),
            ('tv_nsec', ctypes.c_long)
        ]

    def __init__(self):
        # Try different library names for better compatibility
        try:
            self.librt = ctypes.CDLL('librt.so.1', use_errno=True)
        except OSError:
            try:
                self.librt = ctypes.CDLL('librt.so', use_errno=True)
            except OSError:
                self.librt = ctypes.CDLL('libc.so.6', use_errno=True)
                print("Using libc.so.6 for timer functions")
        
        # Set function argument and return types
        self.librt.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(self.Timespec)]
        self.librt.clock_gettime.restype = ctypes.c_int
        
        # Initialize timespec structure
        self.ts = self.Timespec()
        self._start_time = self.get_raw_time()
        
    def get_raw_time(self):
        """Get raw time."""
        if self.librt.clock_gettime(self.CLOCK_MONOTONIC_RAW, ctypes.byref(self.ts)) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return self.ts.tv_sec + self.ts.tv_nsec * 1e-9
        
    def get_time(self):
        """Get elapsed time relative to start time (seconds)."""
        return self.get_raw_time() - self._start_time
        
    def busy_wait(self, target_time):
        """Busy-wait until the target time."""
        target_absolute = self._start_time + target_time
        while self.get_raw_time() < target_absolute:
            pass

# Global variables
global_ball_position = 0.0  # Current ball position in mm
position_lock = Lock()  # Lock to protect the ball position variable
processor = None  # Global processor object, initialized in main

# Variables for data logging
data_lock = Lock()
time_data = []  # Record time
setpoint_data = []  # Record setpoint
position_data = []  # Record actual position
error_data = []  # Record error
output_data = []  # Record control output
jitter_data = []  # Record timing jitter
voltage_data = []  # Record actual voltage
target_voltage_data = []  # Record target voltage
voltage_error_data = []  # Record voltage error
current_data = []  # Record actual current
feedforward_data = []  # Record feedforward voltage
pid_output_data = []  # Record PID output
lead_output_data = []  # Record phase-lead compensator output

# Post-processed data (computed after experiment)
z_position_data = []  # Position data in z-coordinate system
z_position_error = []  # Position error in z-coordinate system
z_setpoint_data = []  # Setpoint in z-coordinate system

# Global variables for control and state
data_already_saved = False
experiment_running = False  # Flag indicating whether the experiment has started

# PID + Phase-Lead Compensator controller
class PIDPhaseLeadController:
    def __init__(self, dt=0.01):
        """
        PID + Phase-Lead Compensator controller based on MATLAB code
        
        MATLAB parameters:
        Kc = 113*113/1000 = 12.769     % Controller gain [V/m]
        tau_i = 0.20                   % Integral time constant [s]
        tau_d = 0.0125                 % Derivative time constant [s]
        tau_f = 0.0021                 % Filter time constant [s]
        tau_1 = 0.0135                 % Phase-lead compensator parameter 1 [s]
        tau_2 = 0.01                   % Phase-lead compensator parameter 2 [s]
        """
        self.dt = dt
        
        # MATLAB controller parameters
        # self.Kc = 113 * 113 / 1000  # 12.769
        # self.tau_i = 0.20
        # self.tau_d = 0.0125
        # self.tau_f = 0.0021
        self.tau_1 = 0.066
        self.tau_2 = 0.01
        
        # PID parameters
        # self.kp = self.Kc
        # self.ki = self.Kc / self.tau_i
        # self.kd = self.Kc * self.tau_d
        # self.kp = 3.669
        # self.ki = 0.63845
        # self.kd = 0.02
        # self.kp = 2.036  #2.169
        # self.ki = 0.63845
        # self.kd = 0.000
        self.kp = 2.040  #2.169
        self.ki = 2.060
        self.kd = 0.010
        self.N = 20  # Filter coefficient
        
        # Ka = 1 (m to mm conversion factor)
        self.Ka = 1.0
        
        # PID state variables
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        
        # Phase-Lead Compensator state variables (discretized)
        # Discretized using bilinear transform: (tau_1*s + 1)/(tau_2*s + 1)
        self._compute_lead_compensator_coefficients()
        self.lead_input_prev = 0.0  # u[k-1]
        self.lead_output_prev = 0.0  # y[k-1]
        
        print(f"PID+Phase-Lead Controller initialized:")
        print(f"  PID: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}, N={self.N:.1f}")
        print(f"  Phase-Lead: tau_1={self.tau_1}, tau_2={self.tau_2}")
        print(f"  Lead coefficients: b0={self.b0:.6f}, b1={self.b1:.6f}, a1={self.a1:.6f}")
    
    def _compute_lead_compensator_coefficients(self):
        """Compute discretized coefficients for the phase-lead compensator."""
        T = self.dt
        
        # Bilinear transform coefficients for G(s) = (tau_1*s + 1)/(tau_2*s + 1)
        # G(z) = (b0 + b1*z^-1) / (1 + a1*z^-1)
        
        denominator = 2*self.tau_2/T + 1
        
        self.b0 = (2*self.tau_1/T + 1) / denominator
        self.b1 = (1 - 2*self.tau_1/T) / denominator  
        self.a1 = (1 - 2*self.tau_2/T) / denominator
        
    def compute(self, error, current_time=None):
        """
        Compute controller output
        error: Position error (mm) - target position minus current position
        current_time: Current time (kept for compatibility, not used)
        Returns: (control voltage, PID output, phase-lead output)
        """
        # ===== PID part =====
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (discrete integration)
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative term (with filter)
        error_derivative = (error - self.previous_error) / self.dt
        
        # Discrete filter implementation: kd*N*s/(s + N)
        self.filtered_derivative = (self.N * self.dt * error_derivative + 
                                   self.filtered_derivative) / (1 + self.N * self.dt)
        
        d_term = self.kd * self.filtered_derivative
        
        # PID output
        pid_output = p_term + i_term + d_term
        # PID output saturation [0-24V]
        pid_output = np.clip(pid_output, 0, 24)

        # ===== Phase-lead compensator part =====
        # Implement discrete transfer function: G(z) = (b0 + b1*z^-1) / (1 + a1*z^-1)
        # y[k] = b0*u[k] + b1*u[k-1] - a1*y[k-1]
        
        lead_output = (self.b0 * pid_output + 
                      self.b1 * self.lead_input_prev - 
                      self.a1 * self.lead_output_prev)
        
        # Full controller output
        controller_output = self.Ka * lead_output
        
        # Limit output voltage range
        controller_output = np.clip(controller_output, 0, SYSTEM_PARAMS['max_voltage'])
        
        # Update state variables
        self.previous_error = error
        self.lead_input_prev = pid_output
        self.lead_output_prev = lead_output
        
        return controller_output, pid_output, lead_output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.lead_input_prev = 0.0
        self.lead_output_prev = 0.0

# Voltage PID controller
class VoltagePIDController:
    def __init__(self, kp=0.69, ki=60.0, kd=0.0, dt=0.01, max_output=24.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_output = max_output
        self.previous_error = 0
        self.integral = 0
        
    def compute(self, error):
        # Compute integral term
        self.integral += error * self.dt
        
        # Integral saturation to prevent windup
        self.integral = np.clip(self.integral, -self.max_output/self.ki, self.max_output/self.ki)
        
        # Compute derivative term
        derivative = (error - self.previous_error) / self.dt
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update previous error
        self.previous_error = error
        
        # Limit output range
        return np.clip(output, 0, self.max_output)
        
    def reset(self):
        """Reset PID controller state."""
        self.previous_error = 0
        self.integral = 0

# Trajectory calculation function (MULTI-STEPS)
def calculate_trajectory(time, total_ramp_time, initial_pos_mm, final_pos_mm):
    """Compute position, velocity, and acceleration for a multi-step trajectory."""
    # Multi-step parameters
    step_height = 0.9  # Height of each step (mm)
    num_steps = 8  # Number of steps
    first_step_duration = 1.0  # Duration of first step (0 mm) in seconds
    normal_step_duration = 14.0  # Duration of other steps in seconds
    transition_time = 8.0  # Transition time (seconds)
    
    # Baseline position (corresponding to 0 mm in MATLAB)
    base_position_mm = initial_pos_mm
    
    # Handle the first step (special case)
    if time <= first_step_duration:
        # First step: stay at initial position
        position_mm = base_position_mm
        velocity_mm_s = 0.0
        acceleration_mm_s2 = 0.0
        
    elif time <= first_step_duration + transition_time:
        # Transition from first step to second step
        transition_progress = (time - first_step_duration) / transition_time
        smooth_factor = (1 - np.cos(np.pi * transition_progress)) / 2
        
        # Position
        position_mm = base_position_mm - smooth_factor * step_height  # Negative sign because motion is downward
        
        # Velocity (derivative of position)
        velocity_mm_s = -step_height * np.pi * np.sin(np.pi * transition_progress) / (2 * transition_time)
        
        # Acceleration (derivative of velocity)
        acceleration_mm_s2 = -step_height * np.pi * np.pi * np.cos(np.pi * transition_progress) / (2 * transition_time * transition_time)
        
    else:
        # Handle subsequent steps (starting from the 2nd step)
        t_adjusted = time - first_step_duration - transition_time  # Adjust time reference
        step_index = int(t_adjusted / normal_step_duration) + 1  # +1 because the first step has passed
        
        # Clamp range
        if step_index >= num_steps:
            position_mm = base_position_mm - (num_steps - 1) * step_height
            velocity_mm_s = 0.0
            acceleration_mm_s2 = 0.0
        else:
            # Compute time within current step period
            time_in_step = t_adjusted % normal_step_duration
            stable_time = normal_step_duration - transition_time
            
            if time_in_step < stable_time:
                # Stable phase
                position_mm = base_position_mm - step_index * step_height
                velocity_mm_s = 0.0
                acceleration_mm_s2 = 0.0
            else:
                # Transition phase - use cosine for smooth transition
                transition_progress = (time_in_step - stable_time) / transition_time
                smooth_factor = (1 - np.cos(np.pi * transition_progress)) / 2
                
                current_level = step_index * step_height
                next_level = (step_index + 1) * step_height
                
                # Position
                position_mm = base_position_mm - (current_level + smooth_factor * step_height)
                
                # Velocity (derivative of position)
                velocity_mm_s = -step_height * np.pi * np.sin(np.pi * transition_progress) / (2 * transition_time)
                
                # Acceleration (derivative of velocity)
                acceleration_mm_s2 = -step_height * np.pi * np.pi * np.cos(np.pi * transition_progress) / (2 * transition_time * transition_time)
    
    # Ensure position stays within a reasonable range
    max_displacement = (num_steps - 1) * step_height
    position_mm = max(base_position_mm - max_displacement, min(position_mm, base_position_mm))
    
    # Convert to SI units
    position_m = position_mm / 1000.0
    velocity_m_s = velocity_mm_s / 1000.0
    acceleration_m_s2 = acceleration_mm_s2 / 1000.0
    
    return position_m, velocity_m_s, acceleration_m_s2

# Feedforward control function
def calculate_feedforward_voltage(position_m, velocity_m_s, acceleration_m_s2):
    """Compute required voltage from the electromagnetic force formula."""
    # System parameters
    g = SYSTEM_PARAMS['gravity']
    D_b = SYSTEM_PARAMS['ball_diameter']
    V_max = SYSTEM_PARAMS['max_voltage']
    m = SYSTEM_PARAMS['ball_mass']
    m_added = SYSTEM_PARAMS['added_mass']
    b = SYSTEM_PARAMS['damping']
    F_b_eg = SYSTEM_PARAMS['glycol_buoyancy']
    
    # Ensure position is greater than zero to avoid division by zero
    position_m = max(position_m, 0.001)
    
    # Compute friction force
    F_F = b * velocity_m_s
    
    # Compute required electromagnetic force using Newton's second law
    # mg - F_EM - F_B - F_F = mz̈
    # Therefore F_EM = mg - F_B - F_F - mz̈
    F_EM = m*g - F_b_eg - F_F - m_added*acceleration_m_s2
    
    # Ensure electromagnetic force is non-negative (only attraction is possible)
    F_EM = max(F_EM, 0)
    
    # Compute the term inside the formula
    term = (0.2095)**2 + 8.728 * F_EM * (position_m/D_b)**2.264 * (m*g)**(-1)
    
    # Prevent square root of a negative number
    term = max(term, 0)
        
    # Compute final voltage
    voltage = (np.sqrt(term) - 0.2095) * (V_max/4.364)
    
    # Clamp voltage to valid range
    voltage = np.clip(voltage, 0, V_max)
    
    return voltage

class FrameProcessor:
    def __init__(self):
        self.running = True
        self.frame_counter = 0  # Frame counter
        self.frame_counter_lock = Lock()

        # ROI parameters - obtained from the second code
        self.roi_params = {
            'roi_x': 156,           # ROI top-left X coordinate
            'roi_y': 19,            # ROI top-left Y coordinate
            'roi_width': 82,       # ROI width
            'roi_height': 132,      # ROI height
            'zero_pixel_y': 17,     # Pixel coordinate of zero point in field direction
            'scale_factor': 13.0/89  # Scale factor (mm/pixel)
        }

    # Reset frame counter
    def reset_counter(self):
        with self.frame_counter_lock:
            self.frame_counter = 0
            
    # Increment frame counter
    def increment_counter(self):
        with self.frame_counter_lock:
            self.frame_counter += 1
            return self.frame_counter
            
    # Get current frame count
    def get_counter(self):
        with self.frame_counter_lock:
            return self.frame_counter

def pixel_to_physical(pixel_y, zero_pixel_y, scale_factor):
    """
    Convert pixel coordinates to physical coordinates.
    pixel_y: Pixel coordinate
    zero_pixel_y: Pixel coordinate corresponding to zero point
    scale_factor: Scale factor (mm/pixel)
    """
    return (pixel_y - zero_pixel_y) * scale_factor + 0.24 #0.2 for calibaration

def print_progress_bar(current, total, elapsed_time, bar_length=50):
    """Print progress bar."""
    percent = float(current) * 100 / total
    arrow = '=' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r[{arrow}{spaces}] {percent:.1f}% - Time: {elapsed_time:.1f}s")
    sys.stdout.flush()

def capture_frames(hCamera, pFrameBuffer, processor):
    """Camera acquisition thread."""
    global global_ball_position
    
    # Get ROI parameters from processor
    roi_params = processor.roi_params
    
    while processor.running:
        try:
            # Perform software trigger to acquire one frame
            mvsdk.CameraClearBuffer(hCamera)  # Clear all cached frames in camera buffer
            mvsdk.CameraSoftTrigger(hCamera)  # Execute one software trigger
            
            # Capture one frame with a 200 ms timeout
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            
            # Vertical flip processing for monochrome camera
            #mvsdk.CameraFlipFrameBuffer(pRawData, FrameHead, 1)
            
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            
            # Convert image data to OpenCV format
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth)
            
            # Get ROI parameters
            roi_x = roi_params['roi_x']
            roi_y = roi_params['roi_y']
            roi_width = roi_params['roi_width']
            roi_height = roi_params['roi_height']
            zero_pixel_y = roi_params['zero_pixel_y']
            scale_factor = roi_params['scale_factor']
            
            try:
                # ROI crop
                roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
                
                # Inverse binary threshold
                _, binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
                
                # Connected component analysis
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
                
                # Initialize ball position
                physical_y = float('nan')
                
                # Process connected components
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= 500:
                        centroid_y = centroids[i][1]
                        pixel_y = centroid_y + roi_y
                        # Convert pixel coordinate to physical coordinate
                        physical_y = (pixel_y - zero_pixel_y) * scale_factor + 0.2  # 0.05 is the camera calibration offset
                        
                        # Update global ball position
                        with position_lock:
                            global_ball_position = physical_y
                            
                        break  # Only process the first connected component that meets the condition
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
            
            # Release image buffer
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
            
            # Control acquisition rate
            time.sleep(0.002)  # Limit maximum frame rate to 500 fps
            
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:  # Ignore timeout errors
                print(f"Camera error({e.error_code}): {e.message}")
            time.sleep(0.01)  # Short wait when an error occurs
        except Exception as e:
            print(f"Error in capture loop: {e}")

def set_realtime_priority():
    """Set real-time priority using pure Python."""
    try:
        # Set CPU affinity to core 3
        current_process = psutil.Process()
        current_process.cpu_affinity([3])
        print("Process pinned to CPU core 3")
        
        # Try to set real-time scheduling policy
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
            print("Process set to FIFO scheduling with priority 99")
        except Exception as e:
            print(f"Failed to set FIFO scheduling: {e}")
        
        # Try memory locking
        try:
            import resource
            if hasattr(resource, 'mlockall'):
                resource.mlockall(resource.MCL_CURRENT | resource.MCL_FUTURE)
                print("Memory locked to prevent swapping")
            else:
                print("Memory locking not supported on this platform")
        except Exception as e:
            print("Memory locking not supported on this platform")
            
    except Exception as e:
        print(f"Failed to set real-time priority: {e}")

def run_control_experiment():
    """Main function for running the control experiment."""
    global time_data, setpoint_data, position_data, error_data, output_data, jitter_data, global_ball_position, experiment_running
    global voltage_data, target_voltage_data, voltage_error_data, current_data, feedforward_data, pid_output_data, lead_output_data
    global processor, position_controller, voltage_pid_controller
    
    # Mark experiment as started
    experiment_running = True
    
    # Clear data from previous experiment
    with data_lock:
        time_data.clear()
        setpoint_data.clear()
        position_data.clear()
        error_data.clear()
        output_data.clear()
        jitter_data.clear()
        voltage_data.clear()
        target_voltage_data.clear()
        voltage_error_data.clear()
        current_data.clear()
        feedforward_data.clear()
        pid_output_data.clear()
        lead_output_data.clear()
    
    # Reset frame counter
    processor.reset_counter()
    
    # Disable garbage collection
    gc.disable()
    
    # Set priority
    set_realtime_priority()
    
    # Create high-precision timer
    timer = Timer()
    
    # Control experiment parameters - adjustable as needed
    DT = 0.01  # Fixed time step, 10 ms
    experiment_duration = 115.0  # Experiment duration (s)
    ramp_time = experiment_duration # Ramp duration (s)
    
    # Additional variable definitions
    l = 14.94  # Total length in mm
    z = 4.5    # Steel ball initial coordinate in mm
    
    initial_position = l  # Initial position (mm)
    final_position = l - z  # Final position (mm)
    
    feed_forward_gain = 0.80  # Feedforward gain coefficient
    expected_points = int(experiment_duration / DT)  # Expected number of data points
    
    print(f"Control parameters: l = {l}mm, z = {z}mm, final_position = {final_position}mm")
    
    # Pre-allocate data arrays
    with data_lock:
        time_data = np.zeros(expected_points)
        setpoint_data = np.zeros(expected_points)
        position_data = np.zeros(expected_points)
        error_data = np.zeros(expected_points)
        output_data = np.zeros(expected_points)
        jitter_data = np.zeros(expected_points)
        voltage_data = np.zeros(expected_points)
        target_voltage_data = np.zeros(expected_points)
        voltage_error_data = np.zeros(expected_points)
        current_data = np.zeros(expected_points)
        feedforward_data = np.zeros(expected_points)
        pid_output_data = np.zeros(expected_points)
        lead_output_data = np.zeros(expected_points)
    
    # PWM settings
    pwm_channel = 0  # Use PWM channel 0
    pwm_freq = SYSTEM_PARAMS['pwm_frequency']  # PWM frequency in Hz
    pwm = HardwarePWM(pwm_channel=pwm_channel, hz=pwm_freq, chip=0)
    pwm.start(0)  # Start PWM with initial duty cycle of 0
    print(f"PWM started on channel {pwm_channel} with frequency {pwm_freq}Hz")
    
    # Create PID + phase-lead compensator controller
    position_controller = PIDPhaseLeadController(dt=DT)
    
    # Create voltage PID controller
    voltage_pid = VoltagePIDController(
        kp=0.69*0.9,
        ki=60*0.9,
        kd=0.0,
        dt=DT,
        max_output=SYSTEM_PARAMS['max_voltage']
    )
    voltage_pid_controller = voltage_pid  # Update global reference
    
    # Initialize ADC
    adc = Adafruit_ADS1x15.ADS1115()
    GAIN = 1  # +/-4.096V
    
    try:
        # Experiment start time
        print(f"Starting multi-steps experiment with duration {experiment_duration}s...")
        print(f"Multi-steps parameters: 6 steps, 0.9mm per step, first step 1s, other steps 14s, transition 8s")
        print("Progress:")
        
        start_time = timer.get_time()
        prev_elapsed_time = 0
        last_progress_print = 0
        
        # Control loop
        for i in range(expected_points):
            # Compute next sampling time point
            next_sample_time = start_time + (i * DT)
            
            # Wait with high precision until next sampling time
            timer.busy_wait(next_sample_time)
            
            # Get current time
            current_time = timer.get_time()
            elapsed_time = current_time - start_time
            
            # Update progress bar (every 0.1 s)
            if current_time - last_progress_print >= 0.1:
                print_progress_bar(i, expected_points, elapsed_time)
                last_progress_print = current_time
            
            # Compute timing jitter
            actual_dt = elapsed_time - prev_elapsed_time if i > 0 else DT
            jitter_ms = (actual_dt - DT) * 1000  # Convert to milliseconds
            prev_elapsed_time = elapsed_time
            
            # Get current ball position
            with position_lock:
                current_position = global_ball_position
                
            # Compute target trajectory in SI units
            target_position_m, target_velocity_m_s, target_acceleration_m_s2 = calculate_trajectory(
                elapsed_time, 
                ramp_time,
                initial_position, 
                final_position
            )
            
            # Convert target position to mm for PID control and data logging
            target_position_mm = target_position_m * 1000.0
            
            # Compute position error (target position - current position)
            position_error = target_position_mm - current_position
            
            # Compute feedforward voltage
            feedforward_voltage = calculate_feedforward_voltage(
                target_position_m,
                target_velocity_m_s,
                target_acceleration_m_s2
            )
            
            # Compute controller output (PID + Phase-Lead Compensator)
            # Note: error sign is reversed here (because control direction is opposite)
            pid_voltage, pid_output, lead_output = position_controller.compute(position_error * (-1), elapsed_time)
            
            # Compute reference voltage = feedforward + position controller output
            reference_voltage = feedforward_voltage * feed_forward_gain + pid_voltage
            reference_voltage = np.clip(reference_voltage, 0, SYSTEM_PARAMS['max_voltage'])
            
            try:
                # Read actual voltage and current
                # Read raw ADC value
                adc_value = adc.read_adc(0, gain=GAIN, data_rate=860)
                
                # Convert ADC value to actual voltage
                voltage_raw = adc_value * 4.096 / 32767.0  # GAIN = 1: ±4.096V
                
                # Current sensor conversion formula; this requires real sensor calibration
                current = (2.511 - voltage_raw) * 1.0159 / 50 / 0.1 - 0.0022 
                actual_voltage = (current * 60)  # Calculated based on actual circuit relationship
                actual_voltage = max(0, min(actual_voltage, SYSTEM_PARAMS['max_voltage']))
            except Exception as e:
                print(f"ADC reading error: {e}")
                actual_voltage = 0.0
                current = 0.0
            
            # Compute voltage error and apply voltage PID control
            voltage_error = reference_voltage - actual_voltage
            final_output = voltage_pid.compute(voltage_error)
            
            # Convert voltage to PWM duty cycle
            duty_cycle = (final_output / SYSTEM_PARAMS['max_voltage']) * 100
            
            # Set PWM duty cycle
            pwm.change_duty_cycle(duty_cycle)
            
            # Log data
            time_data[i] = elapsed_time
            setpoint_data[i] = target_position_mm
            position_data[i] = current_position
            error_data[i] = position_error
            output_data[i] = final_output
            jitter_data[i] = jitter_ms
            
            # Log voltage control data
            voltage_data[i] = actual_voltage
            target_voltage_data[i] = reference_voltage
            voltage_error_data[i] = voltage_error
            current_data[i] = current
            
            # Log feedforward and controller component data
            feedforward_data[i] = feedforward_voltage * feed_forward_gain
            pid_output_data[i] = pid_output
            lead_output_data[i] = lead_output
            
            # Increment frame count
            processor.increment_counter()
        
        # Ensure final progress bar shows 100%
        print_progress_bar(expected_points, expected_points, elapsed_time)
        print("\nExperiment completed!")
            
        # Show frame statistics at experiment end
        frame_count = processor.get_counter()
        capture_rate = (frame_count / expected_points) * 100
        print(f"\nFrame Statistics:")
        print(f"- Expected trigger points: {expected_points}")
        print(f"- Actual frames captured: {frame_count}")
        print(f"- Capture rate: {capture_rate:.2f}%")
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted!")
    finally:
        # Stop PWM
        if 'pwm' in locals():
            pwm.stop()
            print("PWM stopped")
        
        # Re-enable garbage collection
        gc.enable()
        
        # Save experiment data
        save_data_and_plot(l, z)
        
        # Exit program after completion
        sys.exit(0)

    # Data visualization class
class DataVisualizer:
    def __init__(self, time_data, z_position_data, z_setpoint_data, z_position_error, 
                 voltage_data, target_voltage_data, voltage_error_data, control_output_data,
                 feedforward_data=None, pid_output_data=None, lead_output_data=None):
        """Initialize data visualizer."""
        self.time_data = time_data
        self.z_position_data = z_position_data
        self.z_setpoint_data = z_setpoint_data
        self.z_position_error = z_position_error
        self.voltage_data = voltage_data
        self.target_voltage_data = target_voltage_data
        self.voltage_error_data = voltage_error_data
        self.control_output_data = control_output_data
        self.feedforward_data = feedforward_data
        self.pid_output_data = pid_output_data
        self.lead_output_data = lead_output_data
        
        # Create figure
        self.setup_plot()
        
    def setup_plot(self):
        """Set up plots with five subplots sharing the time axis."""
        # Create figure window with shared x-axis
        self.fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
        
        # Assign subplots
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5 = axes
        
        # Z-coordinate position plot - actual and target positions together
        self.ax1.plot(self.time_data, self.z_position_data, 'b-', label='Actual Z Position')
        self.ax1.plot(self.time_data, self.z_setpoint_data, 'r--', label='Target Z Position')
        self.ax1.grid(True)
        self.ax1.set_ylabel('Z Position(mm)')
        self.ax1.set_title('Z-Coordinate Position Tracking')
        self.ax1.legend()
        
        # Z-coordinate displacement error plot
        self.ax2.plot(self.time_data, self.z_position_error, 'r-', label='Z Position Error')
        self.ax2.grid(True)
        self.ax2.set_ylabel('Z Error(mm)')
        self.ax2.set_title('Z-Coordinate Position Error')
        self.ax2.legend()
        
        # Voltage control plot - target and actual voltage
        self.ax3.plot(self.time_data, self.target_voltage_data, 'g--', label='Target Voltage')
        self.ax3.plot(self.time_data, self.voltage_data, 'b-', label='Actual Voltage')
        self.ax3.grid(True)
        self.ax3.set_ylabel('Voltage(V)')
        self.ax3.set_title('Voltage Control')
        self.ax3.legend()
        
        # Voltage error plot
        self.ax4.plot(self.time_data, self.voltage_error_data, 'g-', label='Voltage Error')
        self.ax4.grid(True)
        self.ax4.set_ylabel('Error(V)')
        self.ax4.set_title('Voltage Error (Target - Actual)')
        self.ax4.legend()
        
        # Controller output component plot
        if self.feedforward_data is not None:
            self.ax5.plot(self.time_data, self.feedforward_data, 'k--', label='Feedforward')
        if self.pid_output_data is not None and self.lead_output_data is not None:
            self.ax5.plot(self.time_data, self.pid_output_data, 'c-', label='PID Output')
            self.ax5.plot(self.time_data, self.lead_output_data, 'm-', label='Lead Compensator Output')
        self.ax5.grid(True)
        self.ax5.set_ylabel('Voltage(V)')
        self.ax5.set_xlabel('Time(s)')
        self.ax5.set_title('Controller Output Components')
        self.ax5.legend()
        
        # Adjust layout and increase spacing between subplots
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.3)
    
    def show(self):
        """Display plots."""
        plt.show()

def save_data_and_plot(l, z):
    """Save recorded data to Excel and generate plots."""
    global time_data, setpoint_data, position_data, error_data, output_data, jitter_data, data_already_saved, experiment_running
    global voltage_data, target_voltage_data, voltage_error_data, current_data, feedforward_data, pid_output_data, lead_output_data
    global z_position_data, z_position_error, z_setpoint_data
    global processor, position_controller, voltage_pid_controller
    
    # Do not save again if data is already saved or experiment has not started
    if data_already_saved or not experiment_running:
        print("No data to save or data already saved")
        return
    
    # Specify save path
    save_path = "./data/position_control"
    
    try:
        # Ensure directory exists
        import os
        os.makedirs(save_path, exist_ok=True)
        
        with data_lock:
            if len(time_data) == 0 or isinstance(time_data, list) and not time_data:
                print("No data to save")
                return
                
            # Trim data to valid length (if arrays were pre-allocated)
            if isinstance(time_data, np.ndarray) and np.all(time_data[-1] == 0):
                # Find last non-zero time index
                valid_indices = np.where(time_data > 0)[0]
                if len(valid_indices) > 0:
                    last_valid_idx = valid_indices[-1] + 1
                    time_data = time_data[:last_valid_idx]
                    setpoint_data = setpoint_data[:last_valid_idx]
                    position_data = position_data[:last_valid_idx]
                    error_data = error_data[:last_valid_idx]
                    output_data = output_data[:last_valid_idx]
                    jitter_data = jitter_data[:last_valid_idx]
                    
                    # Trim voltage control data
                    voltage_data = voltage_data[:last_valid_idx]
                    target_voltage_data = target_voltage_data[:last_valid_idx]
                    voltage_error_data = voltage_error_data[:last_valid_idx]
                    current_data = current_data[:last_valid_idx]
                    
                    # Trim feedforward and controller component data
                    feedforward_data = feedforward_data[:last_valid_idx]
                    pid_output_data = pid_output_data[:last_valid_idx]
                    lead_output_data = lead_output_data[:last_valid_idx]
            
            # Post-processing: compute z-coordinate system data
            print("\nPerforming post-processing for z-coordinate system...")
            z_position_data = l - position_data  # Position in z-coordinate system
            z_setpoint_data = l - setpoint_data  # Setpoint in z-coordinate system
            z_position_error = z_setpoint_data - z_position_data  # Position error in z-coordinate system
            
            # Compute and display statistics
            max_jitter = np.max(np.abs(jitter_data))
            avg_jitter = np.mean(np.abs(jitter_data))
            std_jitter = np.std(jitter_data)
            print(f"Timing statistics:")
            print(f"- Maximum jitter: {max_jitter:.3f} ms")
            print(f"- Average jitter: {avg_jitter:.3f} ms")
            print(f"- Jitter std dev: {std_jitter:.3f} ms")
            
            # Position control performance in original coordinates
            mean_pos_error = np.mean(np.abs(error_data))
            max_pos_error = np.max(np.abs(error_data))
            std_pos_error = np.std(error_data)
            print(f"Position control performance (original coordinates):")
            print(f"- Mean absolute error: {mean_pos_error:.3f} mm")
            print(f"- Maximum absolute error: {max_pos_error:.3f} mm")
            print(f"- Error standard deviation: {std_pos_error:.3f} mm")
            
            # Position control performance in z-coordinates
            mean_z_pos_error = np.mean(np.abs(z_position_error))
            max_z_pos_error = np.max(np.abs(z_position_error))
            std_z_pos_error = np.std(z_position_error)
            print(f"Position control performance (z-coordinates):")
            print(f"- Mean absolute error: {mean_z_pos_error:.3f} mm")
            print(f"- Maximum absolute error: {max_z_pos_error:.3f} mm")
            print(f"- Error standard deviation: {std_z_pos_error:.3f} mm")
            
            # Voltage control performance
            mean_volt_error = np.mean(np.abs(voltage_error_data))
            max_volt_error = np.max(np.abs(voltage_error_data))
            std_volt_error = np.std(voltage_error_data)
            print(f"Voltage control performance:")
            print(f"- Mean absolute error: {mean_volt_error:.3f} V")
            print(f"- Maximum absolute error: {max_volt_error:.3f} V")
            print(f"- Error standard deviation: {std_volt_error:.3f} V")
                
            # Get frame count
            frame_count = processor.get_counter()
            expected_points = len(time_data)
            capture_rate = (frame_count / expected_points) * 100 if expected_points > 0 else 0
            print(f"Frame capture statistics:")
            print(f"- Expected trigger points: {expected_points}")
            print(f"- Actual frames captured: {frame_count}")
            print(f"- Capture rate: {capture_rate:.2f}%")
                
            # Save to Excel file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            excel_filename = os.path.join(save_path, f"position_control_analysis_{timestamp}.xlsx")
            
            # Create DataFrame
            df = pd.DataFrame({
                'Time(s)': time_data,
                'Position(mm)': position_data,
                'TargetPosition(mm)': setpoint_data,
                'PositionError(mm)': error_data,
                'Z_Position(mm)': z_position_data,
                'Z_TargetPosition(mm)': z_setpoint_data,
                'Z_PositionError(mm)': z_position_error,
                'ActualVoltage(V)': voltage_data,
                'TargetVoltage(V)': target_voltage_data,
                'VoltageError(V)': voltage_error_data,
                'ControlOutput(V)': output_data,
                'Current(A)': current_data,
                'Jitter(ms)': jitter_data,
                'FeedforwardVoltage(V)': feedforward_data,
                'PIDOutput(V)': pid_output_data,
                'LeadCompensatorOutput(V)': lead_output_data
            })
            
            # Add experiment parameter information
            experiment_params = pd.DataFrame({
                'Parameter': ['Sampling Rate', 'PWM Frequency', 
                            'Total Length (l)', 'Ball Start Position (z)', 
                            'Position-Controller-Kp', 'Position-Controller-Ki', 'Position-Controller-Kd', 'Position-Controller-N',
                            'Lead-tau_1', 'Lead-tau_2',
                            'Voltage-PID-Kp', 'Voltage-PID-Ki', 'Voltage-PID-Kd'],
                'Value': [1/np.mean(np.diff(time_data)), 
                        SYSTEM_PARAMS['pwm_frequency'], 
                        l, z,
                        position_controller.kp if position_controller else 0, 
                        position_controller.ki if position_controller else 0, 
                        position_controller.kd if position_controller else 0,
                        position_controller.N if position_controller else 0,
                        position_controller.tau_1 if position_controller else 0,
                        position_controller.tau_2 if position_controller else 0,
                        voltage_pid_controller.kp if voltage_pid_controller else 0,   
                        voltage_pid_controller.ki if voltage_pid_controller else 0,   
                        voltage_pid_controller.kd if voltage_pid_controller else 0],
                'Unit': ['Hz', 'Hz', 'mm', 'mm', 'V/mm', 'V/(mm*s)', 'V*s/mm', '-', 's', 's', '-', '-', '-']
            })
            
            # Add statistics data to Excel
            statistics_data = pd.DataFrame({
                'Metric': ['Maximum Jitter', 'Average Jitter', 'Jitter StdDev',
                        'Mean Position Error', 'Max Position Error', 'Position Error StdDev',
                        'Mean Z Position Error', 'Max Z Position Error', 'Z Position Error StdDev',
                        'Mean Voltage Error', 'Max Voltage Error', 'Voltage Error StdDev'],
                'Value': [max_jitter, avg_jitter, std_jitter,
                        mean_pos_error, max_pos_error, std_pos_error,
                        mean_z_pos_error, max_z_pos_error, std_z_pos_error,
                        mean_volt_error, max_volt_error, std_volt_error],
                'Unit': ['ms', 'ms', 'ms', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm', 'V', 'V', 'V']
            })
            
            # Add frame statistics to Excel
            frame_stats = pd.DataFrame({
                'Metric': ['Expected Trigger Points', 'Actual Frames Captured', 'Capture Rate (%)'],
                'Value': [expected_points, frame_count, capture_rate],
                'Unit': ['points', 'frames', '%']
            })
            
            # Use ExcelWriter to save multiple sheets into one Excel file
            with pd.ExcelWriter(excel_filename) as writer:
                experiment_params.to_excel(writer, sheet_name='Parameters', index=False)
                statistics_data.to_excel(writer, sheet_name='Statistics', index=False)
                frame_stats.to_excel(writer, sheet_name='Frame Statistics', index=False)
                df.to_excel(writer, sheet_name='Data', index=False)
            
            print(f"Data saved to: {excel_filename}")
            
            # Create data visualizer using z-coordinate data
            visualizer = DataVisualizer(
                time_data,
                z_position_data,
                z_setpoint_data,
                z_position_error,
                voltage_data,
                target_voltage_data,
                voltage_error_data,
                output_data,
                feedforward_data,
                pid_output_data,
                lead_output_data
            )
            
            # Display visualization plots
            print("Displaying data visualization...")
            visualizer.show()
            
            # Mark data as saved
            data_already_saved = True
            
    except Exception as e:
        print(f"Error saving data and plotting: {e}")
        import traceback
        traceback.print_exc()

def ensure_camera_closed():
    """Ensure camera device is closed when the program starts."""
    try:
        # Try enumerating all camera devices
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) == 0:
            print("No camera found!")
            return
            
        # Get information of the first camera
        DevInfo = DevList[0]
        
        # Check whether the camera is already open (via simple operation and error handling)
        try:
            # Try opening and closing immediately; success means it was not open before
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            if hCamera:
                print("Successfully initialized camera, closing...")
                mvsdk.CameraUnInit(hCamera)
        except mvsdk.CameraException as e:
            if e.error_code == -18:  # Device is already open
                print("Camera is already open, attempting to close it...")
                
                # A more direct camera close method is needed here, but SDK usually does not provide one
                # We can try closing processes that are using the camera via system command
                try:
                    if os.name == 'posix':  # Linux
                        # Find and terminate all processes that may occupy the camera
                        os.system("pkill -f python.*320_Junhua.py")
                        print("Terminated potential camera-using processes")
                        time.sleep(2)  # Give system time to release resources
                    else:
                        print("Camera is in use and automatic release is only supported on Linux")
                except Exception as e:
                    print(f"Failed to release camera: {e}")
    except Exception as e:
        print(f"Error checking camera state: {e}")

def main():
    global experiment_running, data_already_saved, processor
    
    # Ensure camera device is closed
    ensure_camera_closed()
    
    # Register exit callback to ensure data saving and plot generation
    atexit.register(lambda: save_data_and_plot(14.86, 4.5))
    
    # Enumerate cameras
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return
        
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]

    # Open camera
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return

    # Get camera capability description
    cap = mvsdk.CameraGetCapability(hCamera)

    # Set ISP output format to MONO8
    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    
    # Switch camera mode to software trigger mode
    mvsdk.CameraSetTriggerMode(hCamera, 1)  # Set to software trigger mode
    print("Camera set to software trigger mode")

    # Manual exposure settings
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 0.8 * 1000)  # Exposure time: 4 ms
    mvsdk.CameraSetAnalogGain(hCamera, 100)          # Analog gain
    mvsdk.CameraSetGamma(hCamera, 250)               # Gamma
    mvsdk.CameraSetContrast(hCamera, 200)            # Contrast

    # Set resolution
    resolution = mvsdk.CameraGetImageResolution(hCamera)
    resolution.iIndex = 1  # Use resolution corresponding to index 1
    mvsdk.CameraSetImageResolution(hCamera, resolution)

    # Set high frame-rate mode
    mvsdk.CameraSetFrameSpeed(hCamera, 2)           # 2: high-speed mode

    # Start SDK internal frame acquisition thread
    mvsdk.CameraPlay(hCamera)

    # Compute buffer size and allocate buffer
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    # Create frame processor
    processor = FrameProcessor()
    
    # Start camera acquisition thread
    capture_thread = Thread(target=capture_frames, args=(hCamera, pFrameBuffer, processor))
    capture_thread.daemon = True
    capture_thread.start()
    
    # Wait for user to start control experiment
    input("Press Enter to start control experiment...")
    
    # Start control experiment directly (no extra thread)
    # This way, the program exits automatically after experiment completion
    run_control_experiment()

    # Clean up resources (normally not reached because run_control_experiment calls sys.exit)
    processor.running = False
    if capture_thread.is_alive():
        capture_thread.join(timeout=1.0)
        
    # Clean up camera resources
    mvsdk.CameraStop(hCamera)
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    print("Camera resources released")
    
    # Try resetting PWM channel
    print("Resetting PWM hardware...")
    try:
        os.system("echo 0 > /sys/class/pwm/pwmchip0/unexport 2>/dev/null || true")
        time.sleep(0.5)
        os.system("echo 0 > /sys/class/pwm/pwmchip0/export 2>/dev/null || true")
        time.sleep(0.5)
        os.system("echo 0 > /sys/class/pwm/pwmchip0/unexport 2>/dev/null || true")
    except Exception as e:
        print(f"Hardware reset attempt failed: {e}")

if __name__ == '__main__':
    main()