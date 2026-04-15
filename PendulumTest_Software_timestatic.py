#without video with current

# ## without video 

import mvsdk
import numpy as np  
import cv2
import time
import sys
import ctypes
import gc
import os
import psutil
from collections import deque
from threading import Thread, Lock, Event
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, Cursor
from scipy import fft
import pandas as pd

# High-precision timer class definition
class Timer:
    """A precise timer class based on the Linux kernel clock."""
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
        
        # Initialize the timespec structure
        self.ts = self.Timespec()
        self._start_time = self.get_raw_time()
        
    def get_raw_time(self):
        """Get the raw timestamp."""
        if self.librt.clock_gettime(self.CLOCK_MONOTONIC_RAW, ctypes.byref(self.ts)) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return self.ts.tv_sec + self.ts.tv_nsec * 1e-9
        
    def get_time(self):
        """Get elapsed time in seconds relative to start."""
        return self.get_raw_time() - self._start_time
        
    def busy_wait(self, target_time):
        """Busy-wait until the target time is reached."""
        target_absolute = self._start_time + target_time
        while self.get_raw_time() < target_absolute:
            pass

# Visualization class definition
class DataVisualizer:
    def __init__(self, time_data, position_data):
        """Initialize the data visualizer."""
        self.time_data = time_data
        self.position_data = position_data
        
        # Compute frequency-domain data
        self.freq, self.amplitude = self.compute_fft()
        
        # Create figures
        self.setup_plot()
        
    def compute_fft(self):
        """Compute FFT to obtain frequency-domain data."""
        # Compute sampling frequency
        fs = 1/np.mean(np.diff(self.time_data))
        
        # Perform FFT on physical position data
        n = len(self.position_data)
        yf = fft.fft(self.position_data)
        freq = fft.fftfreq(n, 1/fs)
        
        # Keep only positive frequencies and avoid zero frequency
        positive_freq_mask = freq > 0  # Use strict > 0 to avoid issues on logarithmic scale
        freq = freq[positive_freq_mask]
        amplitude = 2.0/n * np.abs(yf[positive_freq_mask])
        
        return freq, amplitude
        
    def setup_plot(self):
        """Set up the plots."""
        # Create two figure windows
        self.fig1 = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig1.add_subplot(111)
        
        self.fig2 = plt.figure(figsize=(12, 6))
        self.ax2 = self.fig2.add_subplot(111)
        
        # Position subplot (time domain)
        self.position_line, = self.ax1.plot(self.time_data, self.position_data, 'b-', label='Position')
        self.ax1.grid(True)
        self.ax1.set_xlabel('Time(s)')
        self.ax1.set_ylabel('Position(mm)')
        self.ax1.set_title('Object Position (Time Domain)')
        self.ax1.legend()
        
        # Frequency-domain plot (using logarithmic scale)
        self.freq_line, = self.ax2.semilogx(self.freq, self.amplitude, 'g-', label='Frequency Spectrum')
        self.ax2.grid(True, which="both", ls="-", alpha=0.6)
        self.ax2.grid(True, which="minor", ls=":", alpha=0.3)
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Amplitude')
        
        # Set x-axis range to keep a reasonable log-scale span
        freq_min = max(0.1, min(self.freq))  # Avoid excessively small frequency values
        freq_max = min(max(self.freq), 1000)  # Upper bound can be adjusted as needed
        self.ax2.set_xlim(freq_min, freq_max)
        
        # Set major and minor ticks
        self.ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        self.ax2.set_title('Frequency Domain Analysis')
        self.ax2.legend()
        
        # Add cursors
        self.cursor1 = Cursor(self.ax1, useblit=True, color='gray', linewidth=1)
        self.cursor2 = Cursor(self.ax2, useblit=True, color='gray', linewidth=1)
        
        # Add reset-zoom button for each figure
        self.reset_button_ax1 = self.fig1.add_axes([0.85, 0.02, 0.1, 0.04])
        self.reset_button1 = Button(self.reset_button_ax1, 'Reset Zoom')
        self.reset_button1.on_clicked(self.reset_zoom)
        
        self.reset_button_ax2 = self.fig2.add_axes([0.85, 0.02, 0.1, 0.04])
        self.reset_button2 = Button(self.reset_button_ax2, 'Reset Zoom')
        self.reset_button2.on_clicked(self.reset_zoom)
        
        # Connect events
        self.fig1.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig1.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig2.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig2.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Adjust layout
        self.fig1.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        self.fig2.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        
    def reset_zoom(self, event):
        """Reset plot zoom."""
        if event.inaxes == self.ax1:
            self.ax1.relim()
            self.ax1.autoscale()
            self.fig1.canvas.draw()
        elif event.inaxes == self.ax2:
            self.ax2.relim()
            self.ax2.autoscale()
            # Keep a reasonable range for log scale when resetting
            freq_min = max(0.1, min(self.freq))
            freq_max = min(max(self.freq), 100)
            self.ax2.set_xlim(freq_min, freq_max)
            self.fig2.canvas.draw()
            
    def on_scroll(self, event):
        """Handle mouse wheel zoom events."""
        ax = event.inaxes
        if ax is None: return
        
        scale_factor = 0.9 if event.button == 'up' else 1.1
        
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        x_center = event.xdata
        y_center = event.ydata
        
        if ax == self.ax2:
            # Zoom under logarithmic scale
            x_min_new = x_center * (x_min/x_center)**scale_factor
            x_max_new = x_center * (x_max/x_center)**scale_factor
            x_min = max(0.1, x_min_new)  # Ensure it never zooms below 0.1
            x_max = x_max_new
        else:
            # Zoom under linear scale
            x_min = x_center - (x_center - x_min) * scale_factor
            x_max = x_center + (x_max - x_center) * scale_factor
            
        y_min = y_center - (y_center - y_min) * scale_factor
        y_max = y_center + (y_max - y_center) * scale_factor
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        if ax == self.ax2:
            self.fig2.canvas.draw()
        else:
            self.fig1.canvas.draw()
        
    def on_click(self, event):
        """Handle mouse click events."""
        if event.dblclick:
            self.reset_zoom(event)
    
    def show(self):
        """Display plots."""
        plt.show()
    
    def save_plots(self, save_dir, timestamp):
        """Save plots to files."""
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Save time-domain plot
        time_domain_file = os.path.join(save_dir, f"time_domain_{timestamp}.png")
        self.fig1.savefig(time_domain_file, dpi=300, bbox_inches='tight')
        
        # Save frequency-domain plot
        freq_domain_file = os.path.join(save_dir, f"freq_domain_{timestamp}.png")
        self.fig2.savefig(freq_domain_file, dpi=300, bbox_inches='tight')
        
        print(f"Plots saved to: {time_domain_file} and {freq_domain_file}")

# Global variables
global_ball_position = 0.0  # Current ball position, in mm
position_lock = Lock()  # Lock for protecting the ball position variable

# Variables for data logging
data_lock = Lock()
time_data = []  # Logged time
position_data = []  # Logged physical position
jitter_data = []  # Logged timing jitter
centroid_pixel_data = []  # Logged centroid x-coordinate in pixels
image_acquisition_time_data = []  # Logged image acquisition time
image_processing_time_data = []  # Logged image processing time

# Image acquisition statistics
frame_counter = 0  # Counter for successfully captured frames
frame_counter_lock = Lock()  # Lock for protecting the counter
trigger_event = Event()  # Used to notify new triggers

# Global state flags
data_already_saved = False
experiment_running = False
# Flag used to control progress print frequency
last_progress_print = 0

def pixel_to_physical(pixel_x, zero_pixel_x, scale_factor):
    """
    Convert pixel coordinates to physical coordinates.
    pixel_x: x-coordinate in pixels
    zero_pixel_x: pixel x-coordinate corresponding to physical zero
    scale_factor: scaling factor (mm/pixel)
    """
    return (pixel_x - zero_pixel_x) * scale_factor + 0.20

def process_image(hCamera, pFrameBuffer, timer):
    """
    Process one image frame and return ball position, centroid pixel, and timing stats.
    Returns: (physical position, centroid x pixel, image acquisition time (ms), image processing time (ms))
    """
    # ROI parameters
    ## 11-samll tube
    # roi_x = 156 # ROI starting x-coordinate
    # roi_y = 19   # ROI starting y-coordinate
    # roi_width = 82  # ROI width
    # roi_height = 132 # Corrected to 123, consistent with reference code

    # # Coordinate conversion parameters
    # zero_pixel_x = roi_x  # Set zero point to ROI starting x-coordinate
    # scale_factor = 13.0/89  # Scale factor (mm/pixel)
    
    #Pendulum ROI
    roi_x = 86      # ROI starting x-coordinate
    roi_y = 104       # ROI starting y-coordinate
    roi_width = 132  # ROI width
    roi_height = 82  # Corrected to 123, consistent with reference code

    # Coordinate conversion parameters
    zero_pixel_x = roi_x  # Set zero point to ROI starting x-coordinate
    scale_factor = 14.20/72  # Scale factor (mm/pixel)

    try:
        # Record image acquisition start time
        start_acquisition_time = timer.get_raw_time()
        
        # Try to get an image with a 200 ms timeout
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        
        # Record image acquisition completion time
        end_acquisition_time = timer.get_raw_time()
        image_acquisition_time = (end_acquisition_time - start_acquisition_time) * 1000  # Convert to milliseconds
        
        # Increment counter if image capture succeeds
        with frame_counter_lock:
            global frame_counter
            frame_counter += 1
        
        # Record image processing start time
        start_processing_time = timer.get_raw_time()
        
        # Apply vertical flip for monochrome camera
        mvsdk.CameraFlipFrameBuffer(pRawData, FrameHead, 1)
        
        # Read image data directly from pRawData, skipping CameraImageProcess
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pRawData)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3)
        
        # For single-channel images, remove the last dimension
        if len(frame.shape) > 2 and frame.shape[2] == 1:
            frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth)
        
        # Crop ROI
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # Inverse binary thresholding
        _, binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # Initialize return values
        physical_x = float('nan')
        centroid_pixel_x = float('nan')
        
        # Process connected components
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 500:
                centroid_x = centroids[i][0]  # Centroid x-coordinate within ROI
                pixel_x = centroid_x + roi_x  # Convert to full-image x-coordinate
                centroid_pixel_x = pixel_x  # Save centroid x-coordinate in pixels
                physical_x = pixel_to_physical(pixel_x, zero_pixel_x, scale_factor)  # Compute physical position
                break  # Only process the first component that meets the condition
        
        # Record image processing completion time
        end_processing_time = timer.get_raw_time()
        image_processing_time = (end_processing_time - start_processing_time) * 1000  # Convert to milliseconds
        
        # Finally release the image buffer
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        
        return physical_x, centroid_pixel_x, image_acquisition_time, image_processing_time
    
    except mvsdk.CameraException as e:
        if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:  # Ignore timeout errors
            print("Camera error({}): {}".format(e.error_code, e.message))
        return float('nan'), float('nan'), float('nan'), float('nan')
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return float('nan'), float('nan'), float('nan'), float('nan')

def set_realtime_priority():
    """Set real-time priority using pure Python."""
    try:
        # Set CPU affinity to core 3
        current_process = psutil.Process()
        current_process.cpu_affinity([3])
        print("Process pinned to CPU core 3")
        
        # Try setting real-time scheduling policy
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

def print_progress_bar(current, total, elapsed_time, bar_length=50):
    """Print a progress bar."""
    percent = float(current) * 100 / total
    arrow = '=' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r[{arrow}{spaces}] {percent:.1f}% - Time: {elapsed_time:.1f}s")
    sys.stdout.flush()

def run_camera_experiment(hCamera, pFrameBuffer):
    """Run the camera experiment."""
    global time_data, position_data, jitter_data, centroid_pixel_data, image_acquisition_time_data, image_processing_time_data
    global global_ball_position, experiment_running, last_progress_print, frame_counter, data_already_saved
    
    # Reset flags
    data_already_saved = False
    
    # Mark experiment as started
    experiment_running = True
    
    # Clear data from the previous experiment
    with data_lock:
        time_data = []
        position_data = []
        jitter_data = []
        centroid_pixel_data = []
        image_acquisition_time_data = []
        image_processing_time_data = []
    
    # Reset frame counter
    with frame_counter_lock:
        frame_counter = 0
    
    # Disable garbage collection
    gc.disable()
    
    # Set priority
    set_realtime_priority()
    
    # Create high-precision timer
    timer = Timer()
    
    # Experiment parameters
    DT = 0.01           # Fixed time step, 10 ms
    experiment_duration = 30.0  # Experiment duration (seconds)
    expected_points = int(experiment_duration / DT)  # Expected number of data points
    
    # Pre-allocate data arrays
    with data_lock:
        time_data = np.zeros(expected_points)
        position_data = np.zeros(expected_points)
        jitter_data = np.zeros(expected_points)
        centroid_pixel_data = np.zeros(expected_points)
        image_acquisition_time_data = np.zeros(expected_points)
        image_processing_time_data = np.zeros(expected_points)
    
    try:
        # Experiment start
        print(f"Starting camera experiment with duration {experiment_duration}s...")
        print("Progress:")
        
        # Initialize variables related to progress printing
        last_progress_print = timer.get_time()
        start_time = timer.get_time()
        prev_elapsed_time = 0
        
        # Sampling loop
        for i in range(expected_points):
            # Compute next sampling time point
            next_sample_time = start_time + (i * DT)
            
            # Perform software trigger as shown in official examples
            mvsdk.CameraClearBuffer(hCamera)  # Clear all frames currently buffered in camera
            mvsdk.CameraSoftTrigger(hCamera)  # Execute one software trigger
            
            # Process image and get position, centroid pixel, and timing stats
            position, centroid_pixel, img_acq_time, img_proc_time = process_image(hCamera, pFrameBuffer, timer)
            
            # Wait with high precision until next sampling time
            timer.busy_wait(next_sample_time)
            
            # Get current time
            current_time = timer.get_time()
            elapsed_time = current_time - start_time
            
            # Compute timing jitter
            actual_dt = elapsed_time - prev_elapsed_time if i > 0 else DT
            jitter_ms = (actual_dt - DT) * 1000  # Convert to milliseconds
            prev_elapsed_time = elapsed_time
            
            # Log data
            time_data[i] = elapsed_time
            position_data[i] = position if not np.isnan(position) else 0.0
            jitter_data[i] = jitter_ms
            centroid_pixel_data[i] = centroid_pixel if not np.isnan(centroid_pixel) else 0.0
            image_acquisition_time_data[i] = img_acq_time if not np.isnan(img_acq_time) else 0.0
            image_processing_time_data[i] = img_proc_time if not np.isnan(img_proc_time) else 0.0
            
            # Update progress bar every 0.1 seconds
            if current_time - last_progress_print >= 0.1:
                print_progress_bar(i, expected_points, elapsed_time)
                last_progress_print = current_time
            
        # Ensure final progress bar update shows 100%
        print_progress_bar(expected_points, expected_points, elapsed_time)
        print("\nExperiment completed!")
        
        # Show number of captured frames
        with frame_counter_lock:
            capture_rate = (frame_counter / expected_points) * 100
            print(f"\nTrigger statistics:")
            print(f"- Expected trigger points: {expected_points}")
            print(f"- Actual frames captured: {frame_counter}")
            print(f"- Capture rate: {capture_rate:.2f}%")
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted!")
    finally:
        # Mark experiment as ended
        experiment_running = False
        
        # Re-enable garbage collection
        gc.enable()
        
        # Save experiment data
        save_data()

def save_data():
    """Save recorded data to CSV and visualize it."""
    global time_data, position_data, jitter_data, centroid_pixel_data, image_acquisition_time_data, image_processing_time_data
    global data_already_saved, experiment_running, frame_counter
    
    # Do not save again if data has already been saved
    if data_already_saved:
        print("Data already saved")
        return
    
    # Check whether there is data to save
    with data_lock:
        if not isinstance(time_data, np.ndarray) or len(time_data) == 0:
            print("No data to save")
            return
    
    # Set output path
    save_path = "./camera_data"
    
    try:
        # Ensure directory exists
        os.makedirs(save_path, exist_ok=True)
        
        with data_lock:
            # Trim data to valid length (if arrays were pre-allocated)
            valid_data = True
            if isinstance(time_data, np.ndarray) and len(time_data) > 0:
                # Find the last non-zero time index
                valid_indices = np.where(time_data > 0)[0]
                if len(valid_indices) > 0:
                    last_valid_idx = valid_indices[-1] + 1
                    time_data = time_data[:last_valid_idx]
                    position_data = position_data[:last_valid_idx]
                    jitter_data = jitter_data[:last_valid_idx]
                    centroid_pixel_data = centroid_pixel_data[:last_valid_idx]
                    image_acquisition_time_data = image_acquisition_time_data[:last_valid_idx]
                    image_processing_time_data = image_processing_time_data[:last_valid_idx]
                else:
                    valid_data = False
            else:
                valid_data = False
            
            if not valid_data:
                print("No valid data to save")
                return
                
            # Create data array
            data = np.column_stack((
                time_data,
                position_data,
                centroid_pixel_data,
                jitter_data,
                image_acquisition_time_data,
                image_processing_time_data
            ))
            
            # Save to CSV file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"camera_{timestamp}.csv")
            
            # Add header and save
            header = "Time(s),Position(mm),Centroid_Pixel_X,Jitter(ms),Image_Acquisition_Time(ms),Image_Processing_Time(ms)"
            np.savetxt(filename, data, delimiter=',', header=header, comments='')
            
            print(f"Data saved to {filename}")
            
            # Compute and display statistics
            max_jitter = np.max(np.abs(jitter_data))
            avg_jitter = np.mean(np.abs(jitter_data))
            std_jitter = np.std(jitter_data)
            
            # Compute image acquisition/processing timing statistics
            avg_img_acq_time = np.mean(image_acquisition_time_data[image_acquisition_time_data > 0])
            max_img_acq_time = np.max(image_acquisition_time_data)
            std_img_acq_time = np.std(image_acquisition_time_data[image_acquisition_time_data > 0])
            
            avg_img_proc_time = np.mean(image_processing_time_data[image_processing_time_data > 0])
            max_img_proc_time = np.max(image_processing_time_data)
            std_img_proc_time = np.std(image_processing_time_data[image_processing_time_data > 0])
            
            # Compute statistics for physical position and centroid x-coordinate
            valid_position_data = position_data[position_data != 0]
            valid_centroid_data = centroid_pixel_data[centroid_pixel_data > 0]
            
            if len(valid_position_data) > 0:
                avg_position = np.mean(valid_position_data)
                std_position = np.std(valid_position_data)
                min_position = np.min(valid_position_data)
                max_position = np.max(valid_position_data)
            else:
                avg_position = std_position = min_position = max_position = 0
                
            if len(valid_centroid_data) > 0:
                avg_centroid_x = np.mean(valid_centroid_data)
                std_centroid_x = np.std(valid_centroid_data)
                min_centroid_x = np.min(valid_centroid_data)
                max_centroid_x = np.max(valid_centroid_data)
            else:
                avg_centroid_x = std_centroid_x = min_centroid_x = max_centroid_x = 0
            
            print(f"Timing statistics:")
            print(f"- Maximum jitter: {max_jitter:.3f} ms")
            print(f"- Average jitter: {avg_jitter:.3f} ms")
            print(f"- Jitter std dev: {std_jitter:.3f} ms")
            print(f"Image acquisition time:")
            print(f"- Average: {avg_img_acq_time:.3f} ms")
            print(f"- Maximum: {max_img_acq_time:.3f} ms")
            print(f"- Std dev: {std_img_acq_time:.3f} ms")
            print(f"Image processing time:")
            print(f"- Average: {avg_img_proc_time:.3f} ms")
            print(f"- Maximum: {max_img_proc_time:.3f} ms")
            print(f"- Std dev: {std_img_proc_time:.3f} ms")
            print(f"Physical position statistics:")
            print(f"- Average: {avg_position:.3f} mm")
            print(f"- Std dev: {std_position:.3f} mm")
            print(f"- Min: {min_position:.3f} mm")
            print(f"- Max: {max_position:.3f} mm")
            print(f"Centroid X position statistics:")
            print(f"- Average: {avg_centroid_x:.1f} pixels")
            print(f"- Std dev: {std_centroid_x:.1f} pixels")
            print(f"- Min: {min_centroid_x:.1f} pixels")
            print(f"- Max: {max_centroid_x:.1f} pixels")
            
            # Add trigger statistics to the report
            with frame_counter_lock:
                expected_points = len(time_data)
                capture_rate = (frame_counter / expected_points) * 100 if expected_points > 0 else 0
                print(f"Frame capture statistics:")
                print(f"- Expected trigger points: {expected_points}")
                print(f"- Actual frames captured: {frame_counter}")
                print(f"- Capture rate: {capture_rate:.2f}%")
            
            # Mark data as saved
            data_already_saved = True
            
            # Save additional Excel file containing experiment parameters and frequency-domain data
            excel_filename = os.path.join(save_path, f"camera_analysis_{timestamp}.xlsx")
            
            # Create data visualizer object
            visualizer = DataVisualizer(time_data, position_data)
            
            # Get frequency-domain data
            freq, amplitude = visualizer.freq, visualizer.amplitude
            
            # Create DataFrames
            df_time_domain = pd.DataFrame({
                'Time(s)': time_data,
                'Position(mm)': position_data,
                'Centroid_Pixel_X': centroid_pixel_data,
                'Jitter(ms)': jitter_data,
                'Image_Acquisition_Time(ms)': image_acquisition_time_data,
                'Image_Processing_Time(ms)': image_processing_time_data
            })
            
            df_freq_domain = pd.DataFrame({
                'Frequency(Hz)': freq,
                'Amplitude': amplitude
            })
            
            # Add experiment parameter information
            experiment_params = pd.DataFrame({
                'Parameter': ['Sampling Rate', 'Experiment Duration', 'Zero Pixel X', 'Scale Factor'],
                'Value': [1/np.mean(np.diff(time_data)), len(time_data) * np.mean(np.diff(time_data)), 156, 13.0/89],
                'Unit': ['Hz', 'seconds', 'pixels', 'mm/pixel']
            })
            
            # Add statistics information
            statistics_data = pd.DataFrame({
                'Metric': ['Maximum Jitter', 'Average Jitter', 'Jitter StdDev', 'Capture Rate',
                          'Avg Image Acquisition Time', 'Max Image Acquisition Time', 'StdDev Image Acquisition Time',
                          'Avg Image Processing Time', 'Max Image Processing Time', 'StdDev Image Processing Time',
                          'Avg Position', 'StdDev Position', 'Min Position', 'Max Position',
                          'Avg Centroid X', 'StdDev Centroid X', 'Min Centroid X', 'Max Centroid X'],
                'Value': [max_jitter, avg_jitter, std_jitter, capture_rate,
                         avg_img_acq_time, max_img_acq_time, std_img_acq_time,
                         avg_img_proc_time, max_img_proc_time, std_img_proc_time,
                         avg_position, std_position, min_position, max_position,
                         avg_centroid_x, std_centroid_x, min_centroid_x, max_centroid_x],
                'Unit': ['ms', 'ms', 'ms', '%', 'ms', 'ms', 'ms', 'ms', 'ms', 'ms', 
                        'mm', 'mm', 'mm', 'mm', 'pixels', 'pixels', 'pixels', 'pixels']
            })
            
            # Save multiple sheets into one Excel file using ExcelWriter
            with pd.ExcelWriter(excel_filename) as writer:
                experiment_params.to_excel(writer, sheet_name='Parameters', index=False)
                statistics_data.to_excel(writer, sheet_name='Statistics', index=False)
                df_time_domain.to_excel(writer, sheet_name='Time Domain', index=False)
                df_freq_domain.to_excel(writer, sheet_name='Frequency Domain', index=False)
            
            print(f"Analysis data saved to: {excel_filename}")
            
            # Save plots
            visualizer.save_plots(save_path, timestamp)
            
            # Display visualization plots
            print("Displaying data visualization...")
            visualizer.show()
            
    except Exception as e:
        print(f"Error saving data: {e}")
        import traceback
        traceback.print_exc()

def ensure_camera_closed():
    """Ensure camera device is closed when the program starts."""
    try:
        # Try to enumerate all camera devices
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) == 0:
            print("No camera found!")
            return
            
        # Get information for the first camera
        DevInfo = DevList[0]
        
        # Check whether the camera is already open (via simple operation and error capture)
        try:
            # Try opening and immediately closing; success means it was not previously open
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            if hCamera:
                print("Successfully initialized camera, closing...")
                mvsdk.CameraUnInit(hCamera)
        except mvsdk.CameraException as e:
            if e.error_code == -18:  # Device is already open
                print("Camera is already open, attempting to close it...")
                
                # Attempt to close processes occupying the camera via system command
                try:
                    if os.name == 'posix':  # Linux
                        os.system("pkill -f python.*320_Junhua.py")
                        print("Terminated potential camera-using processes")
                        time.sleep(2)  # Give the system time to release resources
                    else:
                        print("Camera is in use and automatic release is only supported on Linux")
                except Exception as e:
                    print(f"Failed to release camera: {e}")
    except Exception as e:
        print(f"Error checking camera state: {e}")

def main():
    global experiment_running, data_already_saved
    
    # Ensure camera device is closed
    ensure_camera_closed()
    
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
    
    # Switch camera mode to software trigger
    mvsdk.CameraSetTriggerMode(hCamera, 1)  # Set to software trigger mode
    print("Camera set to software trigger mode")

    # Manual exposure settings
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 0.8 * 1000)  # Exposure time 4 ms
    mvsdk.CameraSetAnalogGain(hCamera, 100)          # Analog gain
    mvsdk.CameraSetGamma(hCamera, 250)               # Gamma
    mvsdk.CameraSetContrast(hCamera, 200)            # Contrast
    
    # Set resolution
    resolution = mvsdk.CameraGetImageResolution(hCamera)
    resolution.iIndex = 1  # Use the resolution corresponding to index 1
    mvsdk.CameraSetImageResolution(hCamera, resolution)

    # Set high-frame-rate mode
    mvsdk.CameraSetFrameSpeed(hCamera, 2)           # 2: high-speed mode

    # Start the SDK internal image acquisition thread
    mvsdk.CameraPlay(hCamera)

    # Compute buffer size and allocate buffer
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    print("Preparing camera for capture...")
    time.sleep(2)  # Let the camera start and stabilize
    
    # Wait for user input to start experiment
    input("Press Enter to start camera experiment...")
    
    # Run camera experiment
    run_camera_experiment(hCamera, pFrameBuffer)
    
    # Clean up camera resources
    mvsdk.CameraStop(hCamera)
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    print("Camera resources released")

if __name__ == '__main__':
    main()