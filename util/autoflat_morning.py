import sys
import time
import datetime
import numpy as np
from astropy.io import fits

# PyQt5 and pyqtgraph for the user interface
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
import pyqtgraph as pg

# Import the ZWO camera wrapper from the provided source files
try:
    from zwo_cam_interface import zwoasi_wrapper
except ImportError:
    print("Error: The 'zwo_cam_interface.py' file was not found.")
    print("Please ensure it is in the same directory or in Python's path.")
    sys.exit(1)

# --- Configuration ---
CAM_NAME = "2600"
EXPOSURE_S = 0.1
GAIN = 100
INITIAL_BRIGHTNESS_THRESHOLD = 8000.0
FINAL_BRIGHTNESS_THRESHOLD = 29000.0

class AcquisitionWorker(QObject):
    """
    Worker class to handle all camera operations in a separate thread.
    This prevents the GUI from freezing during exposures.
    """
    # --- Signals to communicate with the GUI ---
    new_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    # Emits statistics: (latest_frame_mean, current_stack_mean, frames_collected)
    update_stats = pyqtSignal(float, float, int)
    finished = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    @pyqtSlot()
    def stop(self):
        """Signals the worker to stop its acquisition loop."""
        self._running = False
        self.update_status.emit("Stopping...")

    def run(self):
        """
        Main acquisition logic. This method is executed when the thread starts.
        """
        camera = None
        try:
            # --- Initialization ---
            self.update_status.emit(f"Initializing camera: {CAM_NAME}...")
            camera = zwoasi_wrapper(
                temp=-5, exp=EXPOSURE_S, gain=GAIN, crop=None,
                cam_name=CAM_NAME, live=True
            )
            camera.start()
            self.update_status.emit(f"Connected to {camera.name()}. Settings: Exp={EXPOSURE_S}s, Gain={GAIN}")

            # --- Phase 1: Wait for sufficient brightness ---
            self.update_status.emit(f"Phase 1: Waiting for image mean >= {INITIAL_BRIGHTNESS_THRESHOLD}...")
            
            while self._running:
                frame = camera.get_frame()
                if frame is not None:
                    mean_val = np.mean(frame)
                    self.new_frame.emit(frame)
                    self.update_stats.emit(mean_val, 0.0, 0) # No stack yet
                    if mean_val >= INITIAL_BRIGHTNESS_THRESHOLD:
                        self.update_status.emit(f"Phase 1 Complete. Mean was {mean_val:.2f}.")
                        break
                time.sleep(0.1)

            if not self._running:
                raise InterruptedError("Process stopped by user during Phase 1.")

            # --- Phase 2: Acquire and average bright frames ---
            self.update_status.emit(f"Phase 2: Acquiring frames until mean > {FINAL_BRIGHTNESS_THRESHOLD}...")
            bright_frames_count = 0
            sum_image = None
            
            while self._running:
                latest_frame = camera.get_frame()
                if latest_frame is not None:
                    # Initialize the running sum with the first frame
                    if sum_image is None:
                        sum_image = latest_frame.astype(np.float64)
                    else:
                        sum_image += latest_frame
                    
                    bright_frames_count += 1
                    latest_mean = np.mean(latest_frame)
                    stack_mean = np.mean(sum_image / bright_frames_count)

                    self.new_frame.emit(latest_frame)
                    self.update_stats.emit(latest_mean, stack_mean, bright_frames_count)

                    if latest_mean > FINAL_BRIGHTNESS_THRESHOLD:
                        self.update_status.emit("Phase 2 Complete. Final brightness reached.")
                        break
                time.sleep(0.1)
            
            if not self._running:
                 raise InterruptedError("Process stopped by user during Phase 2.")

            # --- Phase 3: Calculate average and save file ---
            if bright_frames_count == 0:
                self.finished.emit("Warning: No bright frames were collected. Cannot save.")
                return

            self.update_status.emit(f"Phase 3: Averaging {bright_frames_count} frames...")
            mean_image = sum_image / bright_frames_count

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mean_flat_{timestamp}.fits"

            hdu = fits.PrimaryHDU(mean_image.astype(np.uint16))
            hdu.header['EXPOSURE'] = (EXPOSURE_S, 'Exposure time in seconds')
            hdu.header['GAIN'] = (GAIN, 'Camera gain')
            hdu.header['NFRAMES'] = (bright_frames_count, 'Number of frames averaged')
            
            hdu.writeto(filename, overwrite=True)
            self.finished.emit(f"Success! Saved averaged flat to '{filename}'")

        except InterruptedError as e:
            self.finished.emit(str(e))
        except Exception as e:
            self.finished.emit(f"An error occurred: {e}")
        finally:
            if camera:
                camera.close()

class StatsWindow(QMainWindow):
    """
    The main GUI window for displaying statistics and the live image.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Flat Field Acquisition")
        self.setGeometry(100, 100, 1000, 800)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Statistics Section ---
        stats_group = QWidget()
        stats_layout = QGridLayout(stats_group)
        main_layout.addWidget(stats_group)

        # Create labels for statistics
        self.status_label = QLabel("Status: Initializing...")
        self.mean_label = QLabel("Latest Frame Mean: -")
        self.stack_mean_label = QLabel("Current Average Mean: -")
        self.frames_label = QLabel("Frames in Average: -")
        self.settings_label = QLabel(f"Settings: {EXPOSURE_S}s | Gain {GAIN}")

        # Style the labels
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #33A;")
        self.mean_label.setStyleSheet("font-size: 14px;")
        self.stack_mean_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.frames_label.setStyleSheet("font-size: 14px;")
        self.settings_label.setStyleSheet("font-size: 14px; color: #555;")

        # Add labels to the grid layout
        stats_layout.addWidget(self.status_label, 0, 0, 1, 2)
        stats_layout.addWidget(self.mean_label, 1, 0)
        stats_layout.addWidget(self.stack_mean_label, 1, 1)
        stats_layout.addWidget(self.frames_label, 2, 0)
        stats_layout.addWidget(self.settings_label, 2, 1)

        # --- Image View Section ---
        self.imageView = pg.ImageView()
        main_layout.addWidget(self.imageView)
        
        # --- Worker Thread Setup ---
        self.setup_worker_thread()
        
    def setup_worker_thread(self):
        self.thread = QThread()
        self.worker = AcquisitionWorker()
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.new_frame.connect(self.update_image_view)
        self.worker.update_status.connect(self.set_status)
        self.worker.update_stats.connect(self.update_stats_display)
        self.worker.finished.connect(self.on_finished)

        self.thread.start()

    # --- GUI Update Slots ---
    def set_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_stats_display(self, latest_mean, stack_mean, frame_count):
        self.mean_label.setText(f"Latest Frame Mean: {latest_mean:.2f}")
        if frame_count > 0:
            self.stack_mean_label.setText(f"Current Average Mean: {stack_mean:.2f}")
            self.frames_label.setText(f"Frames in Average: {frame_count}")
        else:
            self.stack_mean_label.setText("Current Average Mean: -")
            self.frames_label.setText("Frames in Average: -")

    def update_image_view(self, frame):
        self.imageView.setImage(np.rot90(frame), autoLevels=True)
        
    def on_finished(self, message):
        self.set_status(message)
        self.settings_label.setText("Process Finished. You can close this window.")

    def closeEvent(self, event):
        """Handle the window close event to ensure a clean shutdown."""
        print("Closing application...")
        if self.thread.isRunning():
            self.worker.stop()  # Signal the worker to stop
            self.thread.quit()  # Ask the event loop to quit
            # Wait for the thread to finish cleanly, with a timeout
            if not self.thread.wait(3000): # Wait up to 3 seconds
                 print("Warning: Worker thread did not terminate gracefully. Forcing exit.")
                 self.thread.terminate() # Force terminate if it's stuck
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatsWindow()
    window.show()
    sys.exit(app.exec_())