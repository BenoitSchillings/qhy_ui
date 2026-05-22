"""
Autoguider using the layered Kalman + CUSUM controller from
https://github.com/BenoitSchillings/autoguider-kalman-cusum

Mount corrections are applied as TheSkyX pulse jogs via skyx.sky6RASCOMTele.
Camera is a ZWO via zwo_cam_interface.zwoasi_wrapper.

Star tracking uses normalized cross-correlation (cv2.matchTemplate) of a small
reference patch grabbed at star-selection time. This is more robust against
brighter interlopers, satellite trails, drift outside a brightest-pixel search
box, and the autoguide.py "click on a faint star but the global max wins" bug.

Pipeline per frame:
  template-match (NCC) -> compute_centroid_improved sub-pixel refine
    -> sanity checks (NCC score, jump magnitude, predicted position)
    -> pixel error -> two AxisControllers -> desired pixel displacement
    -> M_inv (from calibration) -> (W,N) pulse seconds
    -> mount.jog(-pulse_w, pulse_n)
"""

import sys
import time
import numpy as np
from collections import deque

import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGridLayout, QDoubleSpinBox, QGroupBox
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QPointF, QTimer
import pyqtgraph as pg

try:
    from zwo_cam_interface import zwoasi_wrapper
    from skyx import sky6RASCOMTele
    from util import HighValueFinder, compute_centroid_improved
except ImportError as e:
    print(f"Error: A required module is missing: {e}")
    sys.exit(1)


# --- Defaults ---
CAM_NAME_HINT = "220"
EXPOSURE_S = 0.5
GAIN = 30
CALIBRATION_PULSE_S = 1.5
CALIBRATION_SETTLE_TIME_S = 2.0
FRAME_POLL_MS = 100
MOUNT_LATENCY_S = 0.5

# --- Tracking robustness ---
TEMPLATE_HALF = 8           # template patch is (2*half) x (2*half)
TEMPLATE_NCC_MIN = 0.4      # below this NCC score, treat as lost lock
MAX_JUMP_PX = 20.0          # in GUIDING, drop frame if centroid jumps more than this
MAX_LOST_GIVEUP = 30        # frames; after this many lost frames, stop guiding
SEARCH_HALF_BASE = 32       # half-width of template search window (pixels)


# --- Kalman + CUSUM controller (adapted from autoguider-kalman-cusum) ---
class AxisController:
    def __init__(
        self,
        sigma_init=0.5,
        q_drift=1e-4,
        kalman_gain=0.7,
        min_move_sigma=1.0,
        cusum_k_sigma=0.5,
        cusum_h_sigma=4.0,
        cusum_gain=0.9,
        sigma_window=50,
        sigma_floor=0.05,
        mount_latency_s=0.3,
        initial_drift_rate_sigma=1.0,
    ):
        self.q_drift = q_drift
        self.kalman_gain = kalman_gain
        self.min_move_sigma = min_move_sigma
        self.cusum_k_sigma = cusum_k_sigma
        self.cusum_h_sigma = cusum_h_sigma
        self.cusum_gain = cusum_gain
        self.sigma_floor = sigma_floor
        self.mount_latency_s = max(0.0, float(mount_latency_s))

        self.x = np.zeros(2)
        self.P = np.diag([sigma_init ** 2, initial_drift_rate_sigma ** 2])
        self.H = np.array([1.0, 0.0])
        self.R = sigma_init ** 2

        self.S_pos = 0.0
        self.S_neg = 0.0
        self.n_pos = 0
        self.n_neg = 0

        self.innov = deque(maxlen=sigma_window)
        self.sigma = sigma_init

        self.pending = []
        self.last_source = "none"

    def update(self, error, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q_drift * np.array(
            [[dt ** 3 / 3.0, dt ** 2 / 2.0],
             [dt ** 2 / 2.0, dt]]
        )

        for item in self.pending:
            item[0] -= dt
        self.pending = [it for it in self.pending if it[0] > 0]
        in_flight = sum(c for _, c in self.pending)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        expected = self.H @ self.x + in_flight
        y = error - expected
        S = self.H @ self.P @ self.H + self.R
        K = self.P @ self.H / S

        prev_drift = self.x[1]

        self.x = self.x + K * y
        self.P = self.P - np.outer(K, self.H) @ self.P

        self.innov.append(y)
        if len(self.innov) >= 10:
            arr = np.asarray(self.innov)
            mad = np.median(np.abs(arr - np.median(arr)))
            self.sigma = max(1.4826 * mad, self.sigma_floor)
            self.R = self.sigma ** 2

        k = self.cusum_k_sigma * self.sigma
        h = self.cusum_h_sigma * self.sigma
        self.S_pos = max(0.0, self.S_pos + y - k)
        self.S_neg = min(0.0, self.S_neg + y + k)
        self.n_pos = self.n_pos + 1 if self.S_pos > 0 else 0
        self.n_neg = self.n_neg + 1 if self.S_neg < 0 else 0

        correction = 0.0
        self.last_source = "none"

        if self.S_pos > h and self.n_pos > 0:
            self.x[0] = error - in_flight
            self.x[1] = prev_drift
            correction = self.cusum_gain * self.x[0]
            self.last_source = "cusum"
            self._reset_cusum()
        elif self.S_neg < -h and self.n_neg > 0:
            self.x[0] = error - in_flight
            self.x[1] = prev_drift
            correction = self.cusum_gain * self.x[0]
            self.last_source = "cusum"
            self._reset_cusum()
        elif abs(self.x[0]) > self.min_move_sigma * self.sigma:
            correction = self.kalman_gain * self.x[0]
            self.last_source = "kalman"

        self.x[0] -= correction
        if correction != 0.0 and self.mount_latency_s > 0.0:
            self.pending.append([self.mount_latency_s, correction])
        return correction

    def predicted_position(self, dt):
        """Predicted error position one frame ahead, accounting for in-flight pulses."""
        in_flight = sum(c for t, c in self.pending if t > dt)
        return self.x[0] + self.x[1] * dt + in_flight

    def _reset_cusum(self):
        self.S_pos = self.S_neg = 0.0
        self.n_pos = self.n_neg = 0

    def reset(self):
        self.x[:] = 0.0
        self._reset_cusum()
        self.innov.clear()
        self.pending.clear()


class GuideController:
    def __init__(self, **kw):
        self.x_axis = AxisController(**kw)
        self.y_axis = AxisController(**kw)

    def update(self, err_x, err_y, dt):
        return self.x_axis.update(err_x, dt), self.y_axis.update(err_y, dt)

    def predicted_position(self, dt):
        return self.x_axis.predicted_position(dt), self.y_axis.predicted_position(dt)

    def reset(self):
        self.x_axis.reset()
        self.y_axis.reset()


# --- Non-blocking mount jog ---
class MountJogWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, mount, ra_pulse, dec_pulse):
        super().__init__()
        self.mount = mount
        self.ra_pulse = ra_pulse
        self.dec_pulse = dec_pulse

    @pyqtSlot()
    def run(self):
        try:
            self.mount.jog(self.ra_pulse, self.dec_pulse)
        except Exception as e:
            print(f"[JOG-ERROR] Mount jog failed: {e}")
        finally:
            self.finished.emit()


# --- Main guiding worker ---
class GuidingWorker(QObject):
    new_frame = pyqtSignal(object)
    update_zoom = pyqtSignal(object)
    update_status = pyqtSignal(str, str)
    update_graph = pyqtSignal(float, float, float, str)
    calibration_finished = pyqtSignal(bool, str)
    finished = pyqtSignal()

    def __init__(self, params):
        super().__init__()
        self._state = "IDLE"
        self.camera = None
        self.mount = None
        self.params = params
        # Used ONLY for the initial calibration-frame brightest-pixel pick.
        self.finder = HighValueFinder(search_box_size=64, blur_size=5)
        # Template + tracking state for both calibration (after the first frame)
        # and guiding.
        self.template = None
        self.template_half = params.get("template_half", TEMPLATE_HALF)
        self.template_ncc_min = params.get("template_ncc_min", TEMPLATE_NCC_MIN)
        self.max_jump_px = params.get("max_jump_px", MAX_JUMP_PX)
        self.pending_template_grab = False
        self.template_grab_from_brightest = False
        self.last_centroid = None
        self.lost_frames = 0
        self.search_half = SEARCH_HALF_BASE
        self.guide_star_pos = None
        self.cal_data = {}
        self.t0 = time.time()
        self.last_frame_time = None
        self.timer = QTimer(self)
        self.jog_thread = None
        self.jog_worker = None
        self.gc = None

    @pyqtSlot()
    def stop(self):
        self.timer.stop()
        if self.jog_thread and self.jog_thread.isRunning():
            self.jog_thread.quit()
        if self.camera:
            self.camera.close()
        if self.mount:
            try:
                self.mount.Disconnect()
            except Exception:
                pass
        self.finished.emit()

    def _send_jog_command(self, ra_pulse, dec_pulse):
        self.jog_thread = QThread()
        self.jog_worker = MountJogWorker(self.mount, ra_pulse, dec_pulse)
        self.jog_worker.moveToThread(self.jog_thread)
        self.jog_worker.finished.connect(self.jog_thread.quit)
        self.jog_worker.finished.connect(self.jog_worker.deleteLater)
        self.jog_thread.finished.connect(self.jog_thread.deleteLater)
        self.jog_thread.started.connect(self.jog_worker.run)
        self.jog_thread.start()
        self.cal_data["last_cal_move_time"] = time.time()

    def _connect_hardware(self):
        try:
            self.update_status.emit("Connecting to camera...", "blue")
            self.camera = zwoasi_wrapper(
                temp=0,
                exp=self.params["exposure_s"],
                gain=self.params["gain"],
                crop=1.0,
                cam_name=self.params["cam_name"],
                live=True,
            )
            self.camera.start()
            self.update_status.emit(f"Camera '{self.camera.name()}' connected.", "green")
        except Exception as e:
            self.update_status.emit(f"Camera Error: {e}", "red")
            return False
        try:
            self.update_status.emit("Connecting to mount (TheSkyX)...", "blue")
            self.mount = sky6RASCOMTele()
            self.mount.Connect()
            self.update_status.emit("Mount connected.", "green")
        except Exception as e:
            self.update_status.emit(f"Mount Error: {e}", "red")
            return False
        return True

    # --- Template-based star tracking ---
    def _grab_template(self, frame, cx, cy):
        half = self.template_half
        h, w = frame.shape[:2]
        cy_i, cx_i = int(round(cy)), int(round(cx))
        if cy_i - half < 0 or cx_i - half < 0 or cy_i + half >= h or cx_i + half >= w:
            return None
        patch = frame[cy_i - half:cy_i + half, cx_i - half:cx_i + half]
        return patch.astype(np.float32).copy()

    def _template_match(self, frame, px, py, search_half):
        """NCC match. Returns ((peak_x, peak_y), score) or (None, score)."""
        if self.template is None:
            return None, 0.0
        th, tw = self.template.shape
        th_half, tw_half = th // 2, tw // 2
        h, w = frame.shape[:2]
        y0 = max(0, int(py) - search_half - th_half)
        y1 = min(h, int(py) + search_half + th_half)
        x0 = max(0, int(px) - search_half - tw_half)
        x1 = min(w, int(px) + search_half + tw_half)
        window = frame[y0:y1, x0:x1].astype(np.float32)
        if window.shape[0] < th or window.shape[1] < tw:
            return None, 0.0
        result = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        peak_x = x0 + max_loc[0] + tw_half
        peak_y = y0 + max_loc[1] + th_half
        return (float(peak_x), float(peak_y)), float(score)

    def _predicted_centroid(self, dt):
        """Where do we expect the star to be on this frame?"""
        if self.last_centroid is None:
            return self.guide_star_pos
        if self.gc is None or self._state != "GUIDING":
            return self.last_centroid
        # Kalman + drift extrapolation, in pixel-error coordinates.
        dx_pred, dy_pred = self.gc.predicted_position(dt)
        return (self.guide_star_pos[0] + dx_pred,
                self.guide_star_pos[1] + dy_pred)

    @pyqtSlot(QPointF)
    def set_guide_star(self, pos):
        # Mitigation #1: seed tracking from the user's click rather than
        # falling back to the brightest pixel in the field.
        self.guide_star_pos = (pos.x(), pos.y())
        self.template = None
        self.pending_template_grab = True
        self.template_grab_from_brightest = False
        self.last_centroid = (pos.x(), pos.y())
        self.lost_frames = 0
        self.gc = GuideController(
            sigma_init=self.params["sigma_init"],
            kalman_gain=self.params["kalman_gain"],
            min_move_sigma=self.params["min_move_sigma"],
            cusum_h_sigma=self.params["cusum_h_sigma"],
            cusum_gain=self.params["cusum_gain"],
            mount_latency_s=self.params["mount_latency_s"],
        )
        self.last_frame_time = None
        self._state = "GUIDING"
        self.update_status.emit(
            f"Guide star at ({pos.x():.1f}, {pos.y():.1f}). Capturing template...",
            "blue",
        )

    @pyqtSlot()
    def start_calibration(self):
        if self._state == "CALIBRATING":
            return
        self._state = "CALIBRATING"
        self.cal_data = {"step": "START", "is_settled": True}
        # On the next frame we'll grab the brightest pixel and snapshot a
        # template from it; subsequent calibration steps track via NCC.
        self.template = None
        self.pending_template_grab = True
        self.template_grab_from_brightest = True
        self.last_centroid = None
        self.lost_frames = 0
        self.update_status.emit("Calibrating: acquiring baseline...", "orange")

    def _execute_cal_step(self, x, y):
        step = self.cal_data.get("step")
        if step == "START":
            self.cal_data["base_pos"] = (x, y)
            self.cal_data["step"] = "MOVE_WEST"
            self._send_jog_command(-CALIBRATION_PULSE_S, 0)
            self.cal_data["is_settled"] = False
        elif step == "MOVE_WEST":
            self.cal_data["west_pos"] = (x, y)
            self.cal_data["step"] = "MOVE_EAST"
            self._send_jog_command(CALIBRATION_PULSE_S, 0)
            self.cal_data["is_settled"] = False
        elif step == "MOVE_EAST":
            self.cal_data["step"] = "MOVE_NORTH"
            self._send_jog_command(0, CALIBRATION_PULSE_S)
            self.cal_data["is_settled"] = False
        elif step == "MOVE_NORTH":
            self.cal_data["north_pos"] = (x, y)
            self.cal_data["step"] = "MOVE_SOUTH"
            self._send_jog_command(0, -CALIBRATION_PULSE_S)
            self.cal_data["is_settled"] = False
        elif step == "MOVE_SOUTH":
            try:
                bx, by = self.cal_data["base_pos"]
                wx, wy = self.cal_data["west_pos"]
                nx, ny = self.cal_data["north_pos"]
                vec_w = ((wx - bx) / CALIBRATION_PULSE_S, (wy - by) / CALIBRATION_PULSE_S)
                vec_n = ((nx - bx) / CALIBRATION_PULSE_S, (ny - by) / CALIBRATION_PULSE_S)
                M = np.array([[vec_w[0], vec_n[0]], [vec_w[1], vec_n[1]]])
                if abs(np.linalg.det(M)) < 0.1:
                    raise np.linalg.LinAlgError("Calibration move too small or degenerate.")
                self.cal_data["matrix_inv"] = np.linalg.inv(M)
                self.cal_data["matrix"] = M
                self.calibration_finished.emit(True, "Calibration successful.")
                self.update_status.emit(
                    f"Calibration complete. |vec_w|={np.linalg.norm(vec_w):.2f}px/s, "
                    f"|vec_n|={np.linalg.norm(vec_n):.2f}px/s.",
                    "green",
                )
            except np.linalg.LinAlgError as e:
                self.calibration_finished.emit(False, f"Calibration Failed: {e}")
            self._state = "IDLE"
            self.template = None  # discard cal template; guide star may differ

    def _guide(self, x, y, dt):
        if "matrix_inv" not in self.cal_data or self.gc is None:
            self._state = "IDLE"
            return
        err_x = x - self.guide_star_pos[0]
        err_y = y - self.guide_star_pos[1]

        corr_x, corr_y = self.gc.update(err_x, err_y, dt)

        src_x = self.gc.x_axis.last_source
        src_y = self.gc.y_axis.last_source
        if src_x == "cusum" or src_y == "cusum":
            tag = "cusum"
        elif src_x == "kalman" or src_y == "kalman":
            tag = "kalman"
        else:
            tag = "none"

        self.update_graph.emit(err_x, err_y, time.time() - self.t0, tag)

        if corr_x != 0.0 or corr_y != 0.0:
            pulse_w, pulse_n = self.cal_data["matrix_inv"] @ np.array([-corr_x, -corr_y])
            self._send_jog_command(-pulse_w, pulse_n)

        sig_x = self.gc.x_axis.sigma
        sig_y = self.gc.y_axis.sigma
        self.update_status.emit(
            f"Guiding | err=({err_x:+.2f},{err_y:+.2f})px  "
            f"sigma=({sig_x:.2f},{sig_y:.2f})  src={tag}",
            "blue",
        )

    def _find_centroid(self, frame, dt_hint):
        """Returns (cx, cy, info_str) or (None, None, reason) when lock is lost.

        Mitigations layered here:
          #2: search around predicted (Kalman/drift-extrapolated) position
          #3: NCC template match instead of brightest-pixel
          #4: template is frozen at grab time, not adaptively updated
          #5: NCC score and jump magnitude sanity checks
        """
        # First-time grab: either from the user click or, for calibration,
        # from the brightest pixel in the field.
        if self.pending_template_grab:
            if self.template_grab_from_brightest:
                try:
                    max_x, max_y, _ = self.finder.find_high_value_element(frame)
                    cx_init, cy_init, _ = compute_centroid_improved(frame, max_x, max_y)
                except (ValueError, IndexError):
                    return None, None, "no_bright_star"
                seed_x, seed_y = cx_init, cy_init
            else:
                seed_x, seed_y = self.guide_star_pos
            tmpl = self._grab_template(frame, seed_x, seed_y)
            if tmpl is None:
                return None, None, "edge"
            self.template = tmpl
            self.pending_template_grab = False
            self.last_centroid = (seed_x, seed_y)
            return seed_x, seed_y, "grabbed"

        if self.template is None:
            return None, None, "no_template"

        predicted = self._predicted_centroid(dt_hint)

        # Widen the search window if we have been losing the star.
        sh = self.search_half * (1 + min(self.lost_frames, 5))
        sh = min(sh, max(frame.shape[:2]) // 2)

        peak, score = self._template_match(frame, predicted[0], predicted[1], sh)
        if peak is None or score < self.template_ncc_min:
            self.lost_frames += 1
            return None, None, f"ncc={score:.2f}"

        try:
            cx, cy, _ = compute_centroid_improved(frame, peak[0], peak[1])
        except (ValueError, IndexError, ZeroDivisionError):
            self.lost_frames += 1
            return None, None, "centroid_failed"
        if not (np.isfinite(cx) and np.isfinite(cy)):
            self.lost_frames += 1
            return None, None, "centroid_nan"

        # Continuity / jump check (skip during calibration — mount is moving)
        if self._state == "GUIDING" and self.last_centroid is not None:
            jump = float(np.hypot(cx - self.last_centroid[0], cy - self.last_centroid[1]))
            if jump > self.max_jump_px:
                self.lost_frames += 1
                return None, None, f"jump={jump:.1f}"

        self.last_centroid = (cx, cy)
        self.lost_frames = 0
        return cx, cy, f"ncc={score:.2f}"

    @pyqtSlot()
    def _process_frame(self):
        if self.camera is None:
            return
        frame = self.camera.get_frame()
        if frame is None:
            return
        self.new_frame.emit(frame)

        if self._state not in ("CALIBRATING", "GUIDING"):
            return

        # Wall-clock dt drives the Kalman predict step and the prediction window.
        now = time.time()
        if self.last_frame_time is None:
            dt_hint = self.params["exposure_s"]
        else:
            dt_hint = max(1e-3, now - self.last_frame_time)

        cx, cy, info = self._find_centroid(frame, dt_hint)

        if cx is None:
            if self.lost_frames > MAX_LOST_GIVEUP:
                self.update_status.emit(
                    f"Lost lock for {self.lost_frames} frames ({info}). Stopping.",
                    "red",
                )
                self._state = "IDLE"
            else:
                self.update_status.emit(
                    f"Lock lost ({info}); widening search (lost={self.lost_frames})",
                    "orange",
                )
            return

        # Show zoomed star postage stamp.
        edge = 32
        h, w = frame.shape[:2]
        if 0 < int(cy - edge) and 0 < int(cx - edge) and int(cy + edge) < h and int(cx + edge) < w:
            self.update_zoom.emit(
                frame[int(cy - edge):int(cy + edge), int(cx - edge):int(cx + edge)]
            )

        if self._state == "CALIBRATING":
            if self.cal_data.get("is_settled", False):
                self._execute_cal_step(cx, cy)
            else:
                settle_end = self.cal_data.get("last_cal_move_time", 0) + \
                    CALIBRATION_PULSE_S + CALIBRATION_SETTLE_TIME_S
                if time.time() >= settle_end:
                    self.camera.ClearBuffers()
                    self.cal_data["is_settled"] = True
                else:
                    self.update_status.emit("Calibrating: mount moving, settling...", "orange")
        elif self._state == "GUIDING":
            if info == "grabbed":
                self.last_frame_time = now
                return  # don't issue a correction on the template-grab frame
            if self.last_frame_time is None:
                self.last_frame_time = now
                return
            dt = max(1e-3, now - self.last_frame_time)
            if dt > 10.0:
                self.last_frame_time = now
                return
            self._guide(cx, cy, dt)

        self.last_frame_time = now

    @pyqtSlot()
    def run(self):
        if self._connect_hardware():
            self.timer.timeout.connect(self._process_frame)
            self.timer.start(FRAME_POLL_MS)
        else:
            self.stop()


# --- Main window ---
class GuiderWindow(QMainWindow):
    select_star_signal = pyqtSignal(QPointF)

    def __init__(self, params):
        super().__init__()
        self.setWindowTitle("Auto-Guider (Kalman + CUSUM, NCC tracking)")
        self.setGeometry(50, 50, 1300, 900)
        self.params = params

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QVBoxLayout()
        self.imv = pg.ImageView()
        self.imv.getImageItem().mouseClickEvent = self.image_clicked
        left_panel.addWidget(self.imv)

        self.zoom_view = pg.ImageView()
        self.zoom_view.ui.histogram.hide()
        self.zoom_view.ui.roiBtn.hide()
        self.zoom_view.ui.menuBtn.hide()
        self.zoom_view.setMinimumHeight(256)
        self.zoom_view.setMaximumHeight(256)
        left_panel.addWidget(self.zoom_view)

        right_panel = QVBoxLayout()

        controls = QGroupBox("Control")
        controls_layout = QGridLayout(controls)
        self.calibrate_button = QPushButton("Calibrate")
        self.guide_button = QPushButton("Click image to guide")
        self.guide_button.setEnabled(False)
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label, 0, 0, 1, 2)
        controls_layout.addWidget(self.calibrate_button, 1, 0)
        controls_layout.addWidget(self.guide_button, 1, 1)
        right_panel.addWidget(controls)

        tuning = QGroupBox("Controller (re-applied on next star click)")
        tlay = QGridLayout(tuning)
        self.kalman_gain_box = self._make_spin(0.0, 1.0, 0.05, params["kalman_gain"])
        self.min_move_box = self._make_spin(0.0, 5.0, 0.1, params["min_move_sigma"])
        self.cusum_h_box = self._make_spin(1.0, 10.0, 0.5, params["cusum_h_sigma"])
        self.cusum_gain_box = self._make_spin(0.0, 1.0, 0.05, params["cusum_gain"])
        self.latency_box = self._make_spin(0.0, 5.0, 0.1, params["mount_latency_s"])
        self.ncc_min_box = self._make_spin(0.0, 1.0, 0.05, params["template_ncc_min"])
        self.jump_box = self._make_spin(1.0, 200.0, 1.0, params["max_jump_px"])
        for r, (label, box) in enumerate([
            ("Kalman gain", self.kalman_gain_box),
            ("Min-move sigma", self.min_move_box),
            ("CUSUM h (sigma)", self.cusum_h_box),
            ("CUSUM gain", self.cusum_gain_box),
            ("Mount latency (s)", self.latency_box),
            ("Min NCC score", self.ncc_min_box),
            ("Max jump (px)", self.jump_box),
        ]):
            tlay.addWidget(QLabel(label), r, 0)
            tlay.addWidget(box, r, 1)
        right_panel.addWidget(tuning)

        self.graphWidget = pg.PlotWidget(title="Guiding error (pixels)")
        self.graphWidget.addLegend()
        self.graphWidget.setLabel("left", "Error (px)")
        self.graphWidget.setLabel("bottom", "Time (s)")
        self.graph_data = {
            "t": deque(maxlen=400),
            "dx": deque(maxlen=400),
            "dy": deque(maxlen=400),
        }
        self.plot_dx = self.graphWidget.plot(pen="r", name="dx")
        self.plot_dy = self.graphWidget.plot(pen="b", name="dy")
        self.cusum_marker = self.graphWidget.plot(
            pen=None, symbol="x", symbolBrush="m", symbolSize=10, name="CUSUM"
        )
        self.cusum_points = {"t": deque(maxlen=100), "v": deque(maxlen=100)}
        right_panel.addWidget(self.graphWidget)

        main_layout.addLayout(left_panel, 7)
        main_layout.addLayout(right_panel, 3)

        self.setup_worker_thread()

    def _make_spin(self, lo, hi, step, val):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(3)
        s.setValue(val)
        return s

    def _collect_params(self):
        self.params["kalman_gain"] = self.kalman_gain_box.value()
        self.params["min_move_sigma"] = self.min_move_box.value()
        self.params["cusum_h_sigma"] = self.cusum_h_box.value()
        self.params["cusum_gain"] = self.cusum_gain_box.value()
        self.params["mount_latency_s"] = self.latency_box.value()
        self.params["template_ncc_min"] = self.ncc_min_box.value()
        self.params["max_jump_px"] = self.jump_box.value()

    def setup_worker_thread(self):
        self.thread = QThread()
        self.worker = GuidingWorker(self.params)
        self.worker.moveToThread(self.thread)

        self.calibrate_button.clicked.connect(self.worker.start_calibration)
        self.select_star_signal.connect(self.worker.set_guide_star)

        self.worker.new_frame.connect(self.update_frame)
        self.worker.update_zoom.connect(self.update_zoom_view)
        self.worker.update_status.connect(self.set_status)
        self.worker.update_graph.connect(self.update_graph_data)
        self.worker.calibration_finished.connect(self.on_calibration_finished)
        self.worker.finished.connect(self.thread.quit)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def image_clicked(self, event):
        if self.guide_button.isEnabled():
            self._collect_params()
            self.worker.params = self.params
            self.worker.template_ncc_min = self.params["template_ncc_min"]
            self.worker.max_jump_px = self.params["max_jump_px"]
            pos = self.imv.getImageItem().mapFromScene(event.pos())
            self.select_star_signal.emit(pos)
            self.guide_button.setText("Guiding...")
            self.guide_button.setEnabled(False)
            self.calibrate_button.setEnabled(False)

    @pyqtSlot(object)
    def update_frame(self, frame):
        self.imv.setImage(frame.T, autoRange=False, autoLevels=False, autoHistogramRange=False)

    @pyqtSlot(object)
    def update_zoom_view(self, zoom):
        if zoom.size > 0:
            self.zoom_view.setImage(zoom.T, autoRange=True, autoLevels=True)

    @pyqtSlot(str, str)
    def set_status(self, text, color):
        self.status_label.setText(f"Status: {text}")
        self.status_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color};")

    @pyqtSlot(float, float, float, str)
    def update_graph_data(self, dx, dy, t, src):
        self.graph_data["t"].append(t)
        self.graph_data["dx"].append(dx)
        self.graph_data["dy"].append(dy)
        self.plot_dx.setData(list(self.graph_data["t"]), list(self.graph_data["dx"]))
        self.plot_dy.setData(list(self.graph_data["t"]), list(self.graph_data["dy"]))
        if src == "cusum":
            self.cusum_points["t"].append(t)
            self.cusum_points["v"].append(0.0)
            self.cusum_marker.setData(list(self.cusum_points["t"]), list(self.cusum_points["v"]))

    @pyqtSlot(bool, str)
    def on_calibration_finished(self, success, message):
        self.guide_button.setEnabled(success)
        self.calibrate_button.setEnabled(True)
        self.guide_button.setText("Click star to guide" if success else "Calibration failed")
        print(f"[UI] {message}")

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.worker.stop()
            if not self.thread.wait(5000):
                self.thread.terminate()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kalman+CUSUM autoguider (SkyX pulse jog)")
    parser.add_argument("--cam", default=CAM_NAME_HINT, help="camera name hint")
    parser.add_argument("--exp", type=float, default=EXPOSURE_S, help="guide exposure (s)")
    parser.add_argument("--gain", type=int, default=GAIN, help="guide camera gain")
    parser.add_argument("--sigma-init", type=float, default=0.5, help="initial sigma (px)")
    parser.add_argument("--kalman-gain", type=float, default=0.7)
    parser.add_argument("--min-move-sigma", type=float, default=1.0)
    parser.add_argument("--cusum-h-sigma", type=float, default=4.0)
    parser.add_argument("--cusum-gain", type=float, default=0.9)
    parser.add_argument("--mount-latency-s", type=float, default=MOUNT_LATENCY_S)
    parser.add_argument("--template-half", type=int, default=TEMPLATE_HALF,
                        help="template patch half-size (full = 2*half)")
    parser.add_argument("--template-ncc-min", type=float, default=TEMPLATE_NCC_MIN,
                        help="minimum NCC score before declaring lock lost")
    parser.add_argument("--max-jump-px", type=float, default=MAX_JUMP_PX,
                        help="max allowed centroid jump per frame during guiding")
    args = parser.parse_args()

    params = {
        "cam_name": args.cam,
        "exposure_s": args.exp,
        "gain": args.gain,
        "sigma_init": args.sigma_init,
        "kalman_gain": args.kalman_gain,
        "min_move_sigma": args.min_move_sigma,
        "cusum_h_sigma": args.cusum_h_sigma,
        "cusum_gain": args.cusum_gain,
        "mount_latency_s": args.mount_latency_s,
        "template_half": args.template_half,
        "template_ncc_min": args.template_ncc_min,
        "max_jump_px": args.max_jump_px,
    }

    app = QApplication(sys.argv)
    window = GuiderWindow(params)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
