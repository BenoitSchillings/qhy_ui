
import sys
import math
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QSizePolicy
)
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen, QFont, QTransform
from PyQt5.QtCore import Qt, QPointF, pyqtSignal

# Attempt to import the skyx module, but don't fail if it's not there.
try:
    from skyx import sky6RASCOMTele, SkyxConnectionError
    SKY_X_AVAILABLE = True
except ImportError:
    SKY_X_AVAILABLE = False

class TelescopeController:
    """
    Handles communication with the telescope mount using vector-based motion.
    Provides a simulation mode if the mount is not connected.
    """
    def __init__(self):
        self.simulated = True
        self.telescope = None
        self.original_rate = [15.0, 0.0]  # Default sidereal rate

        if SKY_X_AVAILABLE:
            try:
                self.telescope = sky6RASCOMTele()
                print("TheSkyX module found. Attempting to connect...")
                self.simulated = False
                print("Telescope controller initialized in Real Mode.")
            except SkyxConnectionError as e:
                print(f"Could not connect to TheSkyX: {e}. Switching to Simulation Mode.")
                self.simulated = True
            except Exception as e:
                print(f"An unexpected error occurred with TheSkyX: {e}. Switching to Simulation Mode.")
                self.simulated = True
        else:
            print("TheSkyX module not found. Starting in Simulation Mode.")

    def store_current_rate(self):
        """Reads and stores the telescope's current tracking rate."""
        if self.simulated:
            print("SIM: Storing original rate (defaulting to sidereal).")
            self.original_rate = [15.0, 0.0]
        else:
            try:
                self.original_rate = self.telescope.get_rate()
                print(f"REAL: Stored original rate: {self.original_rate}")
            except Exception as e:
                print(f"Error getting rate: {e}. Defaulting to sidereal.")
                self.original_rate = [15.0, 0.0]

    def move(self, d_ra_offset, d_dec_offset):
        """
        Sets the telescope's slew rate by applying offsets to the original rate.
        """
        new_d_ra = self.original_rate[0] + d_ra_offset
        new_d_dec = self.original_rate[1] + d_dec_offset

        if self.simulated:
            print(f"SIM: Setting vector rate (d_ra={new_d_ra:.2f}, d_dec={new_d_dec:.2f})")
        else:
            try:
                self.telescope.rate(new_d_ra, new_d_dec)
            except Exception as e:
                print(f"Error setting vector rate: {e}")

    def stop(self):
        """Restores the original tracking rate."""
        if self.simulated:
            print(f"SIM: Restoring original rate (d_ra={self.original_rate[0]:.2f}, d_dec={self.original_rate[1]:.2f})")
        else:
            try:
                self.telescope.rate(self.original_rate[0], self.original_rate[1])
                print("REAL: Restored original rate.")
            except Exception as e:
                print(f"Error restoring rate: {e}")


class ArrowPadWidget(QWidget):
    """
    A joystick-style control pad with non-linear speed control.
    """
    start_move_requested = pyqtSignal()
    vector_move_requested = pyqtSignal(float, float)
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.angle = 0
        self.flip_h = 1
        self.flip_v = 1
        
        self.is_slewing = False
        self.mouse_pos = QPointF(0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = side / 2 - 5

        # Background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2C3E50"))
        painter.drawEllipse(center, radius, radius)

        # Speed indicator circles
        painter.setBrush(Qt.NoBrush)
        painter.setFont(QFont('monospace', 10))
        min_radius_ratio = 0.1
        max_radius_ratio = 0.9
        
        for i in range(1, 6):
            percentage = i / 5.0
            circle_radius = radius * (min_radius_ratio + (max_radius_ratio - min_radius_ratio) * percentage)
            painter.setPen(QPen(QColor("#FFFFFF"), 1, Qt.SolidLine))
            painter.drawEllipse(center, circle_radius, circle_radius)
            painter.setPen(QPen(QColor("#FFFFFF")))
            label_point = QPointF(center.x() + circle_radius + 5, center.y() + 4)
            painter.drawText(label_point, f"{int(percentage*100)}%")

        # Center dead zone
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#34495E"))
        painter.drawEllipse(center, radius * min_radius_ratio, radius * min_radius_ratio)

        # Draw indicator arrows (visual only)
        painter.save()
        painter.translate(center)
        painter.rotate(self.angle)
        painter.scale(self.flip_h, self.flip_v)
        painter.translate(-center)
        self._draw_reference_arrows(painter)
        painter.restore()
        
        # Draw joystick position indicator
        if self.is_slewing:
            painter.setPen(QPen(QColor("#3498DB"), 3))
            painter.setBrush(QColor("#5DADE2"))
            painter.drawLine(center, self.mouse_pos)
            painter.drawEllipse(self.mouse_pos, 10, 10)

    def _draw_reference_arrows(self, painter):
        side = min(self.width(), self.height())
        center_x, center_y = self.width() / 2, self.height() / 2
        
        arrow_base_width = side * 0.10
        arrow_head_width = side * 0.18
        inner_radius = side * 0.15
        outer_radius = side * 0.35
        head_len = side * 0.10

        p_up = QPolygonF([
            QPointF(center_x - arrow_base_width, center_y - inner_radius),
            QPointF(center_x + arrow_base_width, center_y - inner_radius),
            QPointF(center_x + arrow_base_width, center_y - outer_radius + head_len),
            QPointF(center_x + arrow_head_width, center_y - outer_radius + head_len),
            QPointF(center_x, center_y - outer_radius),
            QPointF(center_x - arrow_head_width, center_y - outer_radius + head_len),
            QPointF(center_x - arrow_base_width, center_y - outer_radius + head_len),
        ])
        
        transform = QTransform().translate(center_x, center_y).rotate(90).translate(-center_x, -center_y)
        
        painter.setBrush(QBrush(QColor("#34495E")))
        painter.setPen(QPen(QColor("#ECF0F1"), 2))

        painter.drawPolygon(p_up)
        painter.drawPolygon(transform.map(p_up))
        painter.drawPolygon(transform.map(transform.map(p_up)))
        painter.drawPolygon(transform.map(transform.map(transform.map(p_up))))

    def _process_mouse_move(self, pos):
        center = QPointF(self.width() / 2, self.height() / 2)
        control_radius = (min(self.width(), self.height()) / 2 - 5) * 0.9
        dead_zone_radius = (min(self.width(), self.height()) / 2 - 5) * 0.1
        
        vec = pos - center
        dist = math.hypot(vec.x(), vec.y())

        if dist < dead_zone_radius:
            self.mouse_pos = center
            self.vector_move_requested.emit(0.0, 0.0)
            return

        if dist > control_radius:
            vec = vec / dist * control_radius
            self.mouse_pos = center + vec
        else:
            self.mouse_pos = pos

        # Normalized distance from the edge of the dead zone to the control radius
        normalized_dist = (dist - dead_zone_radius) / (control_radius - dead_zone_radius)
        normalized_dist = max(0.0, min(1.0, normalized_dist))

        # Power-law speed calculation
        MIN_SPEED = 0.5  # arcsec/sec
        MAX_SPEED = 90.0 # arcsec/sec
        POWER = 2.0
        
        total_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (normalized_dist ** POWER)
        
        # Calculate vector components
        angle_rad = math.atan2(vec.y(), vec.x())
        d_ra_offset = total_speed * math.cos(angle_rad)
        d_dec_offset = total_speed * math.sin(angle_rad)

        # Apply transformations (rotation, flip)
        transform = QTransform().rotate(-self.angle).scale(1.0/self.flip_h, 1.0/self.flip_v)
        transformed_vec = transform.map(QPointF(d_ra_offset, d_dec_offset))

        final_d_ra = transformed_vec.x()
        final_d_dec = -transformed_vec.y() # Flip Y for telescope coordinates

        # Snap to axis
        snap_threshold = 5.0
        click_angle_deg = math.degrees(math.atan2(vec.y(), vec.x()))
        if -snap_threshold < click_angle_deg < snap_threshold or 180 - snap_threshold < abs(click_angle_deg) < 180 + snap_threshold:
            final_d_dec = 0
        elif 90 - snap_threshold < abs(click_angle_deg) < 90 + snap_threshold:
            final_d_ra = 0

        self.vector_move_requested.emit(final_d_ra, final_d_dec)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_slewing = True
            self.start_move_requested.emit()
            self._process_mouse_move(event.pos())

    def mouseMoveEvent(self, event):
        if self.is_slewing:
            self._process_mouse_move(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_slewing:
            self.is_slewing = False
            self.stop_requested.emit()
            self.update()

    def set_rotation(self, angle):
        self.angle = angle
        self.update()

    def toggle_flip_h(self):
        self.flip_h *= -1
        self.update()

    def toggle_flip_v(self):
        self.flip_v *= -1
        self.update()


class NavigatorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telescope Navigator")
        self.setGeometry(100, 100, 500, 600)
        
        # Define settings file path
        # Assumes the script is in the same directory as the settings file
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'navigator_settings.json')


        self.setStyleSheet("""
            QWidget {
                background-color: #212F3D;
                color: #ECF0F1;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498DB;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5DADE2;
            }
            QPushButton:pressed {
                background-color: #2E86C1;
            }
            QSlider::groove:horizontal {
                border: 1px solid #566573;
                height: 8px;
                background: #566573;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                border: 1px solid #3498DB;
                width: 18px;
                margin: -5px 0; 
                border-radius: 9px;
            }
            #StatusLabel {
                background-color: #17202A;
                border-radius: 5px;
                padding: 8px;
                font-size: 16px;
                font-family: 'monospace';
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        self.controller = TelescopeController()
        self.arrow_pad = ArrowPadWidget()
        main_layout.addWidget(self.arrow_pad)

        controls_layout = QVBoxLayout()
        rotation_layout = QHBoxLayout()
        self.rotation_label = QLabel("Rotation: 0°")
        rotation_layout.addWidget(self.rotation_label)
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.valueChanged.connect(self.on_rotation_changed)
        rotation_layout.addWidget(self.rotation_slider)
        controls_layout.addLayout(rotation_layout)

        flip_layout = QHBoxLayout()
        flip_h_button = QPushButton("Flip Horizontal")
        flip_h_button.clicked.connect(self.on_flip_h_clicked)
        flip_v_button = QPushButton("Flip Vertical")
        flip_v_button.clicked.connect(self.on_flip_v_clicked)
        flip_layout.addWidget(flip_h_button)
        flip_layout.addWidget(flip_v_button)
        controls_layout.addLayout(flip_layout)
        main_layout.addLayout(controls_layout)

        self.status_label = QLabel()
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.update_status_label()
        main_layout.addWidget(self.status_label)

        self.arrow_pad.start_move_requested.connect(self.controller.store_current_rate)
        self.arrow_pad.vector_move_requested.connect(self.on_vector_move_requested)
        self.arrow_pad.stop_requested.connect(self.on_stop_requested)
        
        self.load_settings()

    def on_rotation_changed(self, value):
        self.arrow_pad.set_rotation(value)
        self.rotation_label.setText(f"Rotation: {value}°")
        self.save_settings()

    def on_flip_h_clicked(self):
        self.arrow_pad.toggle_flip_h()
        self.save_settings()

    def on_flip_v_clicked(self):
        self.arrow_pad.toggle_flip_v()
        self.save_settings()

    def on_vector_move_requested(self, d_ra, d_dec):
        self.controller.move(d_ra, d_dec)
        self.update_status_label(d_ra, d_dec)

    def on_stop_requested(self):
        self.controller.stop()
        self.update_status_label()

    def update_status_label(self, d_ra=0.0, d_dec=0.0):
        mode = "REAL" if not self.controller.simulated else "SIMULATION"
        
        if d_ra == 0.0 and d_dec == 0.0:
            action = "IDLE"
        else:
            action = f"SLEWING (RA: {d_ra:+.2f}\", Dec: {d_dec:+.2f}\")"

        self.status_label.setText(f"MODE: {mode} | STATUS: {action}")

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.rotation_slider.setValue(settings.get('angle', 0))
                self.arrow_pad.flip_h = settings.get('flip_h', 1)
                self.arrow_pad.flip_v = settings.get('flip_v', 1)
                self.arrow_pad.update()
                print("Settings loaded.")
        except FileNotFoundError:
            print("Settings file not found. Using defaults.")
        except (json.JSONDecodeError, TypeError):
            print("Error reading settings file. Using defaults.")

    def save_settings(self):
        settings = {
            'angle': self.arrow_pad.angle,
            'flip_h': self.arrow_pad.flip_h,
            'flip_v': self.arrow_pad.flip_v,
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except IOError:
            print("Error: Could not save settings to file.")

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NavigatorWindow()
    window.show()
    sys.exit(app.exec_())
