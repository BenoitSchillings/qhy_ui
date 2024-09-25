import time
import math
import controller as xp

m1_fwd, m1_back = 11690, -16450
m2_fwd, m2_back = 13080, -13800
m3_fwd, m3_back = 11700, -13050
s1 = s2 = s3 = 0

def scale_motor_value(value, forward_max, backward_max):
    if value >= 0:
        return int((value / 1000) * forward_max)
    else:
        return int((value / 1000) * abs(backward_max))

def move_low_level(v1, v2, v3):
    global s1, s2, s3
    scaled_v1 = scale_motor_value(v1, m1_fwd, m1_back)
    scaled_v2 = scale_motor_value(v2, m2_fwd, m2_back)
    scaled_v3 = scale_motor_value(v3, m3_fwd, m3_back)
    print(f"Motor movements: {scaled_v1}, {scaled_v2}, {scaled_v3}")
    #xp.move3(1100, scaled_v1, scaled_v2, scaled_v3)
    s1 += scaled_v1
    s2 += scaled_v2
    s3 += scaled_v3
    print(f"Total movements: {s1}, {s2}, {s3}, Sum: {s1+s2+s3}")

class AOController:
    def __init__(self):
        self.initialize_hardware()
        self.ax = 0
        self.ay = 0

    def initialize_hardware(self):
        xp.init_pico()

    def restart_hardware(self):
        xp.init_pico()

    def home(self):
        self.restart_hardware()
        self.goto(0, 0)

    def clip_position(self, value):
        MAX = 200  # Adjust as needed
        return max(-MAX, min(MAX, value))

    def calculate_motor_movements(self, angle_x, angle_y):
        angle_x_rad = math.radians(angle_x * 10.0)
        angle_y_rad = math.radians(angle_y * 10.0)
        
        # Calculate the tilt vector
        tilt_x = math.sin(angle_x_rad)
        tilt_y = math.sin(angle_y_rad)
        
        # Calculate motor movements
        m1 = -tilt_x
        m2 = 0.5 * tilt_x - 0.866 * tilt_y
        m3 = 0.5 * tilt_x + 0.866 * tilt_y
        
        # Normalize to ensure the sum of movements is zero
        avg_movement = (m1 + m2 + m3) / 3
        m1 -= avg_movement
        m2 -= avg_movement
        m3 -= avg_movement
        
        # Scale the movements to ensure they sum to zero
        total = abs(m1) + abs(m2) + abs(m3)
        if total != 0:
            scale = 3000 / total  # Adjust this value as needed
            m1 *= scale
            m2 *= scale
            m3 *= scale
        
        return m1, m2, m3

    def move_relative(self, dx, dy):
        dx = self.clip_position(dx)
        dy = self.clip_position(dy)
        m1, m2, m3 = self.calculate_motor_movements(dx, dy)
        move_low_level(m1, m2, m3)
        self.ax += dx
        self.ay += dy

    def get_position(self):
        return self.ax, self.ay

    def goto(self, x, y):
        dx = x - self.ax 
        dy = y - self.ay
        print(f"Moving to: {x}, {y}, Delta: {dx}, {dy}")
        self.move_relative(dx, dy)

    def zero(self):
        self.goto(0, 0)

    def move(self, x, y):
        self.move_relative(x, y)

    def circle_test(self, speed, diameter):
        alpha = 0.0
        while True:
            alpha += speed
            x = math.sin(alpha) * diameter
            y = math.cos(alpha) * diameter
            self.goto(int(round(x)), int(round(y)))
            time.sleep(0.02)

    def close(self):
        print("Shutting down hardware...")
        # Add any necessary shutdown procedures here

if __name__ == "__main__":
    controller = AOController()
    try:
        print("Starting circle test...")
        controller.circle_test(speed=0.01, diameter=100)
    except KeyboardInterrupt:
        print("\nCircle test interrupted.")
    finally:
        controller.close()