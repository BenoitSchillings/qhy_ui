import math
from constants import DENSITY_PYREX, SPECIFIC_HEAT_PYREX, CONVECTIVE_COEFFICIENT_AIR

class Mirror:
    """
    Represents the telescope mirror and handles its thermal simulation.
    """
    def __init__(self, diameter_inch, thickness_inch):
        """
        Initializes the mirror with its physical properties.

        Args:
            diameter_inch (float): The diameter of the mirror in inches.
            thickness_inch (float): The thickness of the mirror in inches.
        """
        # Convert imperial units to metric for physics calculations
        self.diameter_m = diameter_inch * 0.0254
        self.thickness_m = thickness_inch * 0.0254
        self.radius_m = self.diameter_m / 2

        # Calculate physical properties
        self.volume_m3 = math.pi * (self.radius_m ** 2) * self.thickness_m
        self.mass_kg = self.volume_m3 * DENSITY_PYREX
        
        # Surface area includes the front, back, and edge of the cylindrical mirror
        self.surface_area_m2 = 2 * (math.pi * self.radius_m ** 2) + (2 * math.pi * self.radius_m * self.thickness_m)

        self.temperature_c = None
        self.last_temp_c = None
        self.cooling_rate_c_per_min = 0.0

        print(f"--- Mirror Initialized ---")
        print(f"Diameter: {self.diameter_m * 100:.1f} cm")
        print(f"Thickness: {self.thickness_m * 100:.1f} cm")
        print(f"Mass: {self.mass_kg:.2f} kg")
        print(f"Surface Area: {self.surface_area_m2:.3f} m^2")
        print(f"--------------------------")

    def set_initial_temperature(self, air_temp_c):
        """
        Sets the initial temperature of the mirror, assuming it starts warmer than the air.
        """
        self.temperature_c = air_temp_c + 5.0
        self.last_temp_c = self.temperature_c
        print(f"Initial mirror temperature set to: {self.temperature_c:.1f}°C")

    def update(self, air_temp_c, time_delta_seconds):
        """
        Updates the mirror's temperature based on Newton's Law of Cooling.

        Args:
            air_temp_c (float): The current ambient air temperature in Celsius.
            time_delta_seconds (float): The time elapsed since the last update in seconds.
        """
        if self.temperature_c is None:
            self.set_initial_temperature(air_temp_c)
            return

        # Store the current temperature for rate calculation
        self.last_temp_c = self.temperature_c

        # Newton's Law of Cooling: dQ/dt = h * A * (T_mirror - T_air)
        # This is the rate of heat energy loss in Joules per second (Watts).
        temp_difference = self.temperature_c - air_temp_c
        heat_loss_watts = CONVECTIVE_COEFFICIENT_AIR * self.surface_area_m2 * temp_difference

        # Calculate the temperature change based on the heat loss and the mirror's thermal mass
        # dT = (dQ/dt) * dt / (m * c)
        temp_change = (heat_loss_watts * time_delta_seconds) / (self.mass_kg * SPECIFIC_HEAT_PYREX)

        # Update the mirror's temperature
        self.temperature_c -= temp_change

        # Calculate the cooling rate for display purposes
        if time_delta_seconds > 0:
            rate_c_per_sec = (self.temperature_c - self.last_temp_c) / time_delta_seconds
            self.cooling_rate_c_per_min = rate_c_per_sec * 60
