print("--- SCRIPT EXECUTION STARTED ---")
import time
import os
from datetime import datetime

from temperature_scraper import TemperatureScraper
from mirror_model import Mirror

# --- Configuration ---
PWS_URL = "https://www.wunderground.com/dashboard/pws/KCAAUBER79"
UPDATE_INTERVAL_SECONDS = 300  # 5 minutes

def fahrenheit_to_celsius(f_temp):
    """Converts Fahrenheit to Celsius."""
    return (f_temp - 32) * 5.0 / 9.0

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_status(air_temp_c, mirror):
    """
    Displays the current air and mirror temperatures in a clean format.
    """
    clear_screen()
    print("-- Mirror Temperature Simulator --")
    print(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 32)
    
    if air_temp_c is not None:
        print(f"[+] Ambient Air Temp: {air_temp_c:.1f}°C")
    else:
        print("[+] Ambient Air Temp: Fetching...")

    if mirror.temperature_c is not None:
        status = "Cooling" if mirror.cooling_rate_c_per_min < 0 else "Warming"
        print(f"[+] Mirror Temp:      {mirror.temperature_c:.1f}°C ({status} at {abs(mirror.cooling_rate_c_per_min):.2f}°C/min)")
    else:
        print("[+] Mirror Temp:      Waiting for first air temp reading...")
        
    print("-" * 32)

def main():
    """
    Main application loop.
    """
    print("Simulator starting up...")
    scraper = TemperatureScraper(PWS_URL)
    mirror = Mirror(diameter_inch=20, thickness_inch=1.5)
    
    last_update_time = time.time()

    try:
        while True:
            # Fetch air temperature
            air_temp_f = scraper.get_current_temperature_fahrenheit()
            
            if air_temp_f is not None:
                air_temp_c = fahrenheit_to_celsius(air_temp_f)
                
                # Update mirror model
                current_time = time.time()
                time_delta = current_time - last_update_time
                last_update_time = current_time
                
                mirror.update(air_temp_c, time_delta)
                
                # Display the new status
                display_status(air_temp_c, mirror)
            else:
                print("Failed to fetch air temperature. Retrying in a moment...")

            print(f"Waiting for {UPDATE_INTERVAL_SECONDS} seconds until next update...")
            time.sleep(UPDATE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down.")
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
    finally:
        scraper.shutdown()

if __name__ == "__main__":
    main()
